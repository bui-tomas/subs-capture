import json
import os
import asyncio
import random
import cv2
import numpy as np
from playwright.async_api import async_playwright, Page, ElementHandle
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

SUBS_PATH = os.getenv('SUBS_PATH')

class SubtitleCalibrator:
    def __init__(self, subs_path: str = None):
        self.url = None
        self.subs_path = SUBS_PATH if SUBS_PATH else subs_path
        self.screenshot_folder = f'screenshots/{SUBS_PATH.split('/')[-1].split('.')[0]}'
        os.makedirs(self.screenshot_folder, exist_ok=True)
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='ch',
        )

        self.video_width = None
        self.video_height = None
        self.subtitle_region = None

        self.sample = None
        self.ad_trigger = -1
        self.ad_offset = 0
    
    def calibrate(self):
        self._load_metadata(self.subs_path)

        screenshots = asyncio.run(self._playwright_pipeline())
        
        # Load semantic model for tie-breaking
        model = SentenceTransformer('sentence-transformers/LaBSE')
        
        # Group screenshots by offset
        offset_groups = {}
        for (offset, idx, screenshot) in screenshots:
            if offset not in offset_groups:
                offset_groups[offset] = []
            offset_groups[offset].append((idx, screenshot))
        
        # Score each offset
        offset_scores = {}
        for offset, screenshot_list in offset_groups.items():
            successful_ocrs = 0
            semantic_scores = []
            
            for idx, screenshot in screenshot_list:
                sub = self.sample[idx]
                cn_text, metadata = self._ocr_subs(sub, [(offset, screenshot)])
                
                if cn_text:  # OCR succeeded
                    successful_ocrs += 1
                    
                    # Calculate semantic similarity if English text exists
                    if sub.get('text'):
                        cn_embedding = model.encode([cn_text])
                        en_embedding = model.encode([sub['text']])
                        similarity = cosine_similarity(cn_embedding, en_embedding).flatten()[0]
                        semantic_scores.append(similarity)
            
            avg_semantic = np.mean(semantic_scores) if semantic_scores else 0.0
            offset_scores[offset] = {
                'count': successful_ocrs,
                'semantic_avg': avg_semantic
            }
        
        # Find winner: max count, or if tie/one-less then best semantic
        max_count = max(s['count'] for s in offset_scores.values())
        
        candidates = []
        for offset, scores in offset_scores.items():
            if scores['count'] == max_count or scores['count'] == max_count - 1:
                candidates.append((offset, scores))
        
        # Pick best semantic among candidates
        winner_offset = max(candidates, key=lambda x: x[1]['semantic_avg'])[0]
        
        for offset, scores in sorted(offset_scores.items()):
            print(f"Offset {offset:+.2f}s: {scores['count']}/{len(self.sample)} OCRs, avg semantic: {scores['semantic_avg']:.3f}")
        
        return winner_offset

    async def _playwright_pipeline(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch_persistent_context(
                user_data_dir='/tmp/playwright-chrome',
                headless=False,
                channel='chrome',
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-automation',
                    '--disable-infobars',
                ],
                viewport={'width': 1920, 'height': 1080}
            )
            page = browser.pages[0] if browser.pages else await browser.new_page()
        
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)
            
            print(f'\n📺 Opening: {self.url}')
            await page.goto(self.url, wait_until='networkidle')

            video = await page.query_selector('#video_player')
            
            # Start playback
            await video.evaluate('v => v.play()')
            
            # Wait for playback to start
            print('⏳ Waiting for playback to start...')
            await page.wait_for_function(
                'document.querySelector("#video_player").currentTime > 0',
                timeout=10000
            )
            
            # Get video duration and calculate ad trigger point
            duration = await video.evaluate('v => v.duration')
            self.ad_trigger = (duration / 2) - 30

            await asyncio.sleep(10)
            
            # Set subtitle region
            await self._set_dimensions(video)
            await self._show_overlay(page, video)

            subs = self._load_subs(self.subs_path)
            offsets = self._get_offset(self.ad_offset)

            screenshots = await self._collect_screenshots(video=video, subs=subs, offsets=offsets)

            await browser.close()
            return screenshots

    def _get_offset(self, initial: float, num_steps=5) -> list[float]:
        offsets = [initial]
        arr1 = np.linspace(initial, initial + 1, num_steps)
        arr2 = np.linspace(initial - 1, initial, num_steps)
        offsets.extend(arr1)
        offsets.extend(arr2)
        return sorted(set(offsets))

    async def _collect_screenshots(self, 
        video: ElementHandle, 
        subs: list[dict],
        offsets: list
    ) -> list[tuple[int, dict, list[tuple[float, bytes]]]]:
        '''
        Single-pass screenshot collection with 6 samples per subtitle window
        '''
        
        def get_samples(subs):
            random.seed(42)
            return random.sample(subs, 20)

        async def seek_to_timestamp(video, target_time):
            '''Seek to timestamp and wait for seek to complete'''
            await video.evaluate(f'''
                v => new Promise(resolve => {{
                    if (Math.abs(v.currentTime - {target_time}) < 0.1) {{
                        resolve();
                    }} else {{
                        v.currentTime = {target_time};
                        v.addEventListener('seeked', () => resolve(), {{ once: true }});
                    }}
                }})
            ''')
            await asyncio.sleep(0.2)

        screenshots = []
        self.sample = get_samples(subs)
        
        for sub in self.sample:
            print(sub['start'])

        for offset in offsets:
            for idx, sub in enumerate(self.sample):
                timestamp = (sub['start'] + sub['end']) / 2 + offset
                await seek_to_timestamp(video, timestamp)
                screenshot = await video.screenshot()
                with open(f'{self.screenshot_folder}/sub_{sub['start']}s_offset_{offset:+.2f}s.png', 'wb') as f:
                    f.write(screenshot)
                screenshots.append((offset, idx, screenshot))

        return screenshots

    def _ocr_subs(self, sub: dict, screenshot_list: list[tuple[float, bytes]], is_lyrics=False) -> tuple[str, dict]:
        '''Worker function that uses global OCR instance'''
        
        ocr_results = []
        text_scores = {}  # Maps text -> list of confidence scores
        
        for offset, screenshot in screenshot_list:
            nparr = np.frombuffer(screenshot, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Extract appropriate region
            if not is_lyrics:
                y1, y2, x1, x2 = self.subtitle_region
            else:
                y1, y2, x1, x2 = self.right_lyrics_region
            
            subtitle_img = img[y1:y2, x1:x2]
            
            # Run OCR
            result = self.ocr.predict(subtitle_img)
            
            if not result or len(result) == 0:
                continue
            
            ocr_result = result[0]
            texts = ocr_result['rec_texts']
            conf_scores = ocr_result['rec_scores']
        
            if not texts:
                continue
            
            # Filter texts by confidence and collect scores per text
            filtered_texts = []
            filtered_scores = []
            for text, conf in zip(texts, conf_scores):
                filtered_texts.append(text)
                filtered_scores.append(conf)
            
            if filtered_texts:
                combined_text = ''.join(filtered_texts)
                avg_score = np.mean(filtered_scores)
                
                # Track this text and its score
                if combined_text not in text_scores:
                    text_scores[combined_text] = []
                text_scores[combined_text].append(avg_score)
                
                ocr_results.append((offset, combined_text))
        
        if not ocr_results:
            return '', {'variants': '', 'confidences': []}
        
        ocr_results.sort(key=lambda x: x[0])
        
        # Deduplicate while tracking variants and their confidences
        unique_texts = []
        variant_confidences = []
        seen = set()
        
        for offset, text in ocr_results:
            if text not in seen:
                unique_texts.append(text)
                seen.add(text)
                # Use the average confidence for THIS specific variant
                variant_confidences.append(np.mean(text_scores[text]))
        
        metadata = {
            'variants': ';'.join(unique_texts),
            'confidences': [round(conf, 3) for conf in variant_confidences]
        }

        return ''.join(unique_texts), metadata

    async def _set_dimensions(self, video):
        screenshot = await video.screenshot()

        if self.video_width is None:
            nparr = np.frombuffer(screenshot, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.video_height, self.video_width = img.shape[:2]

        h, w = self.video_height, self.video_width

        self.subtitle_region = (int(h * 0.76), int(h * 0.89), int(w * 0.27), int(w * 0.73))
        self.left_lyrics_region = (int(h * 0.2), int(h * 0.65), int(w * 0.05), int(w * 0.085))
        self.right_lyrics_region = (int(h * 0.2), int(h * 0.65), int(w * 0.88), int(w * 0.93))

    def _load_subs(self, file_path: str):
        '''
        Loads subtitles from JSON file and apply time offset
        Args:
            file_path: path to JSON file with subtitles
            time_offset: seconds to subtract from all timestamps
        Returns:
            List of subtitle dicts with adjusted timestamps
        '''

        def is_lyrics(text: str) -> bool:
            return '♪' in text

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        subtitles = []
        for sub in data['subtitles']:
            if is_lyrics(sub['text']):
                continue
            if sub['start'] < (self.ad_trigger - 30):      
                subtitles.append({
                    'start': sub['start'],
                    'end': sub['end'],
                    'duration': sub['end'] - sub['start'],
                    'text': sub['text']
                })
        
        return subtitles

    def _load_metadata(self, file_path: str):    
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.ad_offset = data['ad_offset']
        self.url = data['target_url']

    async def _show_overlay(self, page: Page, video: ElementHandle):
        '''
        Draws a yellow border around detected subtitle region
        '''

        # Get video position on page
        box = await video.bounding_box()

        y1, y2, x1, x2 = self.subtitle_region
        ly1, ly2, lx1, lx2 = self.left_lyrics_region
        ry1, ry2, rx1, rx2 = self.right_lyrics_region
        
        # Inject overlay div
        await page.evaluate(f'''
                // Subtitle overlay (yellow)
                const overlay = document.createElement('div');
                overlay.id = 'subtitle-debug-overlay';
                overlay.style.cssText = `
                    position: fixed;
                    left: {box['x'] + x1}px;
                    top: {box['y'] + y1}px;
                    width: {x2 - x1}px;
                    height: {y2 - y1}px;
                    border: 3px solid yellow;
                    pointer-events: none;
                    z-index: 99999;
                `;
                document.body.appendChild(overlay);
                
                // Left lyrics overlay (blue)
                const leftLyrics = document.createElement('div');
                leftLyrics.id = 'left-lyrics-overlay';
                leftLyrics.style.cssText = `
                    position: fixed;
                    left: {box['x'] + lx1}px;
                    top: {box['y'] + ly1}px;
                    width: {lx2 - lx1}px;
                    height: {ly2 - ly1}px;
                    border: 3px solid blue;
                    pointer-events: none;
                    z-index: 99999;
                `;
                document.body.appendChild(leftLyrics);
                
                // Right lyrics overlay (green)
                const rightLyrics = document.createElement('div');
                rightLyrics.id = 'right-lyrics-overlay';
                rightLyrics.style.cssText = `
                    position: fixed;
                    left: {box['x'] + rx1}px;
                    top: {box['y'] + ry1}px;
                    width: {rx2 - rx1}px;
                    height: {ry2 - ry1}px;
                    border: 3px solid green;
                    pointer-events: none;
                    z-index: 99999;
                `;
                document.body.appendChild(rightLyrics);
            ''')

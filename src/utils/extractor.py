import json
import logging
import os
import asyncio
import cv2
import numpy as np
from playwright.async_api import async_playwright, Page, ElementHandle
from paddleocr import PaddleOCR
from pypinyin import lazy_pinyin, Style
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from concurrent.futures import ProcessPoolExecutor

load_dotenv()

SUBS_PATH = os.getenv('SUBS_PATH')
SKIP_SONGS = True
NUM_WORKERS = 4
AD_BUFFER = 30

_worker_ocr = None
_worker_regions = None

def _init_worker(subtitle_region, left_lyrics_region, right_lyrics_region):
    '''Initialize OCR once per worker process'''
    global _worker_ocr, _worker_regions

    logging.getLogger('ppocr').setLevel(logging.ERROR)
    logging.getLogger('paddlex').setLevel(logging.ERROR)

    _worker_ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='ch',
    )
    _worker_regions = {
        'subtitle': subtitle_region,
        'left_lyrics': left_lyrics_region,
        'right_lyrics': right_lyrics_region
    }

def _ocr_subs(sub: dict, screenshot_list: list[tuple[float, bytes]]) -> tuple[str, dict]:
    '''Worker function that uses global OCR instance'''
    global _worker_ocr, _worker_regions
    
    ocr_results = []
    text_scores = {}  # Maps text -> list of confidence scores
    
    for offset, screenshot in screenshot_list:
        nparr = np.frombuffer(screenshot, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract appropriate region
        if sub['is_lyrics']:
            if sub['start'] < 120:
                y1, y2, x1, x2 = _worker_regions['left_lyrics']
            else:
                y1, y2, x1, x2 = _worker_regions['right_lyrics']
        else:
            y1, y2, x1, x2 = _worker_regions['subtitle']
        
        subtitle_img = img[y1:y2, x1:x2]
        
        # Run OCR
        result = _worker_ocr.predict(subtitle_img)
        
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

class SubtitleExtractor:
    def __init__(self, url: str = None, subs_path: str = None):
        self.url = url
        self.subs_path = SUBS_PATH if SUBS_PATH else subs_path
        self.screenshot_folder = f'screenshots/{SUBS_PATH.split('/')[-1].split('.')[0]}'
        os.makedirs(self.screenshot_folder, exist_ok=True)
        self.executor = None

        self.video_width = None
        self.video_height = None
        self.subtitle_region = None 
        self.left_lyrics_region = None
        self.right_lyrics_region = None

        self.subtitles = []
        self.ad_offset = 0
        self.offset_start = None

    def extract(self, file_dir: str = None):
        try:
            self._load_subs(self.subs_path)

            if file_dir:
                # Load screenshots from disk
                self._init_executor_from_disk(file_dir)
                screenshots = self._load_screenshots_from_disk(file_dir)
            else:
                # Normal pipeline
                screenshots = asyncio.run(self._playwright_pipeline())

            results = asyncio.run(self._process_screenshots(screenshots))
            self._process_and_save(results, self.subtitles)
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)
                print('🧹 Cleaned up worker processes')

    def _load_screenshots_from_disk(self, file_dir: str) -> list[tuple[int, dict, list[tuple[float, bytes]]]]:
        '''Load pre-captured screenshots from disk'''
        import glob
        from pathlib import Path
        
        screenshots_by_idx = {}
        self.subtitles = self._process_subs(time_offset=self.ad_offset)
        
        # Pattern: sub_{start}s_offset_{offset}s.png
        for img_path in glob.glob(f'{file_dir}/sub_*.png'):
            filename = Path(img_path).stem
            # Parse: sub_123.45s_offset_+0.50s
            parts = filename.split('_')
            start_time = float(parts[1].rstrip('s'))
            offset = float(parts[3].rstrip('s'))
            
            # Find matching subtitle index
            for idx, sub in enumerate(self.subtitles):
                if abs(sub['start'] - start_time) < 0.01:  # Match within 10ms
                    if idx not in screenshots_by_idx:
                        screenshots_by_idx[idx] = []
                    
                    with open(img_path, 'rb') as f:
                        screenshot_bytes = f.read()
                    screenshots_by_idx[idx].append((offset, screenshot_bytes))
                    break
        
        # Convert to expected format
        return [(idx, self.subtitles[idx], screenshots_by_idx[idx]) 
                for idx in sorted(screenshots_by_idx.keys())]

    def _init_executor_from_disk(self, file_dir: str):
        '''Initialize executor using dimensions from existing screenshots'''
        import glob
        
        # Load first screenshot to get dimensions
        first_img = glob.glob(f'{file_dir}/sub_*.png')[0]
        img = cv2.imread(first_img)
        self.video_height, self.video_width = img.shape[:2]
        
        h, w = self.video_height, self.video_width
        self.subtitle_region = (int(h * 0.76), int(h * 0.89), int(w * 0.27), int(w * 0.73))
        self.left_lyrics_region = (int(h * 0.2), int(h * 0.65), int(w * 0.05), int(w * 0.085))
        self.right_lyrics_region = (int(h * 0.2), int(h * 0.65), int(w * 0.88), int(w * 0.93))
        
        self.executor = ProcessPoolExecutor(
            max_workers=NUM_WORKERS,
            initializer=_init_worker,
            initargs=(self.subtitle_region, self.left_lyrics_region, self.right_lyrics_region)
        )

    def _load_subs(self, file_path: str):    
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.ad_offset = data['ad_offset']
        self.url = data['target_url']
        self.subtitles = data['subtitles']
        self.offset_start = data['offset_start'] if data['offset_start'] != 'None' else None

    async def _set_dimensions(self, video: ElementHandle):
        screenshot = await video.screenshot()

        if self.video_width is None:
            nparr = np.frombuffer(screenshot, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.video_height, self.video_width = img.shape[:2]

        h, w = self.video_height, self.video_width

        self.subtitle_region = (int(h * 0.76), int(h * 0.89), int(w * 0.27), int(w * 0.73))
        self.left_lyrics_region = (int(h * 0.2), int(h * 0.65), int(w * 0.05), int(w * 0.085))
        self.right_lyrics_region = (int(h * 0.2), int(h * 0.65), int(w * 0.88), int(w * 0.93))

        self.executor = ProcessPoolExecutor(
            max_workers=NUM_WORKERS,
            initializer=_init_worker,
            initargs=(self.subtitle_region, self.left_lyrics_region, self.right_lyrics_region)
        )

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

    def _process_subs(self, time_offset=0):
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

        apply_offset = self.offset_start is None or self.offset_start.lower() == "none"
        
        subtitles = []
        for sub in self.subtitles:
            if apply_offset:
                subtitles.append({
                    'start': sub['start'] + time_offset,
                    'end': sub['end'] + time_offset,
                    'offset_add': sub.get('offset_add', 0),
                    'duration': sub['end'] - sub['start'],
                    'is_lyrics': is_lyrics(sub['text']),
                    'text': sub['text']
                })
            else:
                subtitles.append({
                    'start': sub['start'],
                    'end': sub['end'],
                    'offset_add': sub.get('offset_add', 0),
                    'duration': sub['end'] - sub['start'],
                    'is_lyrics': is_lyrics(sub['text']),
                    'text': sub['text']
                })

            if sub['text'] == self.offset_start:
                apply_offset = True
        
        for sub in subtitles:
            offset_add = sub.get('offset_add')
            if offset_add != 0:
                print('Margin: ', float(offset_add), sub['start'])

        return subtitles

    async def _collect_screenshots(self, 
        video: ElementHandle, 
        subtitles: list[dict],
        start_index: int = None
    ) -> list[tuple[int, dict, list[tuple[float, bytes]]]]:
        '''
        Single-pass screenshot collection with 6 samples per subtitle window
        '''
        def get_offset(duration: float, is_lyrics=False, num_steps=2, overlap=0.9) -> list[float]:
            if is_lyrics:
                return [duration / 2 * 0.65, duration / 2 * 0.75]
            else:
                arr = np.linspace(0, duration, num_steps + 1)[1:]
                arr[-1] *= overlap
                offsets = [0]
                offsets.extend([x for val in arr for x in (val, -val)])
                offsets = [x / 2 for x in offsets]
            return sorted(offsets)
        
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
            await asyncio.sleep(0.1)

        screenshots = []
        ad_buffer = []

        # Track screenshots by index
        screenshots_by_idx = {}
        duration = await video.evaluate('v => v.duration')
        ad_breaks = [(duration * 0.5, False), (duration * 0.75, False)]  # (timestamp, played)
        margin = 0

        for idx, sub in enumerate(subtitles):
            if idx != start_index and start_index is not None:
                continue

            # if idx == 0:
            #     print('Sleeping')
            #     await seek_to_timestamp(video, 2200)
            #     await asyncio.sleep(40)

            screenshots_by_idx[idx] = []  # Initialize
            offsets = get_offset(sub['duration'], sub['is_lyrics'])
            print(f'Timestamp: {sub['start']}')

            offset_add = sub.get('offset_add')
            if offset_add != 0:
                print('Margin: ', float(offset_add), sub['start'])
                margin = offset_add

            for offset in offsets:
                timestamp = (sub['start'] + sub['end']) / 2 + offset + margin
                
                if ((timestamp <= 120 or timestamp >= duration - 180) 
                    and SKIP_SONGS and sub['is_lyrics']):
                    continue

                # Check ad breaks
                for i, (ad_time, played) in enumerate(ad_breaks):
                    if ad_time - AD_BUFFER <= timestamp < ad_time and not played:
                        ad_buffer.append((idx, offset, timestamp))
                        break
                    
                    if timestamp >= ad_time and not played:
                        ad_breaks[i] = (ad_time, True)
                        await seek_to_timestamp(video, timestamp)
                        
                        # Capture buffered screenshots
                        for buf_idx, buf_offset, buf_ts in ad_buffer:
                            await seek_to_timestamp(video, buf_ts)
                            screenshot = await video.screenshot()
                            with open(f'{self.screenshot_folder}/sub_{sub['start']}s_offset_{offset:+.2f}s.png', 'wb') as f:
                                f.write(screenshot)
                            screenshots_by_idx[buf_idx].append((buf_offset, screenshot))
                        
                        ad_buffer.clear()
                        break
                else:
                    # Only runs if NO break was hit (normal capture)
                    await seek_to_timestamp(video, timestamp)
                    screenshot = await video.screenshot()
                    with open(f'{self.screenshot_folder}/sub_{sub['start']}s_offset_{offset:+.2f}s.png', 'wb') as f:
                        f.write(screenshot)                    
                    screenshots_by_idx[idx].append((offset, screenshot))

        # Convert dict back to list format
        screenshots = [(idx, subtitles[idx], screenshots_by_idx[idx]) 
                    for idx in sorted(screenshots_by_idx.keys())]

        return screenshots

    async def _process_screenshots(
        self, 
        screenshots: list[tuple[int, dict, list[tuple[float, bytes]]]]
    ) -> list[tuple[int, str]]:
        '''
        Parallel OCR processing. ProcessPoolExecutor handles concurrency limit.
        '''
        async def ocr_task(idx: int, sub: dict, screenshot_list: list[tuple[float, bytes]]) -> tuple[int, str]:
            loop = asyncio.get_event_loop()
            text, conf = await loop.run_in_executor(
                self.executor,
                _ocr_subs, 
                sub, 
                screenshot_list
            )
            return (idx, text, conf)
        
        # Filter out None entries but create tasks with original indices
        tasks = [
            ocr_task(idx, sub, ss_list) 
            for item in screenshots 
            if item is not None
            for idx, sub, ss_list in [item]
        ]
        
        results = []
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc='OCR Processing', unit='subtitle'):
        # for coro in tasks:
            result = await coro
            results.append(result)
        
        return results

    def _process_and_save(self, ocr_results: list[tuple[int, str, dict]], subtitles: list[dict]):
        '''
        Convert Chinese text to pinyin and save final subtitle file
        '''
        # Create a map: index -> (chinese text, metadata)
        results_map = {idx: (cn_text, metadata) for idx, cn_text, metadata in ocr_results}
        
        final_subtitles = []
        failed_segments = []  # Track failed OCR segments
        
        for i, sub in enumerate(subtitles):
            result = results_map.get(i)
            
            if result:
                cn_text, metadata = result
                # Convert to pinyin
                pinyin_text = ' '.join(lazy_pinyin(cn_text, style=Style.TONE))
                
                final_subtitles.append({
                    'start': sub['start'],
                    'end': sub['end'],
                    'hanzi': cn_text,
                    'pinyin': pinyin_text,
                    'english': sub['text'],
                    'metadata': metadata
                })
                
                # Track failed OCR (non-lyrics only)
                if not cn_text and not sub['is_lyrics']:
                    failed_segments.append(sub)
            else:
                # OCR failed or no screenshot, still include the entry
                final_subtitles.append({
                    'start': sub['start'],
                    'end': sub['end'],
                    'hanzi': '',
                    'pinyin': '',
                    'english': sub['text'],
                    'metadata': {'variants': '', 'confidences': []}
                })
                
                # Track failed OCR (non-lyrics only)
                if not sub['is_lyrics']:
                    failed_segments.append(sub)
        
        # Base filename from env
        base_name = os.getenv('SUBS_PATH').split('/')[-1].split('.')[0]
        
        # Save to JSON
        output_data = {
            'subtitles': final_subtitles,
            'ad_offset': self.ad_offset,
            'offset_start': self.offset_start
        }
        
        with open(f'{base_name}_raw.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Save failed segments to text file
        if failed_segments:
            with open(f'{base_name}_errors.txt', 'w', encoding='utf-8') as f:
                f.write('Failed OCR segments (start, end):\n')
                for sub in failed_segments:
                    f.write(f'{sub['start']}, {sub['end']}, {sub['text']}\n')

            print(f'⚠️  Saved {len(failed_segments)} failed segments to {base_name}_errors.txt')
        
        # Calculate success rate (excluding lyrics)
        non_lyrics_subs = [s for s in final_subtitles if '♪' not in s['english']]
        success_count = sum(1 for s in non_lyrics_subs if s['hanzi'])
        total_non_lyrics = len(non_lyrics_subs)
        
        print(f'\nSaved {len(final_subtitles)} subtitles to {base_name}_raw.json')
        print(f'OCR success rate (non-lyrics): {success_count}/{total_non_lyrics} ({success_count/total_non_lyrics*100:.1f}%)')

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

            # Find and click the button to turn off comments
            button = await page.query_selector(os.getenv('BUTTON_SELECTOR'))
            if button:
                await button.click()
            
            video = await page.query_selector('#video_player')
            
            # Start playback
            await video.evaluate('v => v.play()')
            
            # Wait for playback to start
            await page.wait_for_function(
                'document.querySelector("#video_player").currentTime > 0',
                timeout=10000
            )

            # Click the fullscreen button (vg-fullscreen element)
            await page.evaluate('''
                () => {
                    const fullscreenBtn = document.querySelector('vg-fullscreen div[role="button"]');
                    if (fullscreenBtn) {
                        fullscreenBtn.click();
                    }
                }
            ''')
            
            # video = await page.wait_for_selector('mat-video video.video', timeout=10000)

            # # Click the fullscreen button
            # fullscreen_btn = await page.query_selector('mat-fullscreen-button button')
            # if fullscreen_btn:
            #     await fullscreen_btn.click()
            #     await asyncio.sleep(1)

            # # Play the video
            # await video.evaluate('v => v.play()')

            # print('⏳ Waiting for playback to start...')
            # await page.wait_for_function(
            #     'document.querySelector("mat-video video.video").currentTime > 0',
            #     timeout=10000
            # )

            # Set subtitle region
            await self._set_dimensions(video)
            # await self._show_overlay(page, video)
            
            # Load subtitles
            self.subtitles = self._process_subs(time_offset=self.ad_offset)
            
            # Collect screenshots
            screenshots = await self._collect_screenshots(video, self.subtitles)
            await browser.close()    
            
            return screenshots

import json
import re
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

AD_OFFSET = 18.8
SKIP_SONGS = True

_worker_ocr = None
_worker_regions = None

def _init_worker(subtitle_region, left_lyrics_region, right_lyrics_region):
    '''Initialize OCR once per worker process'''
    global _worker_ocr, _worker_regions
    _worker_ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='ch',
    )
    _worker_regions = {
        'subtitle': subtitle_region,
        'left_lyrics': left_lyrics_region,
        'right_lyrics': right_lyrics_region
    }

def _ocr_subs(sub: dict, screenshot_list: list[tuple[float, bytes]], brightness_threshold: int = 210, similar_threshold: int = 100, pixel_threshold: int = 25) -> tuple[str, float]:
    '''Worker function that uses global OCR instance'''
    global _worker_ocr, _worker_regions
    
    ocr_results = []
    all_scores = []
    reference_grey = None
    reference_has_text = False
    
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

        # Apply brightness threshold filtering
        if brightness_threshold:
            subtitle_img = cv2.bitwise_and(
                subtitle_img,
                subtitle_img,
                mask=cv2.inRange(
                    subtitle_img,
                    (brightness_threshold, brightness_threshold, brightness_threshold),
                    (255, 255, 255)
                )
            )

        # Convert to greyscale for similarity checking
        grey = cv2.cvtColor(subtitle_img, cv2.COLOR_BGR2GRAY)
        
        # Set reference on first iteration
        if reference_grey is None:
            reference_grey = grey
        else:
            # Only do similarity check if reference had text
            if reference_has_text:
                _, absdiff = cv2.threshold(
                    cv2.absdiff(reference_grey, grey),
                    pixel_threshold,
                    255,
                    cv2.THRESH_BINARY
                )
                
                if np.count_nonzero(absdiff) < similar_threshold:
                    print(f"Skipping offset {offset:.2f}s - similar to reference")
                    continue
        
        # Run OCR
        result = _worker_ocr.predict(subtitle_img)
        
        if not result or len(result) == 0:
            continue
        
        ocr_result = result[0]
        texts = ocr_result['rec_texts']
        conf_scores = ocr_result['rec_scores']
        
        if not texts:
            continue
        
        # Mark that reference has text (first successful OCR)
        if reference_grey is not None and not reference_has_text:
            reference_has_text = True
        
        # Filter texts by confidence and collect scores
        filtered_texts = []
        for text, conf in zip(texts, conf_scores):
            if conf > 0.7:
                filtered_texts.append(text)
                all_scores.append(conf)
        
        if filtered_texts:
            combined_text = ''.join(filtered_texts)
            ocr_results.append((offset, combined_text))
    
    if not ocr_results:
        return '', 0.0
    
    ocr_results.sort(key=lambda x: x[0])
    
    # Deduplicate
    unique_texts = []
    seen = set()
    for offset, text in ocr_results:
        if text not in seen:
            unique_texts.append(text)
            seen.add(text)

    return ''.join(unique_texts), np.mean(all_scores) if all_scores else 0.0

class SubtitleExtractor:
    def __init__(self, url: str, subs_path: str):
        self.subtitles = []
        self.url = url
        self.subs_path = subs_path
        self.executor = None

        self.video_width = None
        self.video_height = None
        self.subtitle_region = None 
        self.left_lyrics_region = None
        self.right_lyrics_region = None

    async def set_dimensions(self, video: ElementHandle):
        screenshot = await video.screenshot()

        if self.video_width is None:
            nparr = np.frombuffer(screenshot, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.video_height, self.video_width = img.shape[:2]

        h, w = self.video_height, self.video_width

        self.subtitle_region = (int(h * 0.76), int(h * 0.83), int(w * 0.30), int(w * 0.70))
        self.left_lyrics_region = (int(h * 0.2), int(h * 0.65), int(w * 0.05), int(w * 0.085))
        self.right_lyrics_region = (int(h * 0.2), int(h * 0.65), int(w * 0.88), int(w * 0.93))

        self.executor = ProcessPoolExecutor(
            max_workers=5,
            initializer=_init_worker,
            initargs=(self.subtitle_region, self.left_lyrics_region, self.right_lyrics_region)
        )

    async def show_overlay(self, page: Page, video: ElementHandle):
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

    def load_subs(self, file_path: str, time_offset=0):
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
        
        def is_long(start: float, end: float) -> bool:
            return start + end > 2.0

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        apply_offset = False  # Flag to track when to start applying offset
        
        subtitles = []
        for sub in data['subtitles']:
            if apply_offset:
                subtitles.append({
                    'start': sub['start'] + time_offset,
                    'end': sub['end'] + time_offset,
                    'duration': sub['end'] - sub['start'],
                    'is_lyrics': is_lyrics(sub['text']),
                    'is_long': is_long(sub['start'], sub['end']),
                    'text': sub['text']
                })
            else:
                subtitles.append({
                    'start': sub['start'],
                    'end': sub['end'],
                    'duration': sub['end'] - sub['start'],
                    'is_lyrics': is_lyrics(sub['text']),
                    'is_long': is_long(sub['start'], sub['end']),
                    'text': sub['text']
                })

            if sub['text'] == '=Episode 1=':
                apply_offset = True
        
        return subtitles

    async def collect_screenshots(self, 
        video: ElementHandle, 
        subtitles: list[dict]
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
            return offsets
        
        screenshots = []

        duration = await video.evaluate('v => v.duration')

        for idx, sub in enumerate(subtitles):
            temp =[]
            # Dynamic offset range  
            offsets = get_offset(sub['duration'], sub['is_lyrics'])

        for offset in offsets:
            timestamp = (sub['start'] + sub['end']) / 2 + offset
            if (timestamp <= 120 or timestamp >= duration - 180) and SKIP_SONGS:
                continue
                
            await video.evaluate(f'v => v.currentTime = {timestamp}')
            await asyncio.sleep(0.2)
            
            screenshot = await video.screenshot()
            temp.append((offset, screenshot))
                
            screenshots.append((idx, sub, temp))
            
        return screenshots 

    async def process_screenshots(
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
            result = await coro
            results.append(result)
        
        return results

    def process_and_save(self, ocr_results: tuple[str, float], subtitles: dict, output_file='subtitles_cn.json'):
        '''
        Convert Chinese text to pinyin and save final subtitle file
        '''
        # Create a map: index -> chinese text
        results_map = {idx: (cn_text, conf) for idx, cn_text, conf in ocr_results}
        
        final_subtitles = []
        
        for i, sub in enumerate(subtitles):
            result = results_map.get(i)  # Get OCR result if exists
            
            if result:
                cn_text, conf = result
                # Convert to pinyin
                pinyin_text = ' '.join(lazy_pinyin(cn_text, style=Style.TONE))
                
                final_subtitles.append({
                    'start': sub['start'],
                    'end': sub['end'],
                    'hanzi': cn_text,
                    'pinyin': pinyin_text,
                    'english': sub['text'],
                    'confidence': round(conf, 3)
                })
            else:
                # OCR failed or no screenshot, still include the entry
                final_subtitles.append({
                    'start': sub['start'],
                    'end': sub['end'],
                    'hanzi': '',
                    'pinyin': '',
                    'english': sub['text']                
                })
        
        # Save to JSON
        output_data = {
            'subtitles': final_subtitles,
            'version': '1.0',
            'language': 'zh-CN'
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        success_count = sum(1 for s in final_subtitles if s['hanzi'])
        print(f'\n✅ Saved {len(final_subtitles)} subtitles to {output_file}')
        print(f'📊 OCR success rate: {success_count}/{len(final_subtitles)} ({success_count/len(final_subtitles)*100:.1f}%)')

    async def extract(self, output_file='subtitles_cn.json'):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch_persistent_context(
                    user_data_dir='/tmp/playwright-chrome',
                    headless=False,
                    channel='chrome',  # Use real Chrome, not Chromium
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
                print('⏳ Waiting for playback to start...')
                await page.wait_for_function(
                    'document.querySelector("#video_player").currentTime > 0',
                    timeout=10000
                )
                
                # Set subtitle region
                await self.set_dimensions(video)
                # await self.show_overlay(page, video)
                
                # Load subtitles
                print(f'\n📄 Loading subtitles from {self.subs_path}')
                subtitles = self.load_subs(self.subs_path, time_offset=AD_OFFSET)
                
                # Collect screenshots
                print(f'\n📸 Collecting screenshots...')
                screenshots = await self.collect_screenshots(video, subtitles)
                await browser.close()    
                
                results = await self.process_screenshots(screenshots)

                # Save results
                self.process_and_save(results, subtitles, output_file)
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)
                print('🧹 Cleaned up worker processes')
                                   
async def main():
    video_url = os.getenv('VIDEO_URL')
    subs_path = os.getenv('SUBS_PATH')

    extractor = SubtitleExtractor(video_url, subs_path)
    await extractor.extract()

if __name__ == '__main__':
    asyncio.run(main())
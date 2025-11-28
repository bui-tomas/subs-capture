import asyncio
import cv2
import numpy as np
from playwright.async_api import async_playwright, Page, ElementHandle
from paddleocr import PaddleOCR
from pypinyin import lazy_pinyin, Style
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

AD_OFFSET = 18.8

class SubtitleExtractor:
    def __init__(self, url: str, subs_path: str):
        self.ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='ch',
    )
        self.subtitles = []
        self.url = url
        self.subs_path = subs_path

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

        self.subtitle_region = (int(h * 0.75), int(h * 0.85), int(w * 0.25), int(w * 0.75))
        self.left_lyrics_region = (int(h * 0.2), int(h * 0.7), int(w * 0.05), int(w * 0.15))
        self.right_lyrics_region = (int(h * 0.2), int(h * 0.7), int(w * 0.85), int(w * 0.95))

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
                    background: rgba(255, 255, 0, 0.1);
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
                    background: rgba(0, 0, 255, 0.1);
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
                    background: rgba(0, 255, 0, 0.1);
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

    async def collect_screenshots(self, video: ElementHandle, subtitles: list[dict]):
        '''
        Single-pass screenshot collection with 6 samples per subtitle window
        '''
        def get_offset(duration: float, is_lyrics=False, num_steps=3, overlap=0.9) -> list[float]:
            if is_lyrics:
                arr = np.linspace(0, duration / 2, (num_steps * 2))[1:-1]
                offsets = [0]
                offsets.extend([-val for val in arr])
            else:
                arr = np.linspace(0, duration, num_steps + 1)[1:]
                arr[-1] *= overlap
                offsets = [0]
                offsets.extend([x for val in arr for x in (val, -val)])
            return offsets
        
        screenshots = []

        for idx, sub in enumerate(subtitles):
            # Dynamic offset range  
            offsets = get_offset(sub['duration'], sub['is_lyrics'])

            for offset in offsets:
                timestamp = (sub['start'] + sub['end']) / 2 + offset
                
                await video.evaluate(f'v => v.currentTime = {timestamp}')
                await asyncio.sleep(0.2)
                
                screenshot = await video.screenshot()
                screenshots.append((idx, sub, screenshot))
            
            return screenshots 

    async def process_screenshots(self, screenshots: tuple[int, dict, bytes]):
        '''
        Parallel OCR with semaphore to control concurrency
        '''
        semaphore = asyncio.Semaphore(5)  # 5 concurrent OCR tasks
        
        async def ocr_task(idx: int, sub, screenshot) -> tuple[int, str]:
            async with semaphore:
                # Run blocking OCR in executor
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None, 
                    self.ocr_subs,
                    sub, 
                    screenshot
                )
                return (idx, text) 
        
        tasks = [ocr_task(idx, sub, ss) for idx, sub, ss in screenshots]  # ← Unpack 3 items
        results = await asyncio.gather(*tasks)
        return results

    def ocr_subs(self, sub: dict, screenshot_bytes: bytes):
        '''
        Extract Chinese text from screenshot subtitle region via OCR
        '''
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract subtitle region
        y1, y2, x1, x2 = self.subtitle_region
        subtitle_img = img[y1:y2, x1:x2]
        
        # Run OCR on subtitle region
        result = self.ocr.predict(subtitle_img)
        
        if not result or len(result) == 0:
            return None
        
        # Access the OCRResult object
        ocr_result = result[0]
        
        # Get texts and scores
        texts = ocr_result['rec_texts']
        scores = ocr_result['rec_scores']
        
        if not texts:
            return None
        
        # Filter by confidence > 0.2 and print results
        filtered_texts = []
        for text, score in zip(texts, scores):
            if score > 0.2:
                filtered_texts.append(text)
        
        return ''.join(filtered_texts) if filtered_texts else None

    def process_and_save(self, ocr_results: str, subtitles: dict, output_file='subtitles_cn.json'):
        '''
        Convert Chinese text to pinyin and save final subtitle file
        '''
        # Create a map: index -> chinese text
        results_map = {idx: cn_text for idx, cn_text in ocr_results}
        
        final_subtitles = []
        
        for i, sub in enumerate(subtitles):
            cn_text = results_map.get(i)  # Get OCR result if exists
            
            if cn_text:
                # Convert to pinyin
                pinyin_text = ' '.join(lazy_pinyin(cn_text, style=Style.TONE))
                
                final_subtitles.append({
                    'start': sub['start'],
                    'end': sub['end'],
                    'hanzi': cn_text,
                    'pinyin': pinyin_text,
                    'english': sub['text']
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
            
            

            # Save results
            self.process_and_save(screenshots, subtitles, output_file)
            
            await browser.close()          
                
async def main():
    video_url = os.getenv('VIDEO_URL')
    subs_path = os.getenv('SUBS_PATH')

    extractor = SubtitleExtractor(video_url, subs_path)
    await extractor.extract()

if __name__ == '__main__':
    asyncio.run(main())
import asyncio
import cv2
import numpy as np
from playwright.async_api import async_playwright
from paddleocr import PaddleOCR
from pypinyin import lazy_pinyin, Style
import json

AD_OFFSET = 18.8

class SubtitleExtractor:
    def __init__(self, url, subs_path):
        self.ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='ch',
    )
        self.subtitles = []
        self.url = url
        self.subs_path = subs_path
        self.subtitle_region = None  # Will be set dynamically

    async def show_sub_overlay(self, page, video):
        '''
        Draws a yellow border around detected subtitle region
        '''
        screenshot = await video.screenshot()
        self.subtitle_region = self.detect_sub_region(screenshot)
        y1, y2, x1, x2 = self.subtitle_region
        
        # Get video position on page
        box = await video.bounding_box()
        
        # Inject overlay div
        await page.evaluate(f'''
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
        ''')
        
        print(f'Overlay shown. Region: y={y1}-{y2}, x={x1}-{x2}')

    def detect_sub_region(self, screenshot_bytes):
        '''
        Detect the subtitle region from a sample frame
        Returns (y_start, y_end, x_start, x_end)
        '''
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        height, width = img.shape[:2]
        
        # Usually bottom 20-30% of video
        y_start = int(height * 0.75)
        y_end = int(height * 0.85)
        x_start = int(width * 0.25)
        x_end = int(width * 0.75)
        
        return (y_start, y_end, x_start, x_end)

    def load_subs(self, file_path: str, time_offset=0):
        '''
        Loads subtitles from JSON file and apply time offset
        Args:
            json_file_path: path to JSON file with subtitles
            time_offset: seconds to subtract from all timestamps
        Returns:
            List of subtitle dicts with adjusted timestamps
        '''
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        apply_offset = False  # Flag to track when to start applying offset
        
        subtitles = []
        for sub in data['subtitles']:
            if apply_offset:
                subtitles.append({
                    'start': sub['start'] + time_offset,
                    'end': sub['end'] + time_offset,
                    'text': sub['text']
                })
            else:
                subtitles.append({
                    'start': sub['start'],
                    'end': sub['end'],
                    'text': sub['text']
                })

            if sub['text'] == '=Episode 1=':
                apply_offset = True
        
        return subtitles

    async def collect_screenshots(self, video, subtitles):
        '''
        Single-pass screenshot collection with 6 samples per subtitle window
        '''
        screenshots = {}
        
        # Offset range: 18.5 to 20.5 (6 equal parts = 0.4 second intervals)
        offset_range = [0, -0.1, +0.1, -0.2, +0.2, -0.4, +0.4 -0.6, +0.6]
        
        print(f'\n📸 Collecting screenshots with multi-offset sampling...')
        
        for idx, sub in enumerate(subtitles):
            found = False
            
            for offset in offset_range:
                timestamp = (sub['start'] + sub['end']) / 2  + offset
                
                await video.evaluate(f'v => v.currentTime = {timestamp}')
                await asyncio.sleep(0.2)
                
                screenshot = await video.screenshot()
                
                # Try OCR - if we get Chinese text, we're done
                cn_text = self.capture_cn_subs(screenshot)
                
                if cn_text and len(cn_text) >= 2:
                    screenshots[idx] = (idx, cn_text)
                    print(f'  ✅ #{idx}: {cn_text[:20]}... (offset: {offset:+.1f}s) {sub['start']}')
                    found = True
                    break  # First match wins, move to next subtitle
            
            if not found:
                print(f'  ❌ #{idx}: No text found')
        
        # Convert to sorted list
        valid_screenshots = [screenshots[i] for i in sorted(screenshots.keys())]
        
        print(f'\n✅ Final: {len(valid_screenshots)}/{len(subtitles)} captured')
        
        return valid_screenshots

    def hash_subtitle_region(self, screenshot_bytes):
        '''Quick hash of subtitle region to detect changes'''
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        y1, y2, x1, x2 = self.subtitle_region
        subtitle_img = img[y1:y2, x1:x2]
        
        # Simple perceptual hash
        gray = cv2.cvtColor(subtitle_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (8, 8))
        avg = resized.mean()
        diff = resized > avg
        return hash(diff.tobytes())
 
    def capture_cn_subs(self, screenshot_bytes):
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

    def process_and_save(self, ocr_results, subtitles, output_file='subtitles_cn.json'):
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
            await asyncio.sleep(2)  # Let page settle
            
            video = await page.query_selector('#video_player')
            duration = await video.evaluate('v => v.duration')
            print(f'✅ Video duration: {duration:.1f}s ({duration/60:.1f} min)')
            
            # Start playback
            await video.evaluate('v => v.play()')
            await asyncio.sleep(2)
            
            # Wait for playback to start
            print('⏳ Waiting for playback to start...')
            await page.wait_for_function(
                'document.querySelector("#video_player").currentTime > 0',
                timeout=10000
            )
            
            current_time = await video.evaluate('v => v.currentTime')
            print(f'✅ Playback started at {current_time:.1f}s')
            
            # Set subtitle region
            await self.show_sub_overlay(page, video)
            
            # Load subtitles
            print(f'\n📄 Loading subtitles from {self.subs_path}')
            subtitles = self.load_subs(self.subs_path, time_offset=AD_OFFSET)
            print(f'✅ Loaded {len(subtitles)} subtitle entries')
            
            # Collect screenshots
            print(f'\n📸 Collecting screenshots...')
            screenshots = await self.collect_screenshots(video, subtitles)
            print(f'✅ Collected {len(screenshots)} screenshots')
            
            # Save results
            self.process_and_save(screenshots, subtitles, output_file)
            
            await browser.close()          
                
async def main():
    video_url = ''
    subs_path = ''

    extractor = SubtitleExtractor(video_url, subs_path)
    await extractor.extract()

if __name__ == '__main__':
    asyncio.run(main())
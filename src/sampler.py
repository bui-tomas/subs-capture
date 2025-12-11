import asyncio
import cv2
import numpy as np
from playwright.async_api import async_playwright
from paddleocr import PaddleOCR
import os
import time

class ScreenshotSampler:
    def __init__(self):
        start = time.time()
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='ch',
        )
        end = time.time()
        print('secs: ', end - start)

        self.video_width = None
        self.video_height = None
        self.subtitle_region = None
        self.left_lyrics_region = None
        self.right_lyrics_region = None

    async def set_dimensions(self, video):
        screenshot = await video.screenshot()

        if self.video_width is None:
            nparr = np.frombuffer(screenshot, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.video_height, self.video_width = img.shape[:2]

        h, w = self.video_height, self.video_width

        self.subtitle_region = (int(h * 0.75), int(h * 0.85), int(w * 0.25), int(w * 0.75))
        self.left_lyrics_region = (int(h * 0.2), int(h * 0.65), int(w * 0.05), int(w * 0.085))
        self.right_lyrics_region = (int(h * 0.2), int(h * 0.65), int(w * 0.88), int(w * 0.93))

    async def show_overlay(self, page, video):
        box = await video.bounding_box()

        y1, y2, x1, x2 = self.subtitle_region
        ly1, ly2, lx1, lx2 = self.left_lyrics_region
        ry1, ry2, rx1, rx2 = self.right_lyrics_region
        
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
                    pointer-events: none;
                    z-index: 99999;
                `;
                document.body.appendChild(overlay);
                
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

    def ocr_subs(self, screenshot_list: list[tuple[float, bytes]], brightness_threshold=210, is_lyrics=False) -> str:
        """Worker function that uses global OCR instance"""
        
        ocr_results = []
        
        for offset, screenshot in screenshot_list:
            nparr = np.frombuffer(screenshot, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if not is_lyrics:
                y1, y2, x1, x2 = self.subtitle_region
            else:
                y1, y2, x1, x2 = self.right_lyrics_region

            
            subtitle_img = img[y1:y2, x1:x2]

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
            
            # Run OCR
            result = self.ocr.predict(subtitle_img)
            # print(result)
            
            if not result or len(result) == 0:
                continue
            
            ocr_result = result[0]
            texts = ocr_result['rec_texts']
            scores = ocr_result['rec_scores']

            for idx, text in enumerate(texts):
                print(offset, text, scores[idx])
            
            if not texts:
                continue
            
            filtered_texts = [text for text, score in zip(texts, scores) if score > 0.2]
            
            if filtered_texts:
                combined_text = ''.join(filtered_texts)
                ocr_results.append((offset, combined_text))
        
        if not ocr_results:
            return ''
        
        ocr_results.sort(key=lambda x: x[0])
        
        # Deduplicate
        unique_texts = []
        seen = set()
        for offset, text in ocr_results:
            if text not in seen:
                unique_texts.append(text)
                seen.add(text)

        return ''.join(unique_texts)

    def get_offset(self, duration: float, is_lyrics=False, num_steps=2, overlap=0.9) -> list[float]:
        if is_lyrics:
            return [0, duration / 2 * 0.65, duration / 2 * 0.75, duration / 2 * 0.8]
        else:
            arr = np.linspace(0, duration, num_steps + 1)[1:]
            arr[-1] *= overlap
            offsets = [0]
            offsets.extend([x for val in arr for x in (val, -val)])
            offsets = [x / 2 for x in offsets]
        return offsets 

    async def sample(self, url, start, end, is_lyrics=False):
        async with async_playwright() as p:
            browser = await p.chromium.launch_persistent_context(
                user_data_dir='/tmp/playwright-chrome',
                headless=False,
                channel='chrome',
                args=[
                    '--start-maximized',
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
            
            print(f'\n📺 Opening: {url}')
            await page.goto(url, wait_until='networkidle')
            
            video = await page.query_selector('#video_player')
            
            danmu_button = await page.query_selector('i.icondanmukai')
            if danmu_button:
                await danmu_button.click()
                await asyncio.sleep(0.5)
            
            await video.evaluate('v => v.play()')
            
            print('⏳ Waiting for playback to start...')
            await page.wait_for_function(
                'document.querySelector("#video_player").currentTime > 0',
                timeout=10000
            )
            await asyncio.sleep(5)
            
            await self.set_dimensions(video)
            await self.show_overlay(page, video)

            os.makedirs('debug_screenshots', exist_ok=True)

            duration = end - start
            offsets = self.get_offset(duration, is_lyrics)
            # offsets= [0, -0.1, +0.1, -0.2, +0.2, -0.4, +0.4 -0.6, +0.6]

            screenshots = []

            print(offsets)

            for i, offset in enumerate(offsets):
                timestamp = ((start + end) / 2 + offset) - 0.2
                
                await video.evaluate(f'v => v.currentTime = {timestamp}')
                await asyncio.sleep(0.2)
                
                screenshot = await video.screenshot()
                screenshots.append((offset, screenshot))
                
                filename = f'debug_screenshots/sample_{start:.1f}s_offset_{offset:+.2f}s_{i+1}.png'
                with open(filename, 'wb') as f:
                    f.write(screenshot)

            cn_text = self.ocr_subs(screenshots, is_lyrics)
            print(cn_text)
            
            print(f'\n✅ Saved screenshots to debug_screenshots/')
            await browser.close()

if __name__ == '__main__':
    video_url = 'https://www.iyf.tv/play/MBxLp76dbY1?id=wLOU2ZuGh2A'
    #   "start": 172.37900000000002,
    #   "end": 173.58,
    
    start = 172.37900000000002
    end = 173.58
    
    sampler = ScreenshotSampler()
    asyncio.run(sampler.sample(video_url, start, end, is_lyrics=False))
import asyncio
import cv2
import numpy as np
from playwright.async_api import async_playwright
from paddleocr import PaddleOCR
from pypinyin import lazy_pinyin, Style
import json

class SmartSubtitleExtractor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        self.subtitles = []
        self.previous_frame = None
        self.subtitle_region = None  # Will be set dynamically
        
    async def extract_from_video(self, video_url, output_file='subtitles.json'):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            
            await page.goto(video_url)
            await page.wait_for_selector('video')
            
            video = await page.query_selector('video')
            duration = await video.evaluate('v => v.duration')
            
            print(f"Video duration: {duration} seconds")
            
            # Start playback
            await video.evaluate('v => v.play()')
            
            # Detect subtitle region from first few seconds
            await asyncio.sleep(3)  # Wait for subtitles to appear
            initial_screenshot = await video.screenshot()
            self.subtitle_region = self.detect_subtitle_region(initial_screenshot)
            print(f"Detected subtitle region: {self.subtitle_region}")
            
            # Monitor for changes
            check_interval = 0.1  # Check every 100ms
            last_ocr_time = 0
            min_ocr_interval = 0.5  # Minimum 500ms between OCR calls
            
            while True:
                current_time = await video.evaluate('v => v.currentTime')
                
                if current_time >= duration:
                    break
                
                # Capture current frame
                screenshot = await video.screenshot()
                
                # Check if subtitle region changed
                if self.has_subtitle_changed(screenshot):
                    # Only run OCR if enough time has passed since last OCR
                    if current_time - last_ocr_time >= min_ocr_interval:
                        print(f"[{current_time:.1f}s] Subtitle change detected!")
                        self.extract_text_from_frame(screenshot, current_time)
                        last_ocr_time = current_time
                
                await asyncio.sleep(check_interval)
            
            await browser.close()
            self.save_subtitles(output_file)
            print(f"\nExtracted {len(self.subtitles)} subtitle entries")
    
    def detect_subtitle_region(self, screenshot_bytes):
        """
        Detect the subtitle region from a sample frame
        Returns (y_start, y_end, x_start, x_end)
        """
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        height, width = img.shape[:2]
        
        # Usually bottom 20-30% of video
        # You can make this more sophisticated
        y_start = int(height * 0.75)
        y_end = int(height * 0.95)
        x_start = int(width * 0.1)
        x_end = int(width * 0.9)
        
        return (y_start, y_end, x_start, x_end)
    
    def has_subtitle_changed(self, screenshot_bytes):
        """
        Compare current frame with previous frame in subtitle region
        Returns True if significant change detected
        """
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        current_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract subtitle region
        y1, y2, x1, x2 = self.subtitle_region
        current_region = current_frame[y1:y2, x1:x2]
        
        # First frame
        if self.previous_frame is None:
            self.previous_frame = current_region
            return True
        
        # Convert to grayscale for comparison
        current_gray = cv2.cvtColor(current_region, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(current_gray, previous_gray)
        
        # Count changed pixels
        changed_pixels = np.sum(diff > 30)  # Threshold for "changed"
        total_pixels = diff.size
        change_percentage = (changed_pixels / total_pixels) * 100
        
        # Update previous frame
        self.previous_frame = current_region.copy()
        
        # Consider it changed if >5% of pixels changed
        return change_percentage > 5
    
    def extract_text_from_frame(self, screenshot_bytes, timestamp):
        """
        Extract Chinese text from the subtitle region only
        """
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract subtitle region
        y1, y2, x1, x2 = self.subtitle_region
        subtitle_img = img[y1:y2, x1:x2]
        
        # Run OCR only on subtitle region
        result = self.ocr.ocr(subtitle_img, cls=True)
        
        if not result or not result[0]:
            return None
        
        # Extract text
        texts = []
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            
            if confidence > 0.7:
                texts.append(text)
        
        if not texts:
            return None
        
        full_text = ' '.join(texts)
        
        # Convert to pinyin
        pinyin_text = ' '.join(lazy_pinyin(full_text, style=Style.TONE))
        
        # Add to subtitles
        self.subtitles.append({
            'startTime': round(timestamp, 2),
            'endTime': round(timestamp + 2, 2),
            'hanzi': full_text,
            'pinyin': pinyin_text,
            'english': ''
        })
        
        print(f"  → {full_text}")
        return full_text
    
    def save_subtitles(self, output_file):
        """
        Save subtitles and merge duplicates
        """
        # Remove consecutive duplicates
        merged = []
        for sub in self.subtitles:
            if not merged or sub['hanzi'] != merged[-1]['hanzi']:
                if merged:
                    merged[-1]['endTime'] = sub['startTime']
                merged.append(sub)
        
        # Update last subtitle end time
        if merged:
            merged[-1]['endTime'] = merged[-1]['startTime'] + 3
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'subtitles': merged,
                'version': '1.0',
                'language': 'zh-CN'
            }, f, ensure_ascii=False, indent=2)


async def main():
    extractor = SmartSubtitleExtractor()
    video_url = input("Enter video URL: ")
    await extractor.extract_from_video(video_url)

if __name__ == '__main__':
    asyncio.run(main())
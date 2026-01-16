import asyncio
import cv2
import numpy as np
from playwright.async_api import async_playwright
from pypinyin import lazy_pinyin, Style
from paddleocr import PaddleOCR
import os
import json
import time

SCREENSHOT_FOLDER = f'screenshots/test'

os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

def load_failed_segments(error_file: str) -> list[tuple[float, float]]:
    '''
    Load failed OCR segments from error text file
    
    Args:
        error_file: Path to the error file (e.g., 'ep02_errors.txt')
    
    Returns:
        List of (start, end) tuples
    '''
    failed_segments = []
    
    with open(error_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip header or empty lines
            if not line or 'Failed OCR' in line:
                continue
            
            # Parse "96.29, 101.02" format
            start, end = line.split(',')
            failed_segments.append((float(start.strip()), float(end.strip())))
    
    return failed_segments

def get_segments_with_timing(json_path, start_time=None, end_time=None):
    """
    Load JSON subtitle data and return list of tuples (start, end, segment_dict).
    
    Args:
        json_path: Path to the JSON file
        start_time: Optional start time to filter segments (inclusive)
        end_time: Optional end time to filter segments (inclusive)
    
    Returns:
        List of tuples: [(start, end, segment_dict), ...]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create list of tuples with timing info
    result = []
    for segment in data['subtitles']:
        seg_start = segment.get('start')
        seg_end = segment.get('end')
        
        # Skip segments without timing info
        if seg_start is None or seg_end is None:
            continue
        
        # Apply time filters if provided
        if start_time is not None and seg_end < start_time:
            continue
        if end_time is not None and seg_start > end_time:
            continue
        
        result.append((seg_start, seg_end))
    
    return result

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

    async def capture_with_overlay(self, video):
        screenshot_bytes = await video.screenshot()
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        y1, y2, x1, x2 = self.subtitle_region
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow
        
        ly1, ly2, lx1, lx2 = self.left_lyrics_region
        cv2.rectangle(image, (lx1, ly1), (lx2, ly2), (255, 0, 0), 3)  # Blue
        
        ry1, ry2, rx1, rx2 = self.right_lyrics_region
        cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)  # Green
        
        # Encode back to bytes
        _, buffer = cv2.imencode('.png', image)
        return buffer.tobytes()

    async def set_dimensions(self, video):
        screenshot = await video.screenshot()

        if self.video_width is None:
            nparr = np.frombuffer(screenshot, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.video_height, self.video_width = img.shape[:2]

        h, w = self.video_height, self.video_width

        self.subtitle_region = (int(h * 0.76), int(h * 0.88), int(w * 0.30), int(w * 0.70))
        self.left_lyrics_region = (int(h * 0.2), int(h * 0.65), int(w * 0.05), int(w * 0.085))
        self.right_lyrics_region = (int(h * 0.2), int(h * 0.65), int(w * 0.88), int(w * 0.93))

    def _ocr_subs(self, sub: dict, screenshot_list: list[tuple[float, bytes]], is_lyrics=False) -> tuple[str, dict]:
        '''Worker function that uses global OCR instance'''
        global _worker_ocr, _worker_regions
        
        ocr_results = []
        text_scores = {}  # Maps text -> list of confidence scores
        
        for offset, screenshot in screenshot_list:
            nparr = np.frombuffer(screenshot, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Save screenshot to disk
            screenshot_filename = f'{SCREENSHOT_FOLDER}/sub_{sub['start']}s_offset_{offset:+.2f}s.png'
            cv2.imwrite(screenshot_filename, img)
            
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

    def get_offset(self, duration: float, is_lyrics=False, num_steps=2, overlap=0.9) -> list[float]:
        if is_lyrics:
            return [duration / 2 * 0.65, duration / 2 * 0.75]
        else:
            arr = np.linspace(0, duration, num_steps + 1)[1:]
            arr[-1] *= overlap
            offsets = [0]
            offsets.extend([x for val in arr for x in (val, -val)])
            offsets = [x / 2 for x in offsets]
        return offsets 

    async def sample_segment(self, video, start, end, segment_idx, is_lyrics=False):
        """Sample a single subtitle segment"""
        duration = end - start
        offsets = self.get_offset(duration, is_lyrics)

        screenshots = []

        print(f"\n[Segment {segment_idx}] {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
        print(f"Offsets: {offsets}")

        for i, offset in enumerate(offsets):
            timestamp = ((start + end) / 2 + offset) - 0.2
            
            await video.evaluate(f'v => v.currentTime = {timestamp}')
            await asyncio.sleep(0.2)
            
            screenshot = await video.screenshot()
            screenshots.append((offset, screenshot))

        cn_text = self.ocr_subs(screenshots, is_lyrics=is_lyrics)
        print(f"[Segment {segment_idx}] OCR Result: {cn_text}")
        
        return cn_text
    
    def process_and_save(self, ocr_results: list[tuple[int, str, dict]], subtitles: list[dict], output_file='subtitles_cn.json'):
        '''
        Convert Chinese text to pinyin and save final subtitle file
        '''
        # Create a map: index -> (chinese text, metadata)
        results_map = {idx: (cn_text, metadata) for idx, cn_text, metadata in ocr_results}
        
        final_subtitles = []
        failed_segments = []  # Track failed OCR segments
        ad_played = False
        
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
                    failed_segments.append((sub['start'], sub['end']))
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
                    failed_segments.append((sub['start'], sub['end']))
        
        
        # Save to JSON
        output_data = {
            'subtitles': final_subtitles,
            'version': '1.0',
            'language': 'zh-CN'
        }
        
        with open(f'corrections_raw.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Calculate success rate (excluding lyrics)
        non_lyrics_subs = [s for s in final_subtitles if '♪' not in s['english']]
        success_count = sum(1 for s in non_lyrics_subs if s['hanzi'])
        total_non_lyrics = len(non_lyrics_subs)
        
        print(f'\n✅ Saved {len(final_subtitles)} subtitles to corrections_raw.json')
        print(f'📊 OCR success rate (non-lyrics): {success_count}/{total_non_lyrics} ({success_count/total_non_lyrics*100:.1f}%)')

    async def sample(self, url, segments: list[tuple[float, float]], is_lyrics=False):
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
            
            # button = await page.query_selector(os.getenv('BUTTON_SELECTOR'))
            # if button:
            #     await button.click()
            
            video = await page.query_selector('#video_player')
            
            # Start playback
            await video.evaluate('v => v.play()')
            
            # Wait for playback to start
            await page.wait_for_function(
                'document.querySelector("#video_player").currentTime > 0',
                timeout=10000
            )

            # Click the fullscreen button
            # fullscreen_btn = await page.query_selector('vg-fullscreen.control-item div[role="button"]')
            # if fullscreen_btn:
            #     await fullscreen_btn.click()
            #     await asyncio.sleep(1)

            await page.evaluate('''
                () => {
                    const fullscreenBtn = document.querySelector('vg-fullscreen div[role="button"]');
                    if (fullscreenBtn) {
                        fullscreenBtn.click();
                    }
                }
            ''')

            await self.set_dimensions(video)

            # Process all segments
            ocr_results = []
            subtitles = []
            ad_played = False
            ad_buffer = []
            AD_TRIGGER = 1300
            AD_BUFFER = 15

            await asyncio.sleep(5)

            for idx, (start, end) in enumerate(segments, start=1):
                duration = end - start
                offsets = self.get_offset(duration, is_lyrics)
                screenshots = []

                print(f"\n[Segment {idx}] {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
                print(f"Offsets: {offsets}")

                for i, offset in enumerate(offsets):
                    timestamp = ((start + end) / 2 + offset)

                    # Buffer zone: skip for now, capture after ad
                    if AD_TRIGGER - AD_BUFFER <= timestamp < AD_TRIGGER and not ad_played:
                        print(f"  ⏭️  Buffering {timestamp:.1f}s (ad zone)")
                        ad_buffer.append((idx, offset, timestamp))
                        continue
                    
                    # Ad trigger: sleep then process buffer
                    if timestamp >= AD_TRIGGER and not ad_played:
                        print(f'\n⏸️  Ad detected at {timestamp:.1f}s - sleeping 30s...')
                        ad_played = True
                        await video.evaluate(f'v => v.currentTime = {timestamp}')
                        await asyncio.sleep(30)
                        
                        # Capture all buffered screenshots
                        if ad_buffer:
                            print(f'📸 Capturing {len(ad_buffer)} buffered screenshots...')
                            for buf_idx, buf_offset, buf_ts in ad_buffer:
                                await video.evaluate(f'v => v.currentTime = {buf_ts}')
                                await asyncio.sleep(0.2)
                                screenshot = await video.screenshot()
                                
                                # Only add to current segment if it matches
                                if buf_idx == idx:
                                    screenshots.append((buf_offset, screenshot))
                                    print(f"  ✓ Buffered: offset {buf_offset:+.2f}s @ {buf_ts:.1f}s")
                            
                            ad_buffer.clear()
                    
                    # Normal screenshot capture
                    await video.evaluate(f'v => v.currentTime = {timestamp}')
                    await asyncio.sleep(0.2)
                    
                    screenshot = await self.capture_with_overlay(video)
                    screenshots.append((offset, screenshot))
                    print(f"  ✓ Captured: offset {offset:+.2f}s @ {timestamp:.1f}s")

                # OCR this segment
                sub = {'start': start, 'end': end, 'is_lyrics': is_lyrics}
                cn_text, metadata = self._ocr_subs(sub, screenshots, is_lyrics=is_lyrics)
                
                print(f"[Segment {idx}] OCR Result: {cn_text}")
                
                # Store results
                ocr_results.append((idx - 1, cn_text, metadata))
                subtitles.append({
                    'start': start,
                    'end': end,
                    'duration': duration,
                    'is_lyrics': is_lyrics,
                    'text': ''
                })
            
            # ✅ Close browser AFTER all segments processed
            await browser.close()
        
        # ✅ Save results AFTER browser closed
        self.process_and_save(ocr_results, subtitles, output_file='sampled_subtitles.json')
        
        # ✅ Return AFTER everything done
        return ocr_results

if __name__ == '__main__':
    video_url = 'https://www.iyf.tv/play/MBxLp76dbY1?id=nuY8fDGyIt3'
    
    # Define segments to sample
    # segments = load_failed_segments('loves_ambition_ep_2_en_subs.json_errors.txt')
    segments = get_segments_with_timing('subs/loves_ambition/loves_ambition_ep_9_subs.json', 1440, 1460)

    temp = []
    offset = 0 + 10.5

    for start, end in segments:
        temp.append((start + offset, end + offset))

    segments = temp
    print(segments)

    sampler = ScreenshotSampler()
    results = asyncio.run(sampler.sample(video_url, segments, is_lyrics=False))
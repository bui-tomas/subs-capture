import json
import os
import asyncio
import cv2
import numpy as np
import glob
from tqdm import tqdm
from playwright.async_api import async_playwright, ElementHandle


class SubtitleExtractor:
    def __init__(self, 
    subs_path: str,
    ad_template_path: str,
    button_selector: str = None,
    ):
        self.url = None
        self.ad_template_path = ad_template_path
        self.subs_path = subs_path
        self.screenshot_folder = f'screenshots/{subs_path.split('/')[-1].split('.')[0]}'
        os.makedirs(self.screenshot_folder, exist_ok=True)

        self.subtitles = []
        self.button_selector = button_selector


    async def capture_screenshots(self):
        self._load_subs(self.subs_path)

        def get_offset(duration, is_lyrics, num_steps=2, overlap=0.9):
            if is_lyrics:
                return [duration / 2 * 0.65, duration / 2 * 0.75]
            arr = np.linspace(0, duration, num_steps + 1)[1:]
            arr[-1] *= overlap
            offsets = [0]
            offsets.extend([x for val in arr for x in (val, -val)])
            return sorted([x / 2 for x in offsets])

        async with async_playwright() as p:
            browser = await p.chromium.launch_persistent_context(
                user_data_dir='/tmp/playwright-chrome',
                headless=False,
                channel='chrome',
                args=[
                    '--mute-audio',
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

            print(f'\nOpening: {self.url}')
            await page.goto(self.url, wait_until='networkidle')

            if self.button_selector:
                button = await page.query_selector(self.button_selector)
                if button:
                    await button.click()

            video = await page.query_selector('#video_player')
            await video.evaluate('v => v.play()')

            await page.wait_for_function(
                'document.querySelector("#video_player").currentTime > 0',
                timeout=10000
            )

            duration = await video.evaluate('v => v.duration')
            print(f'Seeking to {duration / 2:.1f}s to trigger ad...')
            await video.evaluate(f'v => v.currentTime = {duration / 2}')
            await asyncio.sleep(30)  # Wait for ad to finish
            print('Starting capture...')

            screenshot = await video.screenshot()
            img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
            self.video_height, self.video_width = img.shape[:2]

            ad_regions = await self._collect_screenshots(
                video=video,
                get_offset=get_offset,
                ad_template_path=self.ad_template_path,
            )

            await browser.close()

        return ad_regions


    def _load_subs(self, file_path: str):    
        def is_lyrics(text: str) -> bool:
            return '♪' in text
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.url = data['target_url']

        subtitles = []
        for idx, sub in enumerate(data['subtitles']):
            subtitles.append({
                'idx': idx,
                'start': sub['start'],
                'end': sub['end'],
                'duration': sub['end'] - sub['start'],
                'is_lyrics': is_lyrics(sub['text']),
                'text': sub['text']
            })

        self.subtitles = subtitles


    async def _seek(self, video, timestamp):
        await video.evaluate(f'''
            v => new Promise(resolve => {{
                if (Math.abs(v.currentTime - {timestamp}) < 0.1) {{
                    resolve();
                }} else {{
                    v.currentTime = {timestamp};
                    v.addEventListener('seeked', () => resolve(), {{ once: true }});
                }}
            }})
        ''')
        await asyncio.sleep(0.1)


    async def _linear_scan(self, video, start, end, step, subtitle_region, match_fn, greyscale=False):
        direction = 1 if end > start else -1
        t = start

        while (direction == 1 and t <= end) or (direction == -1 and t >= end):
            await self._seek(video, t)
            screenshot = await video.screenshot()

            img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
            y1, y2, x1, x2 = subtitle_region
            crop = img[y1:y2, x1:x2]

            if greyscale:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            if match_fn(crop):
                return t, crop

            t += direction * step

        return None, None


    async def _binary_scan(self, lo, hi, match_fn, search_for_start=True, precision=0.25):
        '''
        Binary search between lo and hi.
        match_fn(timestamp) -> bool, async.
        search_for_start=True: find leftmost match (narrows hi on match).
        search_for_start=False: find rightmost match (narrows lo on match).
        Returns boundary timestamp.
        '''
        while hi - lo > precision:
            mid = (lo + hi) / 2
            if await match_fn(mid):
                if search_for_start:
                    hi = mid
                else:
                    lo = mid
            else:
                if search_for_start:
                    lo = mid
                else:
                    hi = mid

        return hi if search_for_start else lo


    async def _has_ad(self, video, timestamp, template, region):
        '''Check if ad badge is visible at given video timestamp'''
        await self._seek(video, timestamp)
        screenshot = await video.screenshot()

        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        y1, y2 = int(h * region[0]), int(h * region[1])
        x1, x2 = int(w * region[2]), int(w * region[3])
        crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        tmpl = template
        if tmpl.shape[0] > crop.shape[0] or tmpl.shape[1] > crop.shape[1]:
            scale = min(crop.shape[0] / tmpl.shape[0], crop.shape[1] / tmpl.shape[1]) * 0.9
            tmpl = cv2.resize(tmpl, None, fx=scale, fy=scale)

        result = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        return score > 0.7


    async def _find_ad_boundaries(self, video, hit_time, template, region):
        '''Find ad start and end using binary search + linear scan'''

        async def ad_match(timestamp):
            return await self._has_ad(video, timestamp, template, region)

        # Get pixel coords for ad region from video dimensions
        screenshot = await video.screenshot()
        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        ad_pixel_region = (
            int(h * region[0]), int(h * region[1]),
            int(w * region[2]), int(w * region[3])
        )

        tmpl = template
        def ad_template_match(crop):
            t = tmpl
            if t.shape[0] > crop.shape[0] or t.shape[1] > crop.shape[1]:
                scale = min(crop.shape[0] / t.shape[0], crop.shape[1] / t.shape[1]) * 0.9
                t = cv2.resize(t, None, fx=scale, fy=scale)
            result = cv2.matchTemplate(crop, t, cv2.TM_CCOEFF_NORMED)
            return result.max() > 0.7

        start_boundary = await self._binary_scan(
            max(0, hit_time - 60), hit_time, ad_match, search_for_start=True
        )
        ad_start, _ = await self._linear_scan(
            video, start_boundary - 0.3, start_boundary + 0.3, 0.1, ad_pixel_region, ad_template_match, greyscale=True
        )
        if ad_start is None:
            ad_start = start_boundary

        duration = await video.evaluate('v => v.duration')
        right_bound = hit_time
        while await ad_match(right_bound):
            right_bound = min(right_bound + 30, duration)
            if right_bound >= duration:
                break

        end_boundary = await self._binary_scan(
            hit_time, right_bound, ad_match, search_for_start=False
        )
        ad_end, _ = await self._linear_scan(
            video, end_boundary + 0.3, end_boundary - 0.3, 0.1, ad_pixel_region, ad_template_match, greyscale=True
        )
        if ad_end is None:
            ad_end = end_boundary

        print(f'Ad: {ad_start:.2f}s -> {ad_end:.2f}s ({ad_end - ad_start:.2f}s)')
        return ad_start, ad_end


    async def _collect_screenshots(
        self,
        video: ElementHandle,
        get_offset: callable,
        ad_template_path: str = None,
        ad_region: tuple = (0.81, 0.86, 0.955, 0.995),
        intro_reference: float = 104.25
    ) -> list:
        ad_template = None
        if ad_template_path:
            ad_template = cv2.imread(ad_template_path, cv2.IMREAD_GRAYSCALE)

        offset = 0.0
        ad_regions = []
        idx = 0
        pbar = tqdm(total=len(self.subtitles), desc='Capturing', unit='sub')

        while idx < len(self.subtitles):
            pbar.n = idx
            pbar.refresh()

            sub = self.subtitles[idx]

            if sub['is_lyrics']:
                idx += 1
                continue

            midpoint = (sub['start'] + sub['end']) / 2 + offset

            # Check for ad before capturing
            if ad_template is not None:
                if await self._has_ad(video, midpoint, ad_template, ad_region):
                    ad_start, ad_end = await self._find_ad_boundaries(
                        video, midpoint, ad_template, ad_region
                    )
                    ad_duration = ad_end - ad_start
                    offset += ad_duration

                    # Adjust for intro trim if this ad is near the reference point
                    if intro_reference is not None and abs(ad_start - intro_reference) < 15:
                        trim = intro_reference - ad_start
                        offset -= trim
                        print(f'Intro trim: {trim:.2f}s')

                    ad_regions.append((ad_start, ad_end))
                    print(f'Offset now {offset:.2f}s at subtitle {idx}')

                    # Find restart point
                    restart_idx = idx
                    for j in range(idx - 1, -1, -1):
                        j_mid = (self.subtitles[j]['start'] + self.subtitles[j]['end']) / 2
                        j_video_time = j_mid + (offset - ad_duration)
                        if j_video_time < ad_start - 1.0:
                            restart_idx = j + 1
                            break
                    else:
                        restart_idx = 0

                    # Delete contaminated screenshots
                    for k in range(restart_idx, idx + 1):
                        s = self.subtitles[k]
                        for f in glob.glob(f'{self.screenshot_folder}/sub_{s["start"]}s_offset_*.png'):
                            os.remove(f)
                    print(f'Deleted subs {restart_idx}-{idx}, restarting from {restart_idx}')

                    idx = restart_idx
                    continue

            # Normal capture
            offsets = get_offset(sub['duration'], sub['is_lyrics'])
            for off in offsets:
                timestamp = (sub['start'] + sub['end']) / 2 + off + offset
                await self._seek(video, timestamp)
                screenshot = await video.screenshot()

                filename = f'sub_{sub["start"]}s_offset_{off:+.2f}s.png'
                with open(f'{self.screenshot_folder}/{filename}', 'wb') as f:
                    f.write(screenshot)

            idx += 1

        pbar.n = len(self.subtitles)
        pbar.refresh()
        pbar.close()

        return ad_regions

import cv2
import numpy as np
import asyncio
from playwright.async_api import ElementHandle


class SubtitleCalibrator:
    def __init__(self, ad_template_path: str, ad_region: tuple = (0.81, 0.86, 0.955, 0.995)):
        self.ad_template = cv2.imread(ad_template_path, cv2.IMREAD_GRAYSCALE)
        self.ad_region = ad_region

    async def calibrate(
        self,
        video: ElementHandle,
        intro_reference: float = 104.25,
        sweep_range: tuple = (900, 1500),
        sweep_step: float = 3.0
    ) -> list[tuple[float, float]]:
        '''
        Find all ads and return offset boundaries in subtitle time.

        Returns:
            [(sub_time_boundary, cumulative_offset), ...]
            Apply: for each sub, use highest boundary it exceeds.
        '''
        boundaries = []
        cumulative = 0.0

        # First ad near intro
        hit = await self._sweep_for_ad(video, 80, 130, step=3.0)
        if hit is not None:
            ad_start, ad_end = await self._find_ad_boundaries(video, hit)
            ad_duration = ad_end - ad_start
            trim = intro_reference - ad_start
            cumulative = ad_duration - trim

            # boundary in subtitle time = ad_start (offset was 0 before this)
            boundaries.append((ad_start, cumulative))
            print(f'Ad 1: {ad_start:.2f}s -> {ad_end:.2f}s | trim: {trim:.2f}s | offset: {cumulative:.2f}s')

        # Second ad in middle
        sweep_start = sweep_range[0] + cumulative
        sweep_end = sweep_range[1] + cumulative
        hit = await self._sweep_for_ad(video, sweep_start, sweep_end, step=sweep_step)
        if hit is not None:
            ad_start, ad_end = await self._find_ad_boundaries(video, hit)
            ad_duration = ad_end - ad_start

            # boundary in subtitle time = ad_start - cumulative (undo previous offset)
            boundary_sub_time = ad_start - cumulative
            cumulative += ad_duration

            boundaries.append((boundary_sub_time, cumulative))
            print(f'Ad 2: {ad_start:.2f}s -> {ad_end:.2f}s | offset: {cumulative:.2f}s')

        return boundaries

    async def _sweep_for_ad(self, video, start: float, end: float, step: float = 3.0) -> float | None:
        '''Probe at intervals, return first hit or None'''
        t = start
        while t <= end:
            if await self._has_ad(video, t):
                print(f'Ad badge found at {t:.2f}s')
                return t
            t += step
        return None

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

    async def _has_ad(self, video, timestamp) -> bool:
        '''Check if ad badge is visible at given video timestamp'''
        await self._seek(video, timestamp)
        screenshot = await video.screenshot()

        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        y1, y2 = int(h * self.ad_region[0]), int(h * self.ad_region[1])
        x1, x2 = int(w * self.ad_region[2]), int(w * self.ad_region[3])
        crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        tmpl = self.ad_template
        if tmpl.shape[0] > crop.shape[0] or tmpl.shape[1] > crop.shape[1]:
            scale = min(crop.shape[0] / tmpl.shape[0], crop.shape[1] / tmpl.shape[1]) * 0.9
            tmpl = cv2.resize(tmpl, None, fx=scale, fy=scale)

        result = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
        return result.max() > 0.7

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

    async def _find_ad_boundaries(self, video, hit_time):
        '''Find ad start and end using binary search + linear scan'''

        async def ad_match(timestamp):
            return await self._has_ad(video, timestamp)

        screenshot = await video.screenshot()
        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        ad_pixel_region = (
            int(h * self.ad_region[0]), int(h * self.ad_region[1]),
            int(w * self.ad_region[2]), int(w * self.ad_region[3])
        )

        tmpl = self.ad_template
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

        print(f'Ad boundaries: {ad_start:.2f}s -> {ad_end:.2f}s ({ad_end - ad_start:.2f}s)')
        return ad_start, ad_end
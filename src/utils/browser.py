from typing import Callable, Awaitable, Any
from playwright.async_api import async_playwright, Page, ElementHandle

async def _collect_screenshots( 
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


async def run_browser_pipeline(
    url: str,
    task: Callable[[Page, ElementHandle], Awaitable[Any]],
    button_selector: str = None
) -> Any:
    '''
    Generic browser pipeline that handles setup/teardown.
    
    Args:
        url: Video page URL
        task: Async function that receives (page, video) and returns results
        button_selector: Optional selector for a button to click on load
    '''
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
        
        print(f'\n📺 Opening: {url}')
        await page.goto(url, wait_until='networkidle')

        if button_selector:
            button = await page.query_selector(button_selector)
            if button:
                await button.click()
        
        # Start playback
        video = await page.query_selector('#video_player')
        await video.evaluate('v => v.play()')
        
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
        
        # Run the task
        result = await task(page, video)
        
        await browser.close()
        return result
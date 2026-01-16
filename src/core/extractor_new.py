import json
import logging
import os
import asyncio
import cv2
import numpy as np
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

            # Load screenshots from disk
            self._init_workers(file_dir)
            screenshots = self._load_screenshots_from_disk(file_dir)

            results = asyncio.run(self._process_screenshots(screenshots))
            self._process_and_save(results, self.subtitles)
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)
                print('Cleaned up worker processes')

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

    def _init_workers(self, file_dir: str):
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

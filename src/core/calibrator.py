import json
import os
import asyncio
import random
import cv2
import numpy as np
from playwright.async_api import ElementHandle
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.browser import run_browser_pipeline
from utils.ocr import init_worker, ocr_subs, calculate_regions

class SubtitleCalibrator:
    def __init__(self, subs_path: str = None):
        self.url = None
        self.subs_path = subs_path
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='ch',
        )

        self.video_width = None
        self.video_height = None
        self.subtitle_region = None

        self.sample = None
        self.ad_trigger = -1
        self.ad_offset = 0

    def calibrate(self):
        self._load_metadata(self.subs_path)
        
        async def task(page, video):
            # Set dimensions
            screenshot = await video.screenshot()
            img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
            self.video_height, self.video_width = img.shape[:2]
            self.subtitle_region = calculate_regions(self.video_height, self.video_width)['subtitle']
            
            duration = await video.evaluate('v => v.duration')
            self.ad_trigger = (duration / 2) - 30
            
            subs = self._load_subs(self.subs_path)
            offsets = self._get_offset(self.ad_offset)
            
            return await self._collect_screenshots(video, subs, offsets)
        
        screenshots = asyncio.run(run_browser_pipeline(
            url=self.url,
            task=task
        ))
        
        return self._score_offsets(screenshots)
    
    def _score_offsets(self, screenshots):
        model = SentenceTransformer('sentence-transformers/LaBSE')
        regions = calculate_regions(self.video_height, self.video_width)
        init_worker(regions)  # Sets globals in this process
        
        # Group screenshots by offset
        offset_groups = {}
        for (offset, idx, screenshot) in screenshots:
            if offset not in offset_groups:
                offset_groups[offset] = []
            offset_groups[offset].append((idx, screenshot))
        
        # Score each offset
        offset_scores = {}
        for offset, screenshot_list in offset_groups.items():
            successful_ocrs = 0
            semantic_scores = []
            
            for idx, screenshot in screenshot_list:
                sub = self.sample[idx]
                cn_text, _ = ocr_subs(sub, [(offset, screenshot)])
                
                if cn_text:
                    successful_ocrs += 1
                    
                    if sub.get('text'):
                        cn_embedding = model.encode([cn_text])
                        en_embedding = model.encode([sub['text']])
                        similarity = cosine_similarity(cn_embedding, en_embedding).flatten()[0]
                        semantic_scores.append(similarity)
            
            avg_semantic = np.mean(semantic_scores) if semantic_scores else 0.0
            offset_scores[offset] = {
                'count': successful_ocrs,
                'semantic_avg': avg_semantic
            }
        
        # Find winner
        max_count = max(s['count'] for s in offset_scores.values())
        
        candidates = [
            (offset, scores) 
            for offset, scores in offset_scores.items()
            if scores['count'] >= max_count - 1
        ]
        
        winner = max(candidates, key=lambda x: x[1]['semantic_avg'])[0]
        
        # Print results
        for offset, scores in sorted(offset_scores.items()):
            print(f'Offset {offset:+.2f}s: {scores["count"]}/{len(self.sample)} OCRs, avg semantic: {scores["semantic_avg"]:.3f}')
        
        return winner

    def _get_offset(self, initial: float, num_steps=5) -> list[float]:
        offsets = [initial]
        arr1 = np.linspace(initial, initial + 1, num_steps)
        arr2 = np.linspace(initial - 1, initial, num_steps)
        offsets.extend(arr1)
        offsets.extend(arr2)
        return sorted(set(offsets))

    async def _collect_screenshots(self, 
        video: ElementHandle, 
        subs: list[dict],
        offsets: list
    ) -> list[tuple[int, dict, list[tuple[float, bytes]]]]:
        '''
        Single-pass screenshot collection with 6 samples per subtitle window
        '''
        
        def get_samples(subs):
            random.seed(42)
            return random.sample(subs, 20)

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
        self.sample = get_samples(subs)

        for offset in offsets:
            for idx, sub in enumerate(self.sample):
                timestamp = (sub['start'] + sub['end']) / 2 + offset
                await seek_to_timestamp(video, timestamp)
                screenshot = await video.screenshot()
                screenshots.append((offset, idx, screenshot))

        return screenshots

    def _load_subs(self, file_path: str):
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

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        subtitles = []
        for sub in data['subtitles']:
            if is_lyrics(sub['text']):
                continue
            if sub['start'] < (self.ad_trigger - 30):      
                subtitles.append({
                    'start': sub['start'],
                    'end': sub['end'],
                    'duration': sub['end'] - sub['start'],
                    'text': sub['text'],
                    'is_lyrics': False
                })
        
        return subtitles

    def _load_metadata(self, file_path: str):    
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.ad_offset = data['ad_offset']
        self.url = data['target_url']

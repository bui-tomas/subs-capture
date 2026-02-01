import json
import os
import asyncio
import logging
import warnings
import cv2
import numpy as np
import glob
from pathlib import Path
from pypinyin import lazy_pinyin, Style
from tqdm.asyncio import tqdm
from paddleocr import PaddleOCR
from concurrent.futures import ProcessPoolExecutor

# Suppress PaddleOCR warnings
logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('paddlex').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*ccache.*')


def _calculate_regions(height: int, width: int) -> dict:
    return {
        'subtitle': (int(height * 0.76), int(height * 0.89), int(width * 0.27), int(width * 0.73)),
        'left_lyrics': (int(height * 0.2), int(height * 0.65), int(width * 0.05), int(width * 0.085)),
        'right_lyrics': (int(height * 0.2), int(height * 0.65), int(width * 0.88), int(width * 0.93)),
    }


def _init_worker(regions: dict):
    global _worker_ocr, _worker_regions
    _worker_ocr = PaddleOCR(use_textline_orientation=True, lang='ch')
    _worker_regions = regions


def _ocr_subs(screenshot_list: list[tuple[float, bytes]]) -> tuple[str, dict]:
    global _worker_ocr, _worker_regions

    ocr_results = []
    text_scores = {}

    for offset, screenshot in screenshot_list:
        nparr = np.frombuffer(screenshot, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        y1, y2, x1, x2 = _worker_regions['subtitle']
        subtitle_img = img[y1:y2, x1:x2]

        result = _worker_ocr.predict(subtitle_img)

        if not result or len(result) == 0:
            continue

        ocr_result = result[0]
        texts = ocr_result['rec_texts']
        conf_scores = ocr_result['rec_scores']

        if not texts:
            continue

        filtered_texts = []
        filtered_scores = []
        for text, conf in zip(texts, conf_scores):
            filtered_texts.append(text)
            filtered_scores.append(conf)

        if filtered_texts:
            combined_text = ''.join(filtered_texts)
            avg_score = np.mean(filtered_scores)

            if combined_text not in text_scores:
                text_scores[combined_text] = []
            text_scores[combined_text].append(avg_score)

            ocr_results.append((offset, combined_text))

    if not ocr_results:
        return '', {'variants': '', 'confidences': []}

    ocr_results.sort(key=lambda x: x[0])

    unique_texts = []
    variant_confidences = []
    seen = set()

    for offset, text in ocr_results:
        if text not in seen:
            unique_texts.append(text)
            seen.add(text)
            variant_confidences.append(np.mean(text_scores[text]))

    metadata = {
        'variants': ';'.join(unique_texts),
        'confidences': [round(conf, 3) for conf in variant_confidences]
    }

    return ''.join(unique_texts), metadata


class OCRProcessor:
    def __init__(self, subs_path: str):
        self.subs_path = subs_path
        self.screenshot_folder = f'screenshots/{subs_path.split("/")[-1].split(".")[0]}'
        self.executor = None
        self.subtitles = []


    def extract_hanzi(self, chunk_size: int = 50):
        try:
            self._load_subs(self.subs_path)
            self.executor = self._init_workers()

            all_results = []
            screenshot_paths = self._get_screenshot_paths()

            for i in range(0, len(screenshot_paths), chunk_size):
                chunk_paths = screenshot_paths[i:i + chunk_size]
                screenshots = self._load_screenshots_chunk(chunk_paths)

                results = asyncio.run(self._process_screenshots(screenshots))
                all_results.extend(results)

                print(f'Processed {min(i + chunk_size, len(screenshot_paths))}/{len(screenshot_paths)}')

            self._process_and_save(all_results)
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)


    def _init_workers(self, num_workers=4):
        first_img = glob.glob(f'{self.screenshot_folder}/sub_*.png')[0]
        img = cv2.imread(first_img)
        video_height, video_width = img.shape[:2]

        regions = _calculate_regions(video_height, video_width)

        return ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(regions,)
        )


    def _get_screenshot_paths(self) -> list[tuple[int, list]]:
        paths_by_idx = {}

        for img_path in glob.glob(f'{self.screenshot_folder}/sub_*.png'):
            filename = Path(img_path).stem
            parts = filename.split('_')
            start_time = float(parts[1].rstrip('s'))
            offset = float(parts[3].rstrip('s'))

            for idx, sub in enumerate(self.subtitles):
                if abs(sub['start'] - start_time) < 0.01:
                    if idx not in paths_by_idx:
                        paths_by_idx[idx] = []
                    paths_by_idx[idx].append((offset, img_path))
                    break

        return [(idx, paths_by_idx[idx]) for idx in sorted(paths_by_idx.keys())]


    def _load_screenshots_chunk(self, chunk: list[tuple[int, list]]) -> list[tuple[int, dict, list[tuple[float, bytes]]]]:
        result = []
        for idx, path_list in chunk:
            screenshots = []
            for offset, img_path in path_list:
                with open(img_path, 'rb') as f:
                    screenshots.append((offset, f.read()))
            result.append((idx, self.subtitles[idx], screenshots))
        return result


    def _load_subs(self, file_path: str):
        def is_lyrics(text: str) -> bool:
            return '♪' in text

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

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


    async def _process_screenshots(
        self,
        screenshots: list[tuple[int, dict, list[tuple[float, bytes]]]]
    ) -> list[tuple[int, str]]:
        async def ocr_task(idx: int, sub: dict, screenshot_list: list[tuple[float, bytes]]) -> tuple[int, str]:
            loop = asyncio.get_event_loop()
            text, conf = await loop.run_in_executor(
                self.executor,
                _ocr_subs,
                screenshot_list
            )
            return (idx, text, conf)

        tasks = [
            ocr_task(idx, sub, ss_list)
            for item in screenshots
            if item is not None
            for idx, sub, ss_list in [item]
        ]

        results = []
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc='OCR Processing', unit='subtitle'):
            result = await coro
            results.append(result)

        return results


    def _process_and_save(self, ocr_results: list[tuple[int, str, dict]]):
        results_map = {idx: (cn_text, metadata) for idx, cn_text, metadata in ocr_results}

        final_subtitles = []
        failed_segments = []

        for i, sub in enumerate(self.subtitles):
            result = results_map.get(i)

            if result:
                cn_text, metadata = result
                pinyin_text = ' '.join(lazy_pinyin(cn_text, style=Style.TONE))

                final_subtitles.append({
                    'start': sub['start'],
                    'end': sub['end'],
                    'hanzi': cn_text,
                    'pinyin': pinyin_text,
                    'english': sub['text'],
                    'metadata': metadata
                })

                if not cn_text and not sub['is_lyrics']:
                    failed_segments.append(sub)
            else:
                final_subtitles.append({
                    'start': sub['start'],
                    'end': sub['end'],
                    'hanzi': '',
                    'pinyin': '',
                    'english': sub['text'],
                    'metadata': {'variants': '', 'confidences': []}
                })

                if not sub['is_lyrics']:
                    failed_segments.append(sub)

        base_name = self.subs_path.split('/')[-1].split('.')[0]

        output_data = {
            'subtitles': final_subtitles
        }

        with open(f'{base_name}_raw.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        if failed_segments:
            with open(f'{base_name}_errors.txt', 'w', encoding='utf-8') as f:
                f.write('Failed OCR segments (start, end):\n')
                for sub in failed_segments:
                    f.write(f'{sub["start"]}, {sub["end"]}, {sub["text"]}\n')

            print(f'Saved {len(failed_segments)} failed segments to {base_name}_errors.txt')

        non_lyrics_subs = [s for s in final_subtitles if '♪' not in s['english']]
        success_count = sum(1 for s in non_lyrics_subs if s['hanzi'])
        total_non_lyrics = len(non_lyrics_subs)

        print(f'\nSaved {len(final_subtitles)} subtitles to {base_name}_raw.json')
        print(f'OCR success rate (non-lyrics): {success_count}/{total_non_lyrics} ({success_count/total_non_lyrics*100:.1f}%)')
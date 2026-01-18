import logging
import warnings
import cv2
import numpy as np
from paddleocr import PaddleOCR
from concurrent.futures import ProcessPoolExecutor

# Annoying warning
logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('paddlex').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*ccache.*')

def calculate_regions(height: int, width: int) -> dict:
    '''
    Calculate OCR regions based on video dimensions
    '''
    return {
        'subtitle': (int(height * 0.76), int(height * 0.89), int(width * 0.27), int(width * 0.73)),
        'left_lyrics': (int(height * 0.2), int(height * 0.65), int(width * 0.05), int(width * 0.085)),
        'right_lyrics': (int(height * 0.2), int(height * 0.65), int(width * 0.88), int(width * 0.93)),
    }

def init_worker(regions: dict):
    '''
    Initialize OCR once per worker process
    '''
    global _worker_ocr, _worker_regions
    _worker_ocr = PaddleOCR(use_textline_orientation=True, lang='ch')
    _worker_regions = regions

def init_workers(
    file_dir: str,
    num_workers = 4
):
    '''Initialize executor using dimensions from existing screenshots'''
    import glob
    
    # Load first screenshot to get dimensions
    first_img = glob.glob(f'{file_dir}/sub_*.png')[0]
    img = cv2.imread(first_img)
    video_height, video_width = img.shape[:2]
    
    regions = calculate_regions(video_height, video_width)

    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(regions,)
    )

    return executor

def ocr_subs(sub: dict, screenshot_list: list[tuple[float, bytes]], ocr = None) -> tuple[str, dict]:
    '''
    Worker function that uses global OCR instance
    '''
    global _worker_ocr, _worker_regions
    
    ocr_results = []
    text_scores = {}  # Maps text -> list of confidence scores
    
    for offset, screenshot in screenshot_list:
        nparr = np.frombuffer(screenshot, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract appropriate region
        if sub['is_lyrics']:
            if sub['start'] < 120:
                y1, y2, x1, x2 = _worker_regions['left_lyrics']
            else:
                y1, y2, x1, x2 = _worker_regions['right_lyrics']
        else:
            y1, y2, x1, x2 = _worker_regions['subtitle']
        
        subtitle_img = img[y1:y2, x1:x2]
        
        # Run OCR
        result = _worker_ocr.predict(subtitle_img)
        
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

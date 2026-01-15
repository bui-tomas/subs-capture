import json
import numpy as np
from pypinyin import lazy_pinyin, Style
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from itertools import product

# Load model once at startup
model = SentenceTransformer('sentence-transformers/LaBSE')

def extend_subtitle_durations(subtitles: list[dict], gap: float = 0.2) -> list[dict]:
    """Extend each subtitle's end time to gap seconds before next subtitle starts
    
    Args:
        subtitles: List of subtitle dictionaries with 'start' and 'end' times
        gap: Seconds of gap to leave before next subtitle (default 0.2)
    
    Returns:
        List of subtitles with extended end times
    """
    extended = []
    
    for i, sub in enumerate(subtitles):
        current = sub.copy()
        
        # If there's a next subtitle, extend current to gap before next starts
        if i < len(subtitles) - 1:
            next_start = subtitles[i + 1]['start']
            new_end = next_start - gap
            
            # Only extend if it would actually make the subtitle longer
            # (don't shorten it if next subtitle overlaps)
            if new_end > current['end']:
                current['end'] = round(new_end, 2)
        
        extended.append(current)
    
    return extended

def filter_variants_by_prev_sub(current_variants: list[str], current_confs: list[float],
                                prev_hanzi: str) -> tuple[list[str], list[float]]:
    """Remove variants from current if they're substrings of previous subtitle's final hanzi"""
    if not prev_hanzi or len(current_variants) <= 1:
        return current_variants, current_confs
    
    filtered_variants = []
    filtered_confs = []
    
    for var, conf in zip(current_variants, current_confs):
        # Check if variant is substring of previous winner OR vice versa
        if var not in prev_hanzi and prev_hanzi not in var:
            filtered_variants.append(var)
            filtered_confs.append(conf)
    
    # If all filtered out, keep originals
    return (filtered_variants, filtered_confs) if filtered_variants else (current_variants, current_confs)

def pick_best_variant_semantic(variants: list[str], confidences: list[float], english_text: str, 
                                char_similarity_threshold: float = 0.75, 
                                min_cosine_score: float = 0.4,
                                high_confidence_threshold: float = 0.8) -> tuple[str, float, float]:
    """Pick best variant using character similarity grouping + Cartesian product + individual variants
    
    Returns:
        tuple: (best_text, cosine_score, ocr_confidence)
    """
    if not variants:
        return ('', 0.0, 0.0)
    
    if len(variants) == 1:
        ocr_conf = confidences[0] if confidences else 0.0
        if english_text:
            embedding = model.encode([variants[0]])
            english_embedding = model.encode([english_text])
            score = cosine_similarity(embedding, english_embedding).flatten()[0]
            return (variants[0], float(score), float(ocr_conf))
        return (variants[0], 0.0, float(ocr_conf))
    
    # Step 1: Group variants by character similarity
    groups = []
    used = set()
    
    for i, variant in enumerate(variants):
        if i in used:
            continue
        
        group = [(i, variant, confidences[i] if i < len(confidences) else 0.0)]
        used.add(i)
        
        for j, other_variant in enumerate(variants):
            if j in used:
                continue
            
            similarity = SequenceMatcher(None, variant, other_variant).ratio()
            if similarity >= char_similarity_threshold:
                group.append((j, other_variant, confidences[j] if j < len(confidences) else 0.0))
                used.add(j)
        
        groups.append(group)
    
    # Step 2: Generate candidates
    candidates = []  # (text, ocr_confidence)
    
    # A) Individual variants (standalone)
    for variant, conf in zip(variants, confidences):
        candidates.append((variant, conf))
    
    # B) Cartesian product (if multiple groups)
    if len(groups) > 1:
        group_variants = []
        group_confidences = []
        
        for group in groups:
            group_variants.append([item[1] for item in group])
            group_confidences.append([item[2] for item in group])
        
        for combo in product(*group_variants):
            text = ''.join(combo)
            
            # Calculate average confidence
            selected_confs = []
            for group_idx, variant in enumerate(combo):
                variant_idx = group_variants[group_idx].index(variant)
                selected_confs.append(group_confidences[group_idx][variant_idx])
            
            avg_conf = sum(selected_confs) / len(selected_confs) if selected_confs else 0.0
            candidates.append((text, avg_conf))
    
    # Step 3: Calculate cosine similarity
    if not english_text:
        first_conf = confidences[0] if confidences else 0.0
        return (variants[0], 0.0, float(first_conf))
    
    candidate_texts = [c[0] for c in candidates]
    candidate_confs = [c[1] for c in candidates]
    candidate_embeddings = model.encode(candidate_texts)
    english_embedding = model.encode([english_text])
    cosine_scores = cosine_similarity(candidate_embeddings, english_embedding).flatten()
    
    # Step 4: Pick winner
    best_idx = cosine_scores.argmax()
    best_score = float(cosine_scores[best_idx])
    best_text = candidate_texts[best_idx]
    best_conf = float(candidate_confs[best_idx])
    
    # Step 5: Validation
    if best_score < min_cosine_score and best_conf < high_confidence_threshold:
        return ('', best_score, best_conf)
    
    return (best_text, best_score, best_conf)

def post_process_subtitles(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    subtitles = data['subtitles']
    processed = []
    
    for i, sub in enumerate(subtitles):
        current = sub.copy()
        
        if not current['hanzi'] or 'metadata' not in current:
            processed.append(current)
            continue
        
        metadata = current['metadata']
        variants = metadata.get('variants', '').split(';') if metadata.get('variants') else []
        confidences = metadata.get('confidences', [])
        english_text = current.get('english', '')
        
        if not variants:
            processed.append(current)
            continue
        
        # Step 1: Filter by looking backward to previous subtitle's final hanzi
        if i > 0:
            prev_hanzi = processed[i - 1]['hanzi']
            if prev_hanzi:
                variants, confidences = filter_variants_by_prev_sub(
                    variants, confidences, prev_hanzi
                )
        
        if not variants:
            processed.append(current)
            continue
        
        # Step 2: Filter low confidence (< 0.7) if multiple exist
        if len(variants) > 1:
            high_conf_variants = []
            high_conf_confidences = []
            for v, c in zip(variants, confidences):
                if c >= 0.7:
                    high_conf_variants.append(v)
                    high_conf_confidences.append(c)
            
            if high_conf_variants:
                variants = high_conf_variants
                confidences = high_conf_confidences
        
        # Step 3: Pick best using Cartesian product + individual variants
        best_text, cosine_score, ocr_conf = pick_best_variant_semantic(variants, confidences, english_text)
        current['hanzi'] = best_text
        current['metadata']['cosine_similarity'] = round(cosine_score, 4)
        current['metadata']['ocr_confidence'] = round(ocr_conf, 4)
        
        # Regenerate pinyin
        if current['hanzi']:
            current['pinyin'] = ' '.join(lazy_pinyin(current['hanzi'], style=Style.TONE))
        else:
            current['pinyin'] = ''
        
        processed.append(current)
    
    data['subtitles'] = processed
    data['subtitles'] = extend_subtitle_durations(data['subtitles'], gap=0.1)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f'✅ Processed {len(processed)} subtitles → {output_file}')

if __name__ == '__main__':
    post_process_subtitles('subs/loves_ambition/loves_ambition_ep_7_subs_raw.json', 'subs/loves_ambition/loves_ambition_ep_7_subs_cleaned.json')
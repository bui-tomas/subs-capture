def load_json(filepath: str) -> list[dict]:
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['subtitles']


def seconds_to_srt_time(seconds: float) -> str:
    '''Convert 152.355 -> 00:02:32,355'''
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}'


def subtitles_to_srt(subtitles: list[dict], include_pinyin: bool = True) -> str:
    '''
    Convert subtitle dicts to SRT format.
    Uses {\an8} tag for top positioning (works in VLC and ffmpeg).
    '''
    lines = []
    
    for i, sub in enumerate(subtitles, start=1):
        start = seconds_to_srt_time(sub['start'])
        end = seconds_to_srt_time(sub['end'])
        
        # Stack hanzi and pinyin
        if include_pinyin and sub.get('pinyin'):
            text = f'{sub["hanzi"]}\n{sub["pinyin"]}'
        else:
            text = sub['hanzi']
        
        # {\an8} positions at top center
        lines.append(f'{i}')
        lines.append(f'{start} --> {end}')
        lines.append(f'{{\\an8}}{text}')
        lines.append('')
    
    return '\n'.join(lines)


def save_srt(srt_content: str, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)


if __name__ == '__main__':    
    input_file = 'subs/loves_ambition_reviewed/loves_ambition_ep_15_subs_reviewed.json'
    output_file = input_file.split('/')[-1].rsplit('.', 1)[0] + '.srt'
    
    subtitles = load_json(input_file)
    srt = subtitles_to_srt(subtitles)
    save_srt(srt, output_file)
    
    print(f'Converted {len(subtitles)} subtitles -> {output_file}')
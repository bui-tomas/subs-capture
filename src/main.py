from core.extractor import SubtitleExtractor
from core.ocr_processor import OCRProcessor

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

SUBS_PATH = os.getenv('SUBS_PATH')
AD_TEMPLATE_PATH = os.getenv('AD_TEMPLATE_PATH')
BUTTON_SELECTOR = os.getenv('BUTTON_SELECTOR')

def main():
    extractor = SubtitleExtractor(
        subs_path=SUBS_PATH,
        ad_template_path=AD_TEMPLATE_PATH,
        button_selector=BUTTON_SELECTOR
    )
    asyncio.run(extractor.capture_screenshots())

    processor = OCRProcessor(subs_path=SUBS_PATH)
    processor.extract_hanzi()

if __name__ == '__main__':
    main()
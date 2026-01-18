from core.extractor_new import SubtitleExtractor
from core.calibrator import SubtitleCalibrator

import os
from dotenv import load_dotenv
load_dotenv()

SUBS_PATH = os.getenv('SUBS_PATH')
BUTTON_SELECTOR = os.getenv('BUTTON_SELECTOR')

def pipeline():
    calibrator = SubtitleCalibrator()


    offset = calibrator.calibrate()
    extractor = SubtitleExtractor()

    extractor.extract()
    pass

def main():
    pass

def test():
    calibrator = SubtitleCalibrator(subs_path=SUBS_PATH)
    margins = calibrator.calibrate()

    extractor = SubtitleExtractor(subs_path=SUBS_PATH, button_selector=BUTTON_SELECTOR, margins=margins)
    extractor.capture_screenshots()
    extractor.extract_hanzi()

    # extractor.extract()

    # extractor.extract('screenshots/loves_ambition_ep_8_subs')

if __name__ == '__main__':
    test()

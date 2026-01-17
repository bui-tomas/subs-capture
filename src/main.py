from core.extractor import SubtitleExtractor
from core.calibrator import SubtitleCalibrator

import os
from dotenv import load_dotenv
load_dotenv()

SUBS_PATH = os.getenv('SUBS_PATH')

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
    calibrator.calibrate()

    # extractor = SubtitleExtractor()

    # extractor.extract()

    # extractor.extract('screenshots/loves_ambition_ep_8_subs')

if __name__ == '__main__':
    test()

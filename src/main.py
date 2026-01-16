from core.extractor import SubtitleExtractor
from core.calibrator import SubtitleCalibrator

from dotenv import load_dotenv
load_dotenv()

def pipeline():
    calibrator = SubtitleCalibrator()


    offset = calibrator.calibrate()
    extractor = SubtitleExtractor()

    extractor.extract()
    pass

def main():
    pass

def test():
    # calibrator = SubtitleCalibrator()
    # calibrator.calibrate()

    extractor = SubtitleExtractor()

    extractor.extract()

    # extractor.extract('screenshots/loves_ambition_ep_8_subs')

if __name__ == '__main__':
    test()

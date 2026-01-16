from core.extractor import SubtitleExtractor
from core.calibrator import SubtitleCalibrator

def pipeline():
    calibrator = SubtitleCalibrator()
    offset = calibrator.calibrate()
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

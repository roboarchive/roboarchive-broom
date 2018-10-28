import nist_tools
from nist_tools.extract_nist_text import BaseMain, parse_args, display

import cv2


class BlurMain(BaseMain):
    SRC_DIR = 'text_extracted'
    DST_DIR = 'blurred'

    def process_file(self, args, input_path, output_path):
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        kernel = 5
        blur = cv2.GaussianBlur(img, (kernel, kernel), 0)

        cv2.imwrite(output_path, blur)


if __name__ == '__main__':
    args = parse_args()
    BlurMain().main(args)
    print('done')

import argparse
import cv2
import os
import math

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='train/gen')
    parser.add_argument('--index', type=int)
    return parser.parse_args()


def display(img):
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def extract_text(args, input_path, output_path):
    img = cv2.imread(input_path)

    assert img is not None, 'Image {} could not be read'.format(input_path)

    img = img[2100:, :]

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = (255 - grayscale)

    im2, contours, hierarchy = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if approx.shape[0] == 4:
        mask = np.zeros(grayscale.shape, np.uint8)

        # fill bounding box
        cv2.fillConvexPoly(mask, cnt, 255)

        # reduce bounding box size to get rid of lines
        cv2.polylines(mask, [cnt], True, (0, 0, 0), 20)

        # save inverted matrix to add white borders in future
        inverted = 255 - mask

        # copy bounding box to destination image
        dst = cv2.bitwise_and(grayscale, grayscale, mask=mask)

        # restore white color around bounding box
        dst = inverted + dst

        cv2.imwrite(output_path, dst)
        return True

    else:
        print('Omitting rectangle of size {}'.format(approx.shape[0]))
        return False


class BaseMain:
    SRC_DIR = 'nist_orig'
    DST_DIR = 'text_extracted'

    def get_sorted_files(self, args):
        d = os.path.join(args.data_dir, self.SRC_DIR)

        lst = [f for f in os.listdir(d) if (f.endswith('.png') or f.endswith('.jpg'))]
        lst.sort()
        return lst

    def main(self, args):
        lst = self.get_sorted_files(args)

        if args.index:
            lst = lst[args.index:args.index+1]

        skipped = 0
        for i, f in enumerate(lst):
            input_path = os.path.join(args.data_dir, self.SRC_DIR, f)
            output_path = os.path.join(args.data_dir, self.DST_DIR, f)

            print('Processing {}/{}, omitted {}'.format(i, len(lst), skipped))
            result = self.process_file(args, input_path, output_path)
            if not result:
                skipped += 1

    def process_file(self, args, input_path, output_path):
        raise NotImplementedError


class ExtractTextMain(BaseMain):
    def process_file(self, args, input_path, output_path):
        return extract_text(args, input_path, output_path)


if __name__ == '__main__':
    args = parse_args()
    ExtractTextMain().main(args)
    print('done')

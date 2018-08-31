import os
import random

import cv2
import numpy as np

from nist_tools.extract_nist_text import BaseMain, parse_args, display


class CombineMain(BaseMain):
    SRC_DIR = 'blurred'
    DST_DIR = 'combined_raw'

    BG_DIR = 'backgrounds'
    SMPL_DIR = 'combined_clean'

    def __init__(self):
        self.backgrounds = os.listdir(os.path.join(args.data_dir, self.BG_DIR))
        self.backgrounds.sort()

    def get_random_bg(self):
        filename = random.choice(self.backgrounds)
        return os.path.join(args.data_dir, self.BG_DIR, filename)

    def main(self, args):
        lst = self.get_sorted_files(args)

        a = lst[::2]
        b = lst[1::2]

        pairs = list(zip(a, b))

        if args.index:
            pairs = pairs[args.index:args.index + 1]

        for i, pair in enumerate(pairs):
            a_path = os.path.join(args.data_dir, self.SRC_DIR, pair[0])
            b_path = os.path.join(args.data_dir, self.SRC_DIR, pair[1])

            fname = 'combined-{}.png'.format(i)
            smpl_path = os.path.join(args.data_dir, self.SMPL_DIR, fname)

            bg_path = self.get_random_bg()
            output_path = os.path.join(args.data_dir, self.DST_DIR, fname)

            print('Processing {}/{}'.format(i, len(pairs)))

            self.combine_file(args, a_path, b_path, bg_path, output_path, smpl_path)

    def random_bool(self):
        return random.choice([True, False])

    def load_text_image(self, shape, path, vert_offset, hor_offset):
        layer = np.full(shape, 255)
        sub_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        h, w = sub_image.shape
        layer[vert_offset:vert_offset + h, hor_offset:hor_offset + w] = sub_image
        return layer

    def merge_with_text(self, img, text_file_path, density, vert_offset, hor_offset=200):
        a_img = 255 - self.load_text_image(img.shape, text_file_path, vert_offset, hor_offset)

        img = img - (density * a_img).astype('int')
        return img.clip(0, 255)

    def combine_file(self, args, a_path, b_path, bg_path, output_path, smpl_path):

        # open files and invert text
        bg_img = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE).astype('int')

        # random horizontal flip
        if self.random_bool():
            bg_img = cv2.flip(bg_img, 0)

        # random vertical flip
        if self.random_bool():
            bg_img = cv2.flip(bg_img, 1)

        train_img = np.full(bg_img.shape, 255)

        img = bg_img.copy()
        for path, v_offset in [(a_path, 100), (b_path, 1000)]:
            img = self.merge_with_text(img, path, 0.3, v_offset)
            train_img = self.merge_with_text(train_img, path, 1, v_offset)

            cv2.imwrite(output_path, img)
            cv2.imwrite(smpl_path, train_img)


if __name__ == '__main__':
    random.seed(123)
    args = parse_args()
    CombineMain().main(args)
    print('done')

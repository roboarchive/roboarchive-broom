import os
import random

import cv2
import numpy as np

from gen_textures import add_noise, texture, blank_image
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

        a = lst[::3]
        b = lst[1::3]
        c = lst[2::3]

        text_files = list(zip(a, b, c))

        if args.index:
            text_files = text_files[args.index:args.index + 1]

        for i, chunk in enumerate(text_files):
            paths = [os.path.join(args.data_dir, self.SRC_DIR, p) for p in chunk]

            fname = 'combined-{}.png'.format(i)
            smpl_path = os.path.join(args.data_dir, self.SMPL_DIR, fname)

            bg_path = self.get_random_bg()
            output_path = os.path.join(args.data_dir, self.DST_DIR, fname)

            print('Processing {}/{}'.format(i, len(text_files)))

            self.combine_file(args, bg_path, output_path, smpl_path, *paths)

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

    def combine_file(self, args, bg_path, output_path, smpl_path, *text_paths):

        # open files and invert text
        raw_image = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE).astype('int')

        h, w = raw_image.shape

        # generate random noise
        noise = 160 - add_noise(texture(blank_image(background=125, height=4096, width=4096),
                                        sigma=4), sigma=10).astype('float')[:h, :w]
        noise = (random.random() * noise).astype('int')

        raw_image = raw_image + noise
        raw_image = raw_image.clip(0, 255)

        # random horizontal flip
        if self.random_bool():
            raw_image = cv2.flip(raw_image, 0)

        # random vertical flip
        if self.random_bool():
            raw_image = cv2.flip(raw_image, 1)

        # create a clean training image
        clean_image = np.full(raw_image.shape, 255)

        # save reference to raw image
        for i, path in enumerate(text_paths):
            v_offset = 100 + i * 1250
            density = 0.2 + random.random() * 0.3
            raw_image = self.merge_with_text(raw_image, path, density, v_offset)
            clean_image = self.merge_with_text(clean_image, path, 1, v_offset)

        cv2.imwrite(output_path, raw_image)
        cv2.imwrite(smpl_path, clean_image)


if __name__ == '__main__':
    random.seed(123)
    args = parse_args()
    CombineMain().main(args)
    print('done')

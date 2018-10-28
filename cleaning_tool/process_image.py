#!/usr/bin/env python3
import argparse

import numpy as np

import cv2
from cnn import get_cnn
from common.split_image import slice_tile
from train_cnn import add_common_arguments, configure_backend
from common.utils import display


class FileProcessor:
    def process(self, args, bg_color=0):
        """
        Scale image to width 1024, convert to grayscale and than slice by tiles.
        It's possible to slice image with padding and each tile will contain pixels from surrounding tiles
        """
        configure_backend(args)
        cnn = get_cnn(args)

        tile_size = cnn.tile_size
        img = cv2.imread(args.input_file, cv2.IMREAD_GRAYSCALE)
        assert img is not None, f'No file: {args.input_file}'

        h, w = img.shape

        if args.scale:
            width = args.scale
            height = int(width * h / w)
        else:
            width, height = w, h
        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)

        output_img = np.zeros(img.shape)

        i = 0
        j = 0

        while tile_size * (i * 1) < (width + tile_size):
            while tile_size * (j + 1) < (height + tile_size):
                tile, orig_size = slice_tile(img, i, j, tile_size, args.padding, bg_color=bg_color)
                if not orig_size[0] or not orig_size[1]:
                    j += 1
                    continue

                # convert to CNN format
                cnn_tile = cnn.input_img_to_cnn(tile, tile_size, args.padding)

                # process output
                print('processing tile {}, {}'.format(i, j))
                # TODO: fix this, we should be able to batch processing
                out_arr = cnn.process_tile(cnn_tile)

                # convert to img format
                out_tile = cnn.cnn_output_to_img(out_arr, tile_size)

                output_img[
                    j * tile_size : (j + 1) * tile_size, i * tile_size : (i + 1) * tile_size
                ] = out_tile[: orig_size[0], : orig_size[1]]

                j += 1
            i += 1
            j = 0

        cv2.imwrite(args.output_file, output_img)
        if args.display:
            display(output_img)


def fake_processing(tile):
    return tile[16:48, 16:48].flatten()


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument(
        '-i', '--input-file', dest='input_file', help='Input file to process', required=True
    )

    parser.add_argument(
        '-o',
        '--output-file',
        dest='output_file',
        help='Processed output file',
        default='output.png',
    )
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument('--padding', default=0, type=int)  # maybe 0
    parser.add_argument('--scale', default=None, type=int)  # maybe 0
    return parser.parse_args()


if __name__ == '__main__':
    p = FileProcessor()
    p.process(parse_args())

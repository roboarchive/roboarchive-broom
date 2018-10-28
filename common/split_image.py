#!/usr/bin/env python3
import argparse
import os
import sys

import numpy

import cv2


def slice_tile(img, i, j, tile_size, padding, bg_color=0):
    """
    Slice a tile from the bigger image, if the tile is less then expected size (it's slices at the edge of the image),
    it's extended with blank space of the given background color
    """
    full_size = tile_size + padding * 2

    top_offset, left_offset = (0, 0)

    from_y = j * tile_size - padding
    to_y = (j + 1) * tile_size + padding
    from_x = i * tile_size - padding
    to_x = (i + 1) * tile_size + padding

    if from_y < 0:
        top_offset = abs(from_y)
        from_y = 0

    if from_x < 0:
        left_offset = abs(from_x)
        from_x = 0

    tile = img[from_y:to_y, from_x:to_x]
    h, w = tile.shape
    if h < full_size or w < full_size:
        bg = numpy.full((full_size, full_size), bg_color, numpy.uint8)
        bg[top_offset : top_offset + h, left_offset : left_offset + w] = tile
        # print(f'Return: {h}/{w} => {tile.shape} IN {i} {j} / Color: {bg_color}')
        return bg, (h, w)
    else:
        return tile, (h, w)


def split(filename, scale_to_width=1024, tile_size=32, padding=16, output_dir='split', bg_color=0):
    """
    Scale image to width 1024, convert to grayscale and than slice by tiles.
    It's possible to slice image with padding and each tile will contain pixels from surrounding tiles
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape

    width = scale_to_width
    height = int(width * h / w)
    resized = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)

    i = 0
    j = 0

    while tile_size * i < width:
        while tile_size * j < height:
            tile_file = os.path.join(output_dir, 'tile-{}-{}.png'.format(i, j))

            tile_img, _orig = slice_tile(resized, i, j, tile_size, padding, bg_color=bg_color)

            cv2.imwrite(tile_file, tile_img)
            j += 1
        i += 1
        j = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', default='split', dest='out_dir', help='Output directory')
    parser.add_argument('-i', '--image-file', required=True, dest='image_file', help='Image file')
    parser.add_argument(
        '-t', '--tile-size', dest='tile_size', default='32', type=int, help='Tile size'
    )
    parser.add_argument(
        '-p',
        '--padding',
        dest='padding',
        default='16',
        type=int,
        help='Additional padding around the tile',
    )
    parser.add_argument(
        '-b',
        '--background-color',
        dest='background_color',
        default='0',
        type=int,
        help='Background color from 0 to 255',
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if os.path.isfile(args.image_file):
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        split(
            args,
            args.image_file,
            tile_size=args.tile_size,
            padding=args.padding,
            output_dir=args.out_dir,
            bg_color=args.background_color,
        )
    else:
        sys.exit('File {} does not exist'.format(args.image_file))

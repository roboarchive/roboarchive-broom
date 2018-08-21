#!/usr/bin/env python3
import argparse
import datetime
import functools
import os
from glob import glob

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import cv2
from cnn import get_cnn
from split_image import slice_tile


class XTileLoader:
    """
    Load square tiles with channel
    """

    def __init__(self, args, cnn, tiles_dir, tile_size):
        self.args = args
        self.cnn = cnn
        self.tiles_dir = tiles_dir
        self.tile_size = tile_size

    def get_shape(self):
        return (self.tile_size, self.tile_size, 1)

    def load(self):
        tiles_list = self.file_name()
        shape = (len(tiles_list),) + self.get_shape()
        train_data = np.zeros(shape)
        for i, fname in enumerate(tiles_list):
            path = os.path.join(self.tiles_dir, fname)
            tile = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            tile = tile.reshape(self.get_shape())
            train_data[i] = tile

        train_data = train_data.astype('float32')
        train_data /= 255

        return train_data

    def file_names(self):
        if not self.args.filter:
            src = os.listdir(self.tiles_dir)
        else:
            src = glob(os.path.join(self.tiles_dir, self.args.filter))
        for fname in src:
            yield os.path.basename(fname)


class SplitTileLoader(XTileLoader):
    def split_image(self, path):
        tile_size = self.tile_size
        print(f'Load: {path}')
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        i = 0
        j = 0

        while tile_size * (i * 1) < (width + tile_size):
            while tile_size * (j + 1) < (height + tile_size):
                tile, orig_size = slice_tile(img, i, j, tile_size, 0, bg_color=255)
                if not orig_size[0] or not orig_size[1]:
                    j += 1
                    continue
                # convert to CNN format
                cnn_tile = self.cnn.input_img_to_cnn(tile, tile_size)
                yield cnn_tile
                j += 1
            i += 1
            j = 0

    def load(self):
        for fname in self.file_names():
            yield from self.split_image(os.path.join(self.tiles_dir, fname))


class YTileLoader(XTileLoader):
    """
    Load tile as flat array without channel
    """

    def get_shape(self):
        return (self.tile_size * self.tile_size,)


def load_data(args, cnn, x_path, y_path):
    """
    Check raw/clean (X/Y) data consistency and load data array
    """
    raw_files = sorted(os.listdir(x_path))
    clean_files = sorted(os.listdir(y_path))

    assert raw_files == clean_files, 'X/Y files are not the same'
    x_train = list(SplitTileLoader(args, cnn, x_path, 256).load())
    y_train = list(SplitTileLoader(args, cnn, y_path, 256).load())
    return x_train, y_train


def configure_backend(args):
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    if args.cpu:
        config.device_count['GPU'] = 0
    sess = tf.Session(config=config)
    K.set_session(sess)


class Batch:
    def __init__(self, size, input_size):
        self.size = size
        self.input_size = input_size
        self.imgs = []

    def append(self, img):
        self.imgs.append(img)
        if len(self.imgs) >= self.size:
            b = np.zeros((self.size, *self.input_size, 1), dtype='float32')
            for idx, img in enumerate(self.imgs):
                b[idx] = img
            return b


def generic_skip(x_data, y_data, cnt, skip_if_zero=True):
    c = 0
    out_x, out_y = [], []
    for x, y in zip(x_data, y_data):
        c += 1
        equal_zero = c % cnt == 0
        if skip_if_zero and equal_zero:
            continue
        elif not skip_if_zero and not equal_zero:
            continue
        out_x.append(x)
        out_y.append(y)
    return out_x, out_y


def data_skip(x_data, y_data, cnt):
    return generic_skip(x_data, y_data, cnt, skip_if_zero=True)


def data_take(x_data, y_data, cnt):
    return generic_skip(x_data, y_data, cnt, skip_if_zero=False)


def data_generator(args, model, skip=None, take=None):
    def _batch(inp):
        b = np.zeros((1, *model.input_size, 1), dtype='float32')
        b[0] = inp
        return b

    assert any([skip, take]), 'You should have skip or take, or must know what your are doing'
    x_train, y_train = load_data(args, model, 'train/raw/samples', 'train/clean/samples')
    if skip:
        x_train, y_train = data_skip(x_train, y_train, skip)
        print(f'Skip len: {len(x_train)}')
    if take:
        x_train, y_train = data_take(x_train, y_train, take)
        print(f'Take len: {len(x_train)}')
    print(f'Splitted samples length: {len(x_train)}')
    bx, by = Batch(args.batch_size, model.input_size), Batch(args.batch_size, model.input_size)

    while True:
        c = 0
        for x, y in zip(x_train, y_train):
            outx, outy = bx.append(x), by.append(y)
            c += 1
            if outx is not None:
                yield outx, outy
                bx, by = (
                    Batch(args.batch_size, model.input_size),
                    Batch(args.batch_size, model.input_size),
                )
            if args.display:
                display(x)
                display(y)


VALIDATION_RATE = 8


def train(args):
    configure_backend(args)

    # np.random.seed(123)  # for reproducibility
    print('Creating CNN')

    cnn = get_cnn(args)
    model_checkpoint = ModelCheckpoint(
        args.weights_file,
        monitor=args.monitor,
        verbose=1,
        save_best_only=args.best,
        period=args.period,
    )
    run_name = datetime.datetime.now().strftime(f'%Y/%m/%d/%H_%M_lr_{args.learning_rate}')
    callbacks = [model_checkpoint, TensorBoard(log_dir=f'tensorboard_log/{run_name}')]
    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', verbose=1, patience=50))

    cnn.model.fit_generator(
        data_generator(args, cnn, skip=VALIDATION_RATE),
        validation_data=data_generator(args, cnn, take=VALIDATION_RATE),
        validation_steps=int(args.epoch_steps / args.batch_size),
        steps_per_epoch=args.epoch_steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )


def add_common_arguments(parser):
    parser.add_argument(
        '-w', '--weights', dest='weights_file', help='Save weights to file', default='weights.h5'
    )
    parser.add_argument(
        '-c',
        '--cnn',
        dest='cnn_name',
        choices=['simple', 'unet'],
        help='CNN',
        default=os.environ.get('CNN_NAME') or 'unet',
    )
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--decay', default=0, type=float)


def display(*images):
    for image in images:
        if len(image.shape) == 4:
            image = image[0, :, :, 0]
        plt.imshow(image, cmap=cm.gray)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--monitor', default='val_loss')
    parser.add_argument('--period', default=1, type=int)
    parser.add_argument('-e', '--epochs', default=100000, type=int)
    parser.add_argument('--epoch-steps', default=16, type=int)
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument('-f', '--filter')
    parser.add_argument('--early-stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=50)

    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())

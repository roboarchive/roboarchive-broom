import os
import pickle
import random
from glob import glob
from itertools import chain

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import cv2
from common.split_image import slice_tile

VALIDATION_RATE = 8
SAME_RATE = 7
TRAIN = os.environ.get('TRAIN_ROOT', 'train')
CACHE = os.environ.get('CACHE', './')


class XTileLoader:
    """
    Load square tiles with channel
    """

    def __init__(self, args, cnn, tiles_dir):
        self.args = args
        self.cnn = cnn
        self.tiles_dir = tiles_dir
        self.tile_size = args.tile_size

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


def load_img(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def split_image(img, model, tile_size):
    height, width = img.shape
    i, j = 0, 0
    while tile_size * (i * 1) < (width + tile_size):
        while tile_size * (j + 1) < (height + tile_size):
            tile, orig_size = slice_tile(img, i, j, tile_size, 0, bg_color=255)
            if not orig_size[0] or not orig_size[1]:
                j += 1
                continue
            # convert to CNN format
            cnn_tile = model.input_img_to_cnn(tile, tile_size)
            yield cnn_tile
            j += 1
        i += 1
        j = 0


class SplitTileLoader(XTileLoader):
    def split_image(self, path):
        tile_size = self.tile_size
        print(f'Load: {path}')
        img = load_img(path)
        yield from split_image(img, self.cnn, tile_size)

    def load(self):
        for fname in self.file_names():
            yield from self.split_image(os.path.join(self.tiles_dir, fname))


class Batch:
    def __init__(self, size, input_size):
        self.size = size
        self.input_size = input_size
        self.imgs = []

    def append(self, img):
        self.imgs.append(img)

    @property
    def is_ready(self):
        return len(self.imgs) >= self.size

    def get_data(self, reset=True):
        assert self.is_ready
        b = np.zeros((self.size, *self.input_size, 1), dtype='float32')
        assert len(b.shape) == 4
        for idx, img in enumerate(self.imgs):
            b[idx] = img
        if reset:
            self.reset()
        return b

    def reset(self):
        self.imgs = []


dg = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # brightness_range=(-0.3, 0.3),
    horizontal_flip=False,
    vertical_flip=False,
)


class PickledCache:
    def init_cache(self, size):
        if hasattr(self, 'dct'):
            return
        self.path = '{}/cache.{}.pickle'.format(CACHE, size)
        if not os.path.exists(self.path):
            self.dct = {}
        else:
            with open(self.path, 'rb') as f:
                self.dct = pickle.load(f)

    def check_size(self, size):
        self.init_cache(size)
        _size = self.get('_size')
        if _size is None:
            self.set('_size', size)
        else:
            assert size == _size, f'Cache has wrong tile size: {size} vs {_size}'

    def get(self, key):
        return self.dct.get(key)

    def set(self, key, value):
        self.dct[key] = value
        self.save()

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.dct, f)


cache = PickledCache()


class ImageWrapper:
    def __init__(self, path, args, model, trans=None, validation=True):
        self.path = path
        self.args = args
        self.model = model
        self.ready = False
        self.trans = trans
        self.validation = validation
        cache.check_size(args.tile_size)
        if cache.get(path):
            self.original_shape, self.len_tiles = cache.get(path)
        else:
            image = load_img(path)
            self.original_shape = image.shape
            # image = self.post_load(image)
            self.len_tiles = len(list(self.load_splitted()))
            cache.set(path, (self.original_shape, self.len_tiles))

    def load_splitted(self):
        return list(split_image(load_img(self.path), self.model, self.args.tile_size))

    def set_shuffle(self, order):
        self.shuffle_order = order

    def get_transformated(self, trans):
        return ImageWrapper(self.path, self.args, self.model, trans, validation=False)

    def conv_to(self, img):
        img = img.astype('float32')
        img = img.reshape((*self.original_shape, 1))
        img /= 255
        return img

    def conv_from(self, img):
        img = img.reshape(self.original_shape)
        img *= 255
        img = img.clip(0, 255)
        img = img.astype(np.uint8)
        return img

    def post_load(self, image):
        if self.trans:
            return self.conv_from(dg.apply_transform(self.conv_to(image), self.trans))
        return image

    def shuffle(self):
        order = self.shuffle_order
        all_tiles = [y for x, y in sorted(zip(order, self.load_splitted()))]
        validation_tiles = []
        assert len(all_tiles) > 1
        if self.validation:
            num_validation = max(int(len(all_tiles) / VALIDATION_RATE), 1)
            print(f'Add {num_validation} samples as validation samples')
            for i in range(num_validation):
                validation_tiles.append(all_tiles.pop())
        return all_tiles, validation_tiles

    @property
    def data_generator(self):
        out, _ = self.shuffle()
        return out

    @property
    def validation_data_generator(self):
        _, out = self.shuffle()
        return out

    def debug(self, tile, count):
        if self.args.display:
            print(f'Show: {self.path} {count}')
            # display(tile)

    def get_generation(self):
        return SplitTileLoader(self.args, self.model, None).split_image(self.path)


class ImagePair:
    def __init__(self, args, model, src, dst, should_trans=False):
        assert os.path.basename(src) == os.path.basename(dst)
        self.src_path = src
        self.dst_path = dst
        self.args = args
        self.model = model
        self.src = ImageWrapper(src, args, model)
        self.dst = ImageWrapper(dst, args, model)
        if should_trans:
            td = dg.get_random_transform(self.src.original_shape)
            self.src = self.src.get_transformated(td)
            self.dst = self.dst.get_transformated(td)
        self.set_shuffle(self.src, self.dst)

    def set_shuffle(self, x, y):
        order = list(range(x.len_tiles))
        random.shuffle(order)
        x.set_shuffle(order), y.set_shuffle(order)

    def data_generator(self):
        for x, y in zip(self.src.data_generator, self.dst.data_generator):
            yield x, y

    def validation_generator(self):
        for x, y in zip(self.src.validation_data_generator, self.dst.validation_data_generator):
            yield x, y

    def transformated(self):
        return ImagePair(self.args, self.model, self.src_path, self.dst_path, should_trans=True)


class DataSource:
    """
    load file names: (x, y)
    load tiles from file
    optional - apply transformation and return tiles
    get_data_generator - shouldn't return validation tiles
    get_validation_generator - should return same tiles

    [default_images] + [same_images] + [generated_images]
    + each should have validation_data (tiles that aren't used for data generator)
    """

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.src_dir = [f'{TRAIN}/raw/samples', f'{TRAIN}/combined_raw']
        self.dst_dir = [f'{TRAIN}/clean/samples', f'{TRAIN}/combined_clean']
        self.pure_images = []
        self.generated_images = []
        self.fill_pure_images()

    def get_dir_file_names(self, directory):
        if not self.args.filter:
            src = sorted(os.listdir(directory))
        else:
            src = [os.path.basename(x) for x in glob(os.path.join(directory, self.args.filter))]
            src = sorted(src)
        return src

    def full_path(self, directory, src):
        return [os.path.join(directory, x) for x in src]

    def file_names(self):
        out = []
        for src_dir, dst_dir in zip(self.src_dir, self.dst_dir):
            src = self.get_dir_file_names(src_dir)
            dst = self.get_dir_file_names(dst_dir)
            assert src == dst
            out.append(zip(self.full_path(src_dir, src), self.full_path(dst_dir, dst)))
        return chain(*out)

    @property
    def ideal_steps(self):
        tiles = sum(x.src.len_tiles for x in self.pure_images)
        return int(tiles / self.args.batch_size / self.args.transformated) + 1

    def fill_pure_images(self):
        for x, y in self.file_names():
            self.pure_images.append(ImagePair(self.args, self.model, x, y))

    def generate_images(self):
        if self.args.no_generated:
            return
        print('Generate new images')
        self.generated_images = [x.transformated() for x in self.pure_images]

    def data_generator(self):
        x, y = (
            Batch(self.args.batch_size, self.model.input_size),
            Batch(self.args.batch_size, self.model.input_size),
        )
        while True:
            for img in chain(self.pure_images, self.generated_images):
                for x_tile, y_tile in img.data_generator():
                    x.append(x_tile), y.append(y_tile)
                    if x.is_ready:
                        yield x.get_data(reset=True), y.get_data(reset=True)
            self.generate_images()

    def validation_generator(self):
        x, y = (
            Batch(self.args.batch_size, self.model.input_size),
            Batch(self.args.batch_size, self.model.input_size),
        )
        while True:
            for img in chain(self.pure_images, self.generated_images):
                for x_tile, y_tile in img.validation_generator():
                    x.append(x_tile), y.append(y_tile)
                    if x.is_ready:
                        yield x.get_data(reset=True), y.get_data(reset=True)

    def trans_data_generator(self):
        x, y = (
            Batch(self.args.batch_size, self.model.input_size),
            Batch(self.args.batch_size, self.model.input_size),
        )
        while True:
            for img in chain(self.generated_images):
                x_tile, y_tile = img.next_data()
                x.append(x_tile), y.append(y_tile)
                if x.is_ready:
                    yield x.get_data(reset=True), y.get_data(reset=True)

                x_tile, y_tile = img.next_data()
                x.append(x_tile), y.append(y_tile)
                if x.is_ready:
                    yield x.get_data(reset=True), y.get_data(reset=True)
            self.generate_images()

import pytest
import numpy as np

from common.split_image import slice_tile


@pytest.fixture
def img():
    a = np.zeros((10, 10), np.uint8)
    for i in range(10):
        for j in range(10):
            a[i, j] = i * 10 + j
    return a


def test_slice_tile(img):
    tile, (h, w) = slice_tile(img, 0, 0, tile_size=3, padding=0)

    expected = np.array([[0,  1,  2],
                         [10, 11, 12],
                         [20, 21, 22]], np.uint8)

    assert h == 3, h
    assert w == 3, w
    assert np.array_equal(tile, expected), tile


def test_slice_tile2(img):
    tile, (h, w) = slice_tile(img, 0, 0, tile_size=3, padding=1, bg_color=7)
    expected = np.array([[7, 7, 7, 7, 7],
                         [7, 0, 1, 2, 3],
                         [7, 10, 11, 12, 13],
                         [7, 20, 21, 22, 23],
                         [7, 30, 31, 32, 33]], np.uint8)

    assert h == 4, h
    assert w == 4, h
    assert np.array_equal(tile, expected), tile


def test_slice_tile3(img):
    tile, (h, w) = slice_tile(img, 3, 3, tile_size=3, padding=1, bg_color=7)
    expected = np.array([[88, 89,  7,  7,  7],
                         [98, 99,  7,  7,  7],
                         [7,  7,  7,  7,  7],
                         [7,  7,  7,  7,  7],
                         [7,  7,  7,  7,  7]], np.uint8)

    assert h == 2, h
    assert w == 2, w
    assert np.array_equal(tile, expected), tile

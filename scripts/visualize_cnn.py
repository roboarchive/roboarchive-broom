import argparse
import cv2

from keras import activations

from cnn import get_cnn

from vis.visualization import visualize_activation, visualize_cam, visualize_saliency, visualize_activation_with_losses
from vis.utils import utils
from matplotlib import pyplot as plt

from train_cnn import add_common_arguments


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    cnn = get_cnn(args)
    cnn.load(args.weights_file)

    idx = -1

    model = cnn.model
    model.layers[idx].activation = activations.linear
    model = utils.apply_modifications(model)

    img = cv2.imread('tile.png', cv2.IMREAD_GRAYSCALE)
    img = img.reshape(img.shape + (1,))
    img = img.astype('float32')
    img /= 255

    f, ax = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            print('plotting {} {}'.format(i, j))

            idx = 10 + i + j
            layer = model.layers[idx]

            f = layer.output_shape[3] - 1
            plot = visualize_cam(model, idx, filter_indices=[0, f], seed_input=img)
            ax[i, j].imshow(plot)

    plt.show()

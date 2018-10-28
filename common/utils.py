import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def display(*images):
    if len(images) == 2 and len(images[0]) == len(images[1]):
        ims = []
        for x, y in zip(*images):
            ims.append(x)
            ims.append(y)
        return display(*ims)

    if len(images[0].shape) == 4:
        for i in images:
            display(*i)
        return

    if len(images) == 1:
        sp, axarray = plt.subplots(len(images))
    else:
        ln = len(images)
        if ln % 2 == 0:
            rows = int(ln / 2)
        else:
            rows = int(ln / 2) + 1
        sp, axarray = plt.subplots(rows, 2)

    for image, (_idx, fig) in zip(images, np.ndenumerate(axarray)):
        if len(image.shape) == 3:
            image = image[:, :, 0]
        if image.dtype == 'float32':
            _max = 1
        else:
            _max = 255
        fig.imshow(image, cmap=cm.gray, vmin=0, vmax=_max)
    plt.show()

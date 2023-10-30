import numpy as np


def Normalize(images, type, show=0):
    show = 0
    if show == 1:
        print("min", np.min(images))
        print(" max", np.max(images))
    if type == 1:  # values from 0 to 1 (2)
        im_n = (images - images.min()) / (images.max() - images.min())
    if type == 2:  # random
        im_n = (images - images.min()) / (images.max() - images.min()) * 3
    if type == 3:  # zero mean, unit variance (none)
        im_n = (images - images.mean())
        im_n = im_n / im_n.std()
    if type == 4:  # normalized values (3)
        im_n = images / images.std()
    if type == 5:  # values from -1 to 1 (15)
        im_n = (images - images.min()) / (images.max() - images.min())
        im_n = (im_n - 0.5) * 2
    if type == 6:
        im_n = ((images - images.min()) / (images.max() - images.min())) - 1.001
    if type == 7:
        im_n = images / 255
    if show == 1:
        print("min", np.min(im_n))
        print(" max", np.max(im_n))
    return im_n

import os, sys
from PIL import Image
import scipy.misc
from glob import glob
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data

prefix = './Datas/'


def get_img(img_path, crop_h, resize_h):
    img = scipy.misc.imread(img_path).astype(np.float)
    # crop resize
    #@crop_w = crop_h
    # resize_h = 64
    #resize_w = resize_h
    #h, w = img.shape[:2]
    #j = int(round((h - crop_h) / 2.))
    #i = int(round((w - crop_w) / 2.))
    #cropped_image = scipy.misc.imresize(img[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])

    return (np.array(img) / 127.5)-1


class imagedata():
    def __init__(self, datapath):
        self.z_dim = 100
        self.c_dim = 20
        self.size = 32
        self.channel = 1
        self.datapath=datapath
        self.data = glob(os.path.join(self.datapath, '*.png'))

        self.batch_count = 0

    def __call__(self, batch_size):
        batch_number = len(self.data) / batch_size
        if self.batch_count < batch_number - 1:
            self.batch_count += 1
        else:
            self.batch_count = 0

        path_list = self.data[self.batch_count * batch_size:(self.batch_count + 1) * batch_size]
        batch = [get_img(img_path, self.size, self.size) for img_path in path_list]
        batch_imgs = np.array(batch).astype(np.float32)
        # fig = self.data2fig(batch_imgs[:16,:,:])
        # plt.savefig('out_face/{}.png'.format(str(self.batch_count).zfill(3)), bbox_inches='tight')
        # plt.close(fig)

        return batch_imgs,path_list[0]
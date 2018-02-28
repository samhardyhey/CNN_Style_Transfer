import numpy as np
from PIL import Image
from os import listdir, mkdir, getcwd
import glob
import os


def import_image(file_name, height, width):
    # import image, cast as np array

    image = Image.open(file_name)  # 1.0 retrieve all input image references
    image = image.resize((height, width), Image.ANTIALIAS)  # downsample
    image = np.asarray(image, dtype='float32')  # cast to np array
    image = np.expand_dims(image, axis=0)  # add placeholder dimension

    image[:, :, :, 0] -= 103.939  # subtract mean RGB values from ImageNet
    image[:, :, :, 1] -= 116.779
    image[:, :, :, 2] -= 123.68
    image = image[:, :, :, ::-1]  # flip RGB pixel ordering

    return image


def export_image(img_array, save_dir, height, width):
    # export np array as jpeg image

    img_array = img_array.reshape((height, width, 3))  # reshape
    img_array = img_array[:, :, ::-1]  # flip RGB pixel ordering
    img_array[:, :, 0] += 103.939  # add mean RGB values from ImageNet
    img_array[:, :, 1] += 116.779
    img_array[:, :, 2] += 123.68
    img_array = np.clip(img_array, 0, 255).astype('uint8')  # trim pixels
    Image.fromarray(img_array).save(save_dir + '.jpeg', "jpeg")  # save


def retrieve_inputs():
    # retrieve all style and content image file names/dir paths
    content_names = list(os.path.splitext(each)[0] for each in listdir("./input/contents"))
    content_paths = glob.glob('./input/contents/*')
    style_names = list(os.path.splitext(each)[0] for each in listdir("./input/styles"))
    style_paths = glob.glob('./input/styles/*')

    return content_names, content_paths, style_names, style_paths

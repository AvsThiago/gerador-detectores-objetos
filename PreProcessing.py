#!/usr/bin/env python

"""
 These methods provides ways to pre processing images,
 the main objective is to encapsulate methods that will
 be used in the main projects.

 @author  Thiago da Silva Alves
 @version 1.0, 23/12/16
"""

import os
import cv2 as cv
import numpy as np

def compare_images(image1, image2):
    """Compare two images and return they are"""
    try:
        return image1.shape == image2.shape and \
                not(np.bitwise_xor(image1, image2).any())
    except Exception as e:
        print(e)
        return False

def remove_image(image_path):
    """Remove one file from it's directory"""
    os.remove(image_path)

def rename_image(path, old_name, new_name):
    """Rename one image from it's directory"""
    os.rename(os.path.join(path, old_name), os.path.join(path, new_name))

def is_corrupted(image_path):
    """Verify if the image passed by parameter is corrupted"""
    try:
        img_loaded = cv.imread(image_path)
        # If some error occurs when trying to fetch images' shape it's corrupted
        shape = img_loaded.shape
        return False
    except:
        return True

def rename_images_path(path, prefix="img", out_ext=".png",
                        extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """Rename all images in a path, using a prefix concatenated with a number"""
    files = [fl for fl in os.listdir(path) if fl.lower().endswith(extensions)]
    for i, nome in enumerate(files):
        rename_image(path, nome, ''.join([prefix, str(i), out_ext]))

def remove_corrupted_img_path(path, extensions=(".jpg", ".jpeg", ".png",
                                                ".bmp")):
    """Remove all corrupted images from a folder"""
    files = [fl for fl in os.listdir(path) if fl.lower().endswith(extensions)]
    for nome in files:
        if is_corrupted(os.path.join(path, nome)):
            remove_image(os.path.join(path, nome))

def remove_same_images(path_images_to_compare,
                        path_images_to_be_compared,
                        extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """Compare if two images are equal, if yes, ther second is removed"""
    to_compare = [fl for fl in os.listdir(path_images_to_compare)
                            if fl.lower().endswith(extensions)]
    to_be_compared = [fl for fl in os.listdir(path_images_to_be_compared)
                                if fl.lower().endswith(extensions)]
    for i in to_compare:
        for j in to_be_compared:
            dir_img1 = os.path.join(path_images_to_compare, i)
            dir_img2 = os.path.join(path_images_to_be_compared, j)
            if os.path.isfile(dir_img1) and os.path.isfile(dir_img2):
                img1 = cv.imread(dir_img1)
                img2 = cv.imread(dir_img2)
                if compare_images(img1, img2):
                    remove_image(dir_img2)

def convert_to_gray(path, image_name):
    """Convert an image to gray scale"""
    img = cv.imread(os.path.join(path, image_name), 0)
    cv.imwrite(os.path.join(path, image_name), img)

def convert_to_gray_path(path, extensions=(".jpg", ".jpeg", ".png",
                                            ".bmp")):
    """Convert all images from a path to gray"""
    for i in os.listdir(path):
        convert_to_gray(path, i)

def resize_image(path, image_name, max_size=500.0):
    """Resize an image where it's bigger side will be resized to the value of
       the parameter max_size and the another side will respect the new ratio"""
    img = cv.imread(os.path.join(path, image_name))
    height, width = img.shape[:2]
    if height > width:
        ratio = height / max_size
        w = int(width / ratio)
        res = cv.resize(img, (w, int(max_size)), interpolation=cv.INTER_AREA)
        cv.imwrite(os.path.join(path, image_name), res)
    else:
        ratio = width / max_size
        h = int(height / ratio)
        res = cv.resize(img, (int(max_size), h), interpolation=cv.INTER_AREA)
        cv.imwrite(os.path.join(path, image_name), res)

def resize_image_path(path, max_size=500.0, extensions=(".jpg", ".jpeg", ".png",
                                                    ".bmp")):
    """Resize all images of a path to a given max size."""
    for i in os.listdir(path):
        resize_image(path, i, max_size)

"""

 This class provides many methods for pre processing of images,
 the main objective is to encapsulate methods that will
 be used in the main projects.

 @author  Thiago da Silva Alves
 @version 1.0, 23/12/16

"""

import os
import cv2 as cv

def compare_images(image1, image2):
    try:
        return image1.shape == image2.shape
                and not(np.bitwise_xor(image1, image2).any())
    except Exception as e:
        return False

def remove_file(image_path):
    os.remove(path)

def rename_image(path, old_name, new_name):
    os.rename(os.path.join(path, old_name), os.path.join(path, new_name))

def is_corrupted(image_path):
    try:
        img_loaded = cv.imread(image_path)
        # If some error occurs when trying to fetch images' shape it's corrupted
        shape = img_loaded.shape
        return False
    except:
        return True

def rename_images_path(path, prefix, out_ext=".png"
                        extensions=(".jpg", ".jpeg", ".png", ".gif", ".bmp")):
    files = [fl for fl in os.listdir(path) if fl.lower().endswith(extension)]
    for i, nome in enumerate(files):
        rename_image(path, nome, ''.join([prefix, str(i), out_ext]))

def remove_corrupted_img_path(path, extensions=(".jpg", ".jpeg", ".png",
                                                ".gif", ".bmp")):
    files = [fl for fl in os.listdir(path) if fl.lower().endswith(extension)]
    for nome in files:
        if is_corrupted(os.path.join(path, nome)):
            remove_file(os.path.join(path, nome))

def remove_same_images(path_images_to_compare,
                        path_images_to_be_compared
                        extensions=(".jpg", ".jpeg", ".png", ".gif", ".bmp")):
    to_compare = [fl for fl in os.listdir(path)
                            if fl.lower().endswith(extension)]
    to_be_compared = [fl for fl in os.listdir(path)
                                if fl.lower().endswith(extension)]
    for i in to_compare:
        for j in to_be_compared:
            dir_img1 = os.path.join(path_images_to_compare, i)
            dir_img2 = os.path.join(path_images_to_be_compared, j)
            img1 = cv.imread(dir_img1)
            img2 = cv.imread(dir_img2)
            if compare_images(img1, img2):
                remove_image(dir_img2)

def convert_to_gray(path, image_name):
    img = cv.imread(os.path.join(path, image_name), 0)
    cv.imwrite(path, img)

def convert_to_gray_path(path, extensions=(".jpg", ".jpeg", ".png",
                                            ".gif", ".bmp")):
    for i in os.path.listdir(path):
        convert_to_gray(path, i)

def resize_image(path, image_name, max_size=500.0):
    img = cv2.imread(os.path.join(path, i))
    height, width = img.shape[:2]
    if height > width:
        ratio = height / max_size
        w = int(width / ratio)
        res = cv2.resize(img, (w, int(max_size)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(src + i, res)
    else:
        ratio = width / max_size
        h = int(height / ratio)
        res = cv2.resize(img, (int(max_size), h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(src + i, res)

def resize_image_path(path, max_size, extensions=(".jpg", ".jpeg", ".png",
                                                    ".gif", ".bmp")):
    for i in os.path.listdir(path):
        resize_image(path, i, max_size)

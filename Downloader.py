#!/usr/bin/env python

"""
 This class provides many methods for download images,
 from a list of urls that can be in a file or a website.
 After downloaded, the user can automatcally resize,
 rename, transform to gray scale up or down and remove
 default stock images.

 @author  Thiago da Silva Alves
 @version 1.0, 26/12/16
"""

from urllib.parse import urlparse
from requests import exceptions
from requests import get
import mimetypes
import os
import argparse
import PreProcessing as prp


def name_from_url(response, url):
    """Gets an url and by this searches by image name"""
    parsedResult = urlparse(url)
    file_name = os.path.split(parsedResult.path)[1]
    file_name_without_ext = os.path.splitext(file_name)[0]
    content_type = response.headers['content-type']
    extension = mimetypes.guess_extension(content_type)
    extension = "" if extension == None else extension
    return ''.join([file_name_without_ext, extension]), extension

def download_and_save(url, save_path, timeout=2,
                    extensions_filter=(".jpg", ".jpeg", ".png", ".bmp")):
    """gets the request"""
    try:
        response = get(url, timeout=timeout)
        if response.status_code == 200: #200 == success
            file_name, extension = name_from_url(response, url)
            print(url)
            if extension in extensions_filter:
                with open(os.path.join(save_path, file_name), "wb") as file:
                    file.write(response.content)
    except exceptions.Timeout:
        print("Timeout was reached for the url: [{}]".format(url))
    except exceptions.TooManyRedirects:
        print("Too many redirects for the url: [{}]".format(url))
    except exceptions.RequestException as e:
        print (e)

def download_images_by_list(file, save_path, timeout=2,
                            extensions_filter=(".jpg", ".jpeg", ".png",
                                                ".bmp")):
    """Using an list of urls, iterates all over them and downloads each image"""
    #if the file is in a local machine
    if os.path.isfile(file):
        url_images = open(file).readlines()
        for i, url in enumerate(url_images):
            if url: # If url content is ""
                download_and_save(url, save_path, timeout, extensions_filter)
    #If the file is in some webside
    else:
        response = get(file, stream=True, timeout=timeout)
        if response.status_code == 200:
            lines = response.iter_lines()
            for  line in lines:
                if line: # If line content is ""
                    download_and_save(line.decode('utf-8'),
                                        save_path, timeout, extensions_filter)

def parse_arguments():
    """Parses arguments choosen by the user."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls',
                        help='File where urls are, can be an site or a file.',
                        required=True)
    parser.add_argument('--out',
                        help='Where the images will be saved.', required=True)
    parser.add_argument('--timeout',
                        help='Maximum time to download each image.', default=2)
    parser.add_argument('--img-extensions',
                        help='Tuple of image extensions that will be accepted.',
                        default=(".jpg", ".jpeg", ".png", ".bmp"))
    parser.add_argument('--prefix',
                        help='Name prefix to save images.',
                        default="img")
    parser.add_argument('--out-extension',
                        help='Extension to use when sava an image.',
                        default=".png")
    parser.add_argument('--default-images',
                        help="Path that contains default images, these will be"
                        " used to verify if the downloaded"
                        " image is a default image and delete it.")
    parser.add_argument('--no-convert-gray',
                        help="Don't convert downloaded images to gray.",
                        action='store_true')
    parser.add_argument('--no-resize',
                        help="Don't resizes each downloaded image.",
                        action='store_true')
    parser.add_argument('--max-size',
                        help="Resizes the bigger side of an image to "
                        "fit in this param.",
                        type=float, default=500.0)
    parser.add_argument('--no-std-names',
                        help="Don't standardize image names using --prefix",
                        action='store_true')

    return parser.parse_args()

def validations(arg):
    """Validations to argumens"""
    if not os.path.isdir(arg.out):
        raise Exception("Out path doesn't exists.")

if __name__ == '__main__':
    try:
        args = parse_arguments()
        validations(args)

        download_images_by_list(args.urls, args.out, args.timeout,
                                args.img_extensions)
        #Verify if there is some corrupted image, if yes, delete it.
        prp.remove_corrupted_img_path(args.out, args.img_extensions)
        if args.default_images != None:
            """Verify images in the path that is equals the images in the first
            argument, if found, delete it."""
            prp.remove_same_images(args.default_images, args.out,
                                    args.img_extensions)
        if not args.no_resize:
            #Resizes all images in a path to the max_size passed"""
            prp.resize_image_path(args.out, args.max_size, args.img_extensions)
        if not args.no_convert_gray:
            #Converts all images in a path to gray scale
            prp.convert_to_gray_path(args.out, args.img_extensions)
        if not args.no_std_names:
            #Standardizes all image names to starts with args.prefix.
            prp.rename_images_path(args.out, args.prefix, args.out_extension)
    except Exception as e:
        print(' '.join(["Error: ", str(e)]))

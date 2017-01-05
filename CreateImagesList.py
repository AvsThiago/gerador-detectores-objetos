#!/usr/bin/env python

"""
 This code provides a way recieve a path with images inside, and creates
 an output text file, where in each line there is a path to one image.

 @author  Thiago da Silva Alves
 @version 1.0, 29/12/16
"""

import os
import argparse

def parse_arguments():
    """Parses arguments choosen by the user."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        help='Path where the images are.',
                        required=True)
    parser.add_argument('--out',
                        help='Name of output file with a list of images path.',
                        required=True)
    parser.add_argument('--img-extensions',
                        help='Tuple of image extensions that will be accepted. '
                        '(".jpg", ".jpeg", ".png", ".bmp")',
                        default=(".jpg", ".jpeg", ".png", ".bmp"))
    return parser.parse_args()

if __name__ == '__main__':
    try:
        args = parse_arguments()
        #Lists the recived path and finds all images
        images = [fl for fl in os.listdir(args.path) \
                    if fl.lower().endswith(args.img_extensions)]
        #Writes the name of each image inside the created file
        with open(args.out, "w+") as f:
            for image_name in images:
                f.write(os.path.join(args.path, image_name) + '\n')
    except Exception as e:
        print(' '.join(["Error: ", str(e)]))

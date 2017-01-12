#!/usr/bin/env python

"""
 This code has the objective of using some classifiers, find objects in images
 and videos, independent of a scale or rotation.

 @author  Thiago da Silva Alves
 @version 1.0, 10/01/17
"""


import cv2 as cv
import numpy as np
import argparse
import os
from random import randint

#Class that will store the classifiers
class CascadeParameters:
    def __init__(self, cascade, level=0, minsize=(0,0), mn=3, sf=1.09,
                name="", color=(255,255,255)):
        self.cascade = cascade
        self.level = level
        self.min_size = minsize
        self.min_neighbors = mn
        self.scale_factor = sf
        self.name = name
        self.color = color


def parse_arguments():
    """Parses arguments choosen by the user."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-cascade',
                        help="File where there are many cascade files location "
                        "described with it's parameters.",
                        required=True)
    parser.add_argument('--angle', type=int,
                        help='Number divisible by 360, to rotate the image and '
                        'make the algorithm rotation invariant. <0>',
                        default=0)
    parser.add_argument('--num-camera', type=int,
                        help='Camera number. <0>',
                        default=0)
    parser.add_argument('--images-folder',
                        help='Folder where the images are.')
    parser.add_argument('--max-iou-ratio', type=float,
                        help="Use intersection over union to decide if a "
                        "detected objetc should be another one. If one "
                        "predicted rectangle intersects another in a value "
                        "above than this argument, the object with a bigger "
                        "level in --file-cascade will be chosen. <0>",
                        default=0.0)
    return parser.parse_args()

def rotate_bound(image, angle):
    #As seen in: https://goo.gl/aKzGiA
    """Rotates an image witout cut some part"""
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))

def fill_classifiers(args):
    """Takes the arguments and use it to fill a list of classifiers"""
    classifiers = []

    #If the argument's file exists, iterates over each line of it
    if os.path.isfile(args.file_cascade):
        for i in open(args.file_cascade).readlines()[1:]:

            #Split the arguments by comma anf fill the list of classifiers
            parameters = i.split(',')
            classifier = cv.CascadeClassifier(parameters[0])
            cascade = CascadeParameters(classifier, parameters[1],
                                        (int(parameters[2]), int(parameters[3])),
                                        parameters[4], parameters[5],
                                        parameters[6], (randint(0,255),
                                        randint(0,255), randint(0,255)))
            classifiers.append(cascade)
    return classifiers

def calc_IoU(boxA, boxB):
    """Calc of intersection over union"""
    #This code used some comcepts seen in: https://goo.gl/cQ3ykX
    # determine the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    return interArea / float(boxAArea + boxBArea - interArea)

def should_insert(current, list_detected, max_iou, level):
    """Only returns true if the object found has the biggest
    probability of been right"""
    to_remove = []

    #Iterates over all detected objects
    for i in list_detected:
        factor = calc_IoU(current, i[0])

        #If two objects intersects each other in a ratio bigger than max_iou
        #That who has the biggest level is chosen
        if factor >= max_iou:

            #If the object found has the lower level, it is discarded.
            if level < i[1]:
                return False
            else:
                to_remove.append(i)

    #Removes all objetcs that intersects the new object found
    [list_detected.remove(i) for i in to_remove]
    return True

def find_objects(img, angle, classifiers, max_iou):
    objects_found = []
    rotated_angle = 0
    img2 = img.copy()

    #Rotates the image until reach 360 degrees, it is increased by arg. angle.
    while rotated_angle < 360:

        #Iterates over all classifiers
        for i in classifiers:

            #Detects all objects in the current classifier
            rects = i.cascade.detectMultiScale(img2,minSize=i.min_size,
                                            scaleFactor=float(i.scale_factor),
                                            minNeighbors=int(i.min_neighbors))

            #Iterates over all found objects
            for (x, y, w, h) in rects:
                r_found = [x, y, w, h]
                #If the user assigned some value to max_iou, it's tested.
                if max_iou > 0.0:
                    if should_insert(r_found, objects_found, max_iou, i.level):
                        objects_found.append([r_found, i.level,
                                                i.name, i.color, rotated_angle])
                else:
                    #Or else, just inserts the new object in the list
                    objects_found.append([r_found, 0, i.name,
                                        i.color, rotated_angle])

        #If the angle is bigger than 0 rotates the image and searches again
        if angle > 0:
            img2 = img.copy()
            rotated_angle += angle
            img2 = rotate_bound(img2, rotated_angle)
        else:
            break
    return objects_found

def draw_rect(img, rect, text, color):
    """Draws the enclosing rectangle, and writes the object's name"""
    cv.rectangle(img, (rect[0], rect[1]),
                        (rect[0] + rect[2], rect[1] + rect[3]), color, 2)

    cv.putText(img, text, (rect[0], rect[1] + 15),  cv.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2, cv.LINE_AA)

def process_image(img, max_iou, angle, classifiers):
    """Takes one image and searches the objects on it"""

    #Converts the image to gray scale.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Gets the list of all objects found.
    objects_found = find_objects(gray, angle, classifiers, max_iou)

    #All angles where objects were found.
    angs = set([i[4] for i in objects_found])

    #Iterates over all angles.
    for ang in angs:

        #Gets all objects found in the current angle.
        obj_angle = filter(lambda l: l[4] == ang, objects_found)

        #Rotates the image to the current angle.
        img = rotate_bound(img, ang) if ang != 0 else img

        #Iterates over all objects found in the current angle.
        #And draws it
        for obj in obj_angle:
            draw_rect(img, obj[0], obj[2], obj[3])

        #Rotates the image to the original position
        img = rotate_bound(img, -ang) if ang != 0 else img

    return img

def process_by_camera(num_camera, max_iou, angle, classifiers):
    """Detects images on a webcam video"""
    cap = cv.VideoCapture(num_camera)

    #Iterates until the user press ESC
    while 1:
        #Get the current frame
        _, img = cap.read()

        #Find all objects and draws the enclosing rectangles on it
        img = process_image(img, max_iou, angle, classifiers)

        #Displays the image
        cv.imshow("Result", img)

        #Breakes if the user press ESC
        char = cv.waitKey(30) & 0xff
        if char == 27:
            break

def process_by_image(images_folder, max_iou, angle, classifiers):
    """Detects images on a list of images stored in a folder"""
    images = os.listdir(images_folder)

    #Iterates over all images
    for i in images:
        #Load the image
        img = cv.imread(os.path.join(images_folder, i))

        #Find all objects and draws the enclosing rectangles on it
        img = process_image(img, max_iou, angle, classifiers)

        #Displays the image
        cv.imshow("Result", img)

        #Breakes if the user press ESC
        char = cv.waitKey(0) & 0xff
        if char == 27:
            break


if __name__ == '__main__':
    #Parses teh user's arguments
    args = parse_arguments()

    #Creates a list of CascadeParameters object
    clsf = fill_classifiers(args)

    #If the user user wants to precess a list of images in a folder,
    #so do it. Or else starts the process by the webcam.
    if args.images_folder == None:
        process_by_camera(args.num_camera, args.max_iou_ratio,
                            args.angle, clsf)
    else:
        process_by_image(args.images_folder, args.max_iou_ratio,
                        args.angle, clsf)

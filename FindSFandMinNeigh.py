#!/usr/bin/env python

"""
 This code has the objective of find the best values to the OpenCV method
 detectMultiScale(), the values are scaleFactor and minNeighbors. This software
 was ceated because is hard to choose the best combination of this two values
 once there are many options, so this software tests many combinations and
 finds the best one.

 @author  Thiago da Silva Alves
 @version 1.0, 05/01/17
"""

import argparse
import cv2 as cv
import os

def parse_arguments():
    """Parses arguments choosen by the user."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cascade-file',
                        help='XML file with cascade trained.',
                        required=True)
    parser.add_argument('--max-obg-img', type=int,
                        help='Maximum number of objects in an image. <10>',
                        required=False, default=10)
    parser.add_argument('--iwi', type=int,
                        help='Number of iterations without improvement '
                        'until break. <5>',
                        required=False, default=5)
    parser.add_argument('--step', type=int,
                        help='Steps between SFs and MinNeighbors. <1>',
                        required=False, default=1)
    parser.add_argument('--list-of-images',
                        help='File that describes where are the images and the '
                        'bounding boxes.')
    parser.add_argument('--visualize',
                        help='Visualize the process.',
                        action='store_true')
    parser.add_argument('--resize-predict',
                        help='Resizes each founded rectangle that intercepts '
                        'the ground-truth to the same size of ground truth.',
                        action='store_true')

    return parser.parse_args()

def intersection(rec1, rec2):
    """Detects if has an intersection between two rectangles"""
    """
    rec1.X1 < rec2.X2:	needs to be true
    rec1.X2 > rec2.X1:	needs to be true
    rec1.Y1 < rec2.Y2:	needs to be true
    rec1.Y2 > rec2.Y1:	needs to be true
    Intersect:	true
    """
    return rec1[0] <= rec2[0] + rec2[2] and \
            rec1[0] + rec1[2] >= rec2[0] and \
            rec1[1] <= rec2[1] + rec2[3] and \
            rec1[1] + rec1[3] >= rec2[1]

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

def recalc_rect(old_rect, orig_w_h):
    """Resizes the predicted retangle to the size of the ground truth."""
    old_rect[0] -= (orig_w_h[0] - old_rect[2]) // 2
    old_rect[2] += (orig_w_h[0] - old_rect[2])
    old_rect[1] -= (orig_w_h[1] - old_rect[3]) // 2 # // returns only int part
    old_rect[3] += (orig_w_h[1] - old_rect[3])
    return old_rect

def draw_recalculed_rect(img, nr):
    """Draws a predicted rectangle with the size of the ground-truth"""
    cv.rectangle(img, (nr[0], nr[1]),
                (nr[0] + nr[2], nr[1] + nr[3]), (255, 0, 0), 2)

def resize_rect(args, img, old_rect, rect):
    """If the user wants to resize the rectangle predicted if the user wants"""
    if args.resize_predict:
        nr = recalc_rect(old_rect, rect[2:])
        if args.visualize:
            draw_recalculed_rect(img, nr)
        return rect, nr
    else:
        return rect, old_rect

def tot_IoU(args, img, ret, sf, mn):
    """Calculates the sum of IoU to each predicted rectangle"""
    #Loads the classifier and make the prediction of where the object is.
    classifier = cv.CascadeClassifier(args.cascade_file)
    rects = classifier.detectMultiScale(img, scaleFactor=sf, minNeighbors=mn,
                                        minSize=(13, 20))
    #Sum of accuracy of prediction to each found rectangle.
    sum_accuracy = 0
    if args.max_obg_img >= len(rects):
        for (x, y, w, h) in rects:
            #Verify if this found rectangle intersects the ground truth.
            if intersection(ret, [x, y, w, h]):
                #Resizes the found rect to the ground-truth, if the user wants.
                boxA, boxB = resize_rect(args, img, [x, y, w, h], ret)
                #Sum the calculate IoU with the sum_accuracy.
                sum_accuracy += calc_IoU(boxA, boxB)
                #Draws the found rectangle in the final image
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #Returns the sum_accuracy divided by the number of rects predicted.
    return sum_accuracy / len(rects) if len(rects) > 0 else 0

def process_bb(args):
    #Reads all lines in the files that describes the images' location
    imgs = [i.split()  for  i in open(args.list_of_images).readlines()]
    #Loads all images in memory
    imgs_loaded = [cv.imread(os.path.join("IoU", i[0])) for i in imgs]
    print("imagens carregadas")

    #[Best_sum, Best_Scale_Factor, Best_Min_Neighbors]
    best_values = [0,0,0]
    #This list manages the improvement of Scale Factor
    sf_converged = []
    for sf in range(101, 190, 1):
        #Creates a list that stores the last args.iwi values, if the best
        #sum_tot_iou doesn't improves in args.iwi iterations the loop stops.
        sf_converged.insert(0, best_values[0])
        if len(sf_converged) == args.iwi:
            if sf_converged.count(sf_converged[0]) == args.iwi:
                break
            else:
                sf_converged.pop()
        #This list manages the improvement of minNeighbors
        mn_converged = []
        for mn in range(1, 30):
            sum_tot_iou = 0
            #Creates a list that stores the last args.iwi values, if the best
            #sum_tot_iou doesn't improves in args.iwi iterations the loop stops.
            mn_converged.insert(0, sum_tot_iou)
            if len(mn_converged) == args.iwi:
                if mn_converged.count(mn_converged[0]) == args.iwi:
                    break
                else:
                    mn_converged.pop()
            #Iterates over all images
            for i, img in enumerate(imgs_loaded[:100]):
                x = int(imgs[i][2])
                y = int(imgs[i][3])
                w = int(imgs[i][4])
                h = int(imgs[i][5])

                sum_tot_iou += tot_IoU(args, img, [x, y, w, h], sf/100, mn)

                #Shows the processed image if the user wants
                if args.visualize:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv.imshow("Original",img)
                    cv.waitKey(0)
            #Verify if the sum_tot_iou was improved
            if sum_tot_iou > best_values[0]:
                print(sum_tot_iou, sf/100, mn)
                best_values = [sum_tot_iou, sf/100, mn]

if __name__ == '__main__':
    args = parse_arguments()
    """Processes the bounding boxes"""
    process_bb(args)

"""
Just some tests

thiago@NoThiago:/tenacious/Projetos/Python/rmc$ python \
FindSFandMinNeigh.py --cascade-file ./bud.xml --list-of-images IoU/images.lst

imagens carregadas
11.7608993869 1.01 1
14.9421513724 1.01 2
17.2791647037 1.01 3
19.5369352573 1.01 4
21.3052754519 1.02 2
25.9422066108 1.02 3
28.4374411223 1.02 4
29.9473355393 1.03 4
32.6103687937 1.04 4
33.535151138 1.05 4
34.1895558244 1.06 4
34.6404146868 1.07 4

thiago@NoThiago:/tenacious/Projetos/Python/rmc$ python \
FindSFandMinNeigh.py --cascade-file ./bud.xml --list-of-images IoU/images.lst \
 --resize-predict

imagens carregadas
19.2822310736 1.01 1
23.204152833 1.01 2
26.7174457047 1.01 3
30.359548311 1.01 4
32.7544552385 1.02 2
39.205029059 1.02 3
43.4669310955 1.02 4
45.5354363362 1.03 4
46.6643763919 1.04 3
50.2056377186 1.04 4
50.6796756712 1.05 4
52.349141331 1.06 4
52.6294877208 1.07 4
avsthiago@NoThiago:/tenacious/Projetos/Python/rmc$
"""

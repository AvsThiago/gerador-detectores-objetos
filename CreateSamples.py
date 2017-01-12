#!/usr/bin/env python

"""
 This code has the objective of using a set of positive samples create
 a set of samples to train haar cascade. The main difference between this code
 and the opencv_createsamples is that with this is possible to use many positive
 images to create the vec file.

 @author  Thiago da Silva Alves
 @version 1.0, 29/12/16
"""

import os
import subprocess
import argparse
from  shutil import rmtree
from  shutil import move

def parse_arguments():
    """Parses arguments choosen by the user."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--vec',
                        help='Name of the output file containing the positive'
                        ' samples for training.')
    parser.add_argument('--pos',
                        help='Positive samples file name.', required=True)
    parser.add_argument('--bg',
                        help='Background description file; contains a list of '
                        'images which are used as a background for randomly '
                        'distorted versions of the object.', required=True)
    parser.add_argument('--num', type=int,
                        help='Number of positive samples to generate.',
                        required=True)
    parser.add_argument('--bgcolor',
                        help='Background color (currently grayscale images are '
                        'assumed); the background color denotes the transparent'
                        ' color. Since there might be artifacts, the'
                        ' amount of color tolerance can be specified by '
                        '-bgthresh. All pixels withing bgcolor-bgthresh and '
                        'bgcolor+bgthresh range are interpreted as '
                        'transparent. <0>')
    parser.add_argument('--bgthresh',
                        help='Background color threshold.<80>')
    parser.add_argument('--inv',
                        help='If specified, colors will be inverted.',
                        action='store_true')
    parser.add_argument('--randinv',
                        help='If specified, colors will be inverted.',
                        action='store_true')
    parser.add_argument('--out-img-folder',
                        help='Save all created images in a folder.')
    parser.add_argument('--maxidev',
                        help='Maximal intensity deviation of pixels in '
                        'foreground samples. <40>')
    parser.add_argument('--maxzangle', type=float,
                        help='Maximum rotation in angle Z must be given in '
                        'radians. <0.50000>')
    parser.add_argument('--maxxangle', type=float,
                        help='Maximum rotation in angle X must be given in '
                        'radians. <1.10000>')
    parser.add_argument('--maxyangle', type=float,
                        help='Maximum rotation in angle Y must be given in '
                        'radians. <1.10000>')
    parser.add_argument('--height',
                        help='Height (in pixels) of the output samples. <24>',
                        default='24')
    parser.add_argument('--width',
                        help='Width (in pixels) of the output samples. <24>',
                        default='24')

    return parser.parse_args()

def generate_line(args, num_samples, img_pos, path_info):
    """Returns a line with one OpenCV command that create samples"""
    line = " ".join(["opencv_createsamples -img", img_pos, "-bg", args.bg,
                    "-num", str(num_samples), "-info",
                    os.path.join(path_info, "images.lst")])
    #Creates the command according with arguments given by the user
    if args.bgcolor != None:
        line = " ".join([line, "-bgcolor", args.bgcolor])
    if args.bgthresh != None:
        line = " ".join([line, "-bgthresh", args.bgthresh])
    if args.inv:
        line = " ".join([line, "-inv"])
    if args.randinv:
        line = " ".join([line, "-randinv"])
    if args.maxidev != None:
        line = " ".join([line, "-maxidev", args.maxidev])
    if args.maxzangle != None:
        line = " ".join([line, "-maxzangle", str(args.maxzangle)])
    if args.maxxangle != None:
        line = " ".join([line, "-maxxangle", str(args.maxxangle)])
    if args.maxyangle != None:
        line = " ".join([line, "-maxyangle", str(args.maxyangle)])
    if args.height != None:
        line = " ".join([line, "-h", args.height])
    if args.width != None:
        line = " ".join([line, "-w", args.width])

    return line

def generate_commands_to_opencv(args):
    """Creates an OpenCV command according to how many positive samples exists"""
    #A list of commands that will be returned
    lines = []
    #Reads all lines of positive file and stores in a variable
    pos_files = open(args.pos, "r+").readlines()
    #Divides the number of samples that the user wants by number of pos. files.
    num_samples_x_pos = int(args.num / len(pos_files))
    #Same of above line but adding the remainder
    num_samples_x_pos_last = num_samples_x_pos + (args.num % len(pos_files))
    #File where the generated images and description files will be stored
    temporary_path = "./.created_samples_temp"
    #Creates the temporaty path
    if os.path.exists(temporary_path):
        rmtree(os.path.abspath("./.created_samples_temp"))
    os.mkdir(temporary_path)
    #Total of positive images
    num_pos = len(pos_files) - 1
    #Iterates over each positive image
    for i, pos in enumerate(pos_files):
        #Creates a file with a unique name (i is the number of the iteration)
        os.mkdir(os.path.join(temporary_path, str(i)))
        path = os.path.join(temporary_path, str(i))
        #If is the last line the created command will differ in samples number

        if i < num_pos:
            lines.append(generate_line(args, num_samples_x_pos,
                        pos.strip('\n'), path))
        else:
            lines.append(generate_line(args, num_samples_x_pos_last,
                        pos.strip('\n'), path))

    return lines

def execute_commands(commands):
    """Executes all command lines passed by argument, one by one"""
    for new_process in commands:
        command = new_process.split()
        with subprocess.Popen(command) as proc:
            proc.wait()

def move_images_and_list(path, final_path):
    """After generate samples to each positive image, these samples are in
    different folders, so this method join all images ans description
    lists in just one folder"""
    #Lists all created folders
    directories = os.listdir(path)
    #Array that stores the path to each image
    lists = []
    #This variable will be used to give a unique name to each image
    tot_images = 0
    #Creates the path where will be stored all files
    if not os.path.exists(final_path):
        os.mkdir(final_path)
    #Iterates over each folder
    for ph in directories:
        #Iterates over each line of the generated file images.lst
        for img in open(os.path.join(path, ph, "images.lst")).readlines():
            """Images are stored with a name, how many objects have and
            where it is, like this '01_0252_0067_0139_0222.jpg 1 252 67 139 222'
            so these five lines under changes the first part before '_', because
            in some cases, the command opencv_createsamples creates a same name
            to different positive images, this ensures a different name to each
            image"""
            split_space = img.split()
            split_underscore = split_space[0].split("_")
            split_underscore[0] = str(tot_images)
            join_underscore = "_".join(split_underscore)
            join_space = " ".join([join_underscore, *split_space[1:]])
            #Appends the new image's name to the list
            lists.append(join_space)
            #Moves each image in the folder to the final path, with a new name
            move(os.path.join(path, ph, split_space[0]),
                                os.path.join(final_path, join_space.split()[0]))
            tot_images += 1
    #Writes a file withe the name of all images in the folder
    with open(os.path.join(final_path, "images.lst"), "w+") as f:
        for i in lists:
            f.write("".join([i, '\n']))
    #Removes the temporary path
    rmtree(os.path.abspath(path))
    #Name of the created file
    return "images.lst"

def name_final_path(out_img_folder):
    """If the user doesn't want to see the generated samples,
        so stores them temporarily in a folder. Will be removed after"""
    if out_img_folder == None:
        return "./.out_hidden_images"
    else:
        return out_img_folder

def generate_vec_file(args, path_list, file_name):
    """Generatas a vec file, used to train the haar"""
    if args.vec != None:
        #If the user wants to create a vec file, so calls opencv to create it.
        command = " ".join(["opencv_createsamples -vec", args.vec,
                            "-info", os.path.join(path_list, file_name),
                            "-num", str(args.num), "-h", str(args.height),
                            "-w", str(args.width)])
        execute_commands([command])
    if args.out_img_folder == None:
        """If the user don't want to save created samples,
        so deletes the entire folder"""
        rmtree(os.path.abspath(path_list))

if __name__ == '__main__':
    try:
        args = parse_arguments()
        #Generates opencv commands that will create samples
        lines_to_process = generate_commands_to_opencv(args)
        execute_commands(lines_to_process)
        #Path where the sample's list is
        list_samples = name_final_path(args.out_img_folder)
        #Move the files created to each positive sample to just one place
        lst = move_images_and_list("./.created_samples_temp", list_samples)
        generate_vec_file(args, list_samples, lst)
    except Exception as e:
        print(' '.join(["Error: ", str(e)]))

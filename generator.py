"""
Module Docstring
"""

__author__ = "Yolanne Lee"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse
from create import makeDataset
import numpy as np
import time
from curvature_flow import computeFlow as flow

def main(args):
    """ Main generator """
    print("<<Data generator>>\nCurrently supports curvature flow post-processing.")
    print(args)

    # num_images = 50
    # size = 250
    # frames = 5
    # dt = 1e-3
    ops = {'dx':100/args.size, 'Tmax':20, 'frames':args.frames, 'version': args.curv_version}

    if args.time:
        start = time.time()
    dataset = makeDataset(args.size,args.num_images,args.min,args.max,args.colour)

    if args.time:
        end = time.time()
        print("Time: ", end-start)
    print(args.num_images," images created.")

    if args.transform == 'curvature':
        processed = np.zeros((args.num_images,args.size,args.size),np.float32)
        frames = np.zeros((args.num_images,args.frames,args.size,args.size), np.float32)
        if args.time:
            start = time.time()
        for i in range(dataset.shape[0]):
            processed[i,...], _, frames[i,...]= flow(dataset[i,...], dt=args.dt, **ops)
        if args.time:
            end = time.time()
            print("Time: ", end-start)
        print(args.num_images," images processed using fancy flat curvature flow.")

    else:
        frames = dataset
    # for i in range(frames.shape[0]):
    #     for j in range(frames.shape[1]):
    #         out = frames[i,j,...]
    #         out -= np.min(out)
    #         out /= np.max(out)
    #         out = np.min(dataset[i]) + (np.max(dataset[i])-np.min(dataset[i]))*out
    #         frames[i,j,...] = out
    np.save(args.filename, frames)



if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--size", action="store", type=int, default=256, dest="size", help="Desired image size; assumes x_dim=y_dim for square images")
    parser.add_argument("-n", "--num", action="store", type=int, default=1, dest="num_images", help="Desired number of images")
    parser.add_argument("-f", "--frames", action="store", type=int, default=2, dest="frames", )
    parser.add_argument("-c", "--colour", action="store_true", default=False, dest="colour", help="Colour images - if not set, grayscale")
    parser.add_argument("-min", action="store", type=int, default=5, dest="min", help="Minimum # objects in image")
    parser.add_argument("-max", action="store", type=int, default=12, dest="max", help="Maximum # objects in image")
    parser.add_argument("-o", "--output", action="store", type=str, default="temp", dest="filename", help="Output filename to save as")
    parser.add_argument("-t", "--transform", action="store", type=str, default="curvature", dest="transform", help="Optional post-processing: 'curvature', 'anisodiff (NOT implemented)'")
    parser.add_argument("-dt", action="store", type=float, default=1e-3, dest="dt", help="Time step for post-processing; accepts notation like 1e-3")
    parser.add_argument("-maxt", action="store", default=20, type=int, dest="maxt", help="Max time for post-processing")
    parser.add_argument("--time", action="store_true", default=False, dest="time", help="Time each step")
    parser.add_argument("--stepversion", action="store", type=str, default="fancy", dest="curv_version", help="Version of curvature flow")

    # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    # parser.add_argument(
        # "-v",
        # "--verbose",
        # action="count",
        # default=0,
        # help="Verbosity (-v, -vv, etc)")

    # Specify output of "--version"

    args = parser.parse_args()
    main(args)
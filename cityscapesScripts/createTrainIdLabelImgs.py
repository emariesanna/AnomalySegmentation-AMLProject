#!/usr/bin/python
#
# Converts the polygonal annotations of the Cityscapes dataset
# to images, where pixel values encode ground truth classes.
#
# The Cityscapes downloads already include such images
#   a) *color.png             : the class is encoded by its color
#   b) *labelIds.png          : the class is encoded by its ID
#   c) *instanceIds.png       : the class and the instance are encoded by an instance ID
# 
# With this tool, you can generate option
#   d) *labelTrainIds.png     : the class is encoded by its training ID
# This encoding might come handy for training purposes. You can use
# the file labels.py to define the training IDs that suit your needs.
# Note however, that once you submit or evaluate results, the regular
# IDs are needed.
#
# Uses the converter tool in 'json2labelImg.py'
# Uses the mapping defined in 'labels.py'

from __future__ import print_function, absolute_import, division

DATASET_NAME = 'Cityscapes' # Cityscapes, LostAndFound

# python imports

import os, glob, sys

from json2labelImg import json2labelImg

# function copied from csHelpers.py
def printError(message):
    """Print an error message and quit"""
    print('ERROR: ' + str(message))
    sys.exit(-1)

# The main method
def main():
    # Where to look for Cityscapes
    # if 'CITYSCAPES_DATASET' in os.environ:
    #     cityscapesPath = os.environ['CITYSCAPES_DATASET']
    # else:
    #     cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

    if DATASET_NAME == 'Cityscapes':
        dataset_path = './Dataset/Cityscapes'
    elif DATASET_NAME == 'LostAndFound':
        dataset_path = './Dataset/LostAndFound'

    # how to search for all ground truth
    searchFine   = os.path.join( dataset_path , "gtFine"   , "*" , "*" , "*_gt*_polygons.json" )
    searchCoarse = os.path.join( dataset_path , "gtCoarse" , "*" , "*" , "*_gt*_polygons.json" )

    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()
    filesCoarse = glob.glob( searchCoarse )
    filesCoarse.sort()

    # concatenate fine and coarse
    files = filesFine + filesCoarse
    # files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything
    if not files:
        printError( "Did not find any files. Please consult the README." )

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        # create the output filename
        dst = f.replace( "_polygons.json" , "_labelTrainIds.png" )

        # do the conversion
        try:
            json2labelImg( f , dst , "trainIds", DATASET_NAME)
        except:
            print("Failed to convert: {}".format(f))
            raise

        # status
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()


# call the main
if __name__ == "__main__":
    main()

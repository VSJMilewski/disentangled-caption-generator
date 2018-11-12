import argparse
import random
from os import listdir
from os.path import isfile, join

import numpy as np


def print_files(filename, files):
    """
    Print a list of names to a file
    :param filename: The name of the file to write to
    :param files: A list of filenames
    :return: Nothing
    """
    with open(filename, 'w') as f:
        for file in files:
            print(file, file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_dir", type=str, required=True, help='File containing reference sentences.')
    parser.add_argument("-o", "--output_file", type=str, required=True, help='Filename and path of the output files')
    parser.add_argument("-d", "--dev_size", type=int, required=True, help='Size of development set')
    parser.add_argument("-t", "--test_size", type=int, required=True, help='Size of test set')
    args = parser.parse_args()

    # find all the image files in the data directory
    files = [f for f in listdir(args.data_dir) if isfile(join(args.data_dir, f))]
    # setup the set sizes. If given as fractions, compute the real size
    dev_size = args.dev_size
    if dev_size <= 1:
        dev_size = np.ceil(len(files) * dev_size)
    test_size = args.test_size
    if dev_size <= 1:
        test_size = np.ceil(len(files) * test_size)
    train_size = len(files) - dev_size - test_size

    # make sure that every run the same division is created by first sorting, followed by shuffling with a fixed seed
    files.sort()
    random.seed(42)
    random.shuffle(files)

    # create three lists of file names for the different sets
    train_files = files[:train_size]
    dev_files = files[train_size:train_size + dev_size]
    test_files = files[train_size + dev_size:]

    # print the created sets to files
    print_files("{}.trainImages.txt".format(args.output_file), train_files)
    print_files("{}.devImages.txt".format(args.output_file), dev_files)
    print_files("{}.testImages.txt".format(args.output_file), test_files)


if __name__ == "__main__":
    main()

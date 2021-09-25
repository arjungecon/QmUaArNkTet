import os
from os import listdir
from os.path import isfile, join
import _pickle as cPickle
import bz2


def print_pickles():

    """
        Prints a list of pickles currently stored in the system.
    :return: List of pickles.
    """

    pkl_list = []

    os.chdir("../Pickles")

    file_list = [f for f in listdir(os.path.abspath(os.getcwd())) if isfile(join(os.path.abspath(os.getcwd()), f))]

    for f in file_list:

        if f.endswith(".pkl") or f.endswith(".pbz2"):

            pkl_list.append(f)

    os.chdir('../../')

    return pkl_list


def open_compressed_pickle(pkl_file):

    """
        This function converts a .pbz2 file into a Pandas dataframe.
    :param pkl_file: Input .pbz2 to be converted into a dataframe.
    :return: Pandas dataframe
    """

    os.chdir("./Pickles")
    data = bz2.BZ2File(pkl_file + '.pbz2', 'rb')
    data = cPickle.load(data)

    os.chdir('../')

    return data


def delete_pickle(pkl_file):

    """
        Deletes a pickle from the system.
    :param pkl_file:  Input .pbz2 or .pkl to be deleted.
    :return:
    """

    os.chdir("./Pickles")

    os.remove(pkl_file)

    os.chdir('../')

    print(pkl_file + ' has been deleted.')




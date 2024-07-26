import numpy as np
from scipy.signal import convolve2d
import os
import math

import os.path
import numpy as np
import matplotlib.image as mpimg


def load_array_from_folder(folder_path: str) -> np.ndarray:
    """Load grayscale OCT volume as numpy array from a given string path

    Args:
        folder_path (str): folder path as a string

    Returns:
        np.ndarray: OCT cbe as numpy array, if a loading error occurs, returns an empty array
    """
    folder_path += '/'
    files = [f for f in os.listdir(folder_path) if os.path.isfile(
        os.path.join(folder_path, f)) if f[-4:] == '.jpg' or f[-4:] == '.png']
   
    # Longest file name
    max_len = len(max(files, key=len))
    # Recompute files number as three digits
    files_extended = ['0' * (max_len - len(f)) + f for f in files]
    # Sort files by number
    files = [x for _, x in sorted(zip(files_extended, files))]

    try:
        image_shape = mpimg.imread(folder_path + files[0]).shape
    except BaseException:
        return np.empty((1, 1))
    
    volume = np.empty((len(files), image_shape[0], image_shape[1]))
    for index, file in enumerate(files):
      
        if (os.path.isfile(folder_path + file)):
            
            try:
                # Tries to load each image
                volume[index] = (mpimg.imread(folder_path + file))    
                
            except BaseException:
                # Return an empty list of images if one scan could not be
                # loaded
                return np.empty((1, 1))
        else:
        
            return np.empty((1, 1))
 
    return volume.astype('uint8')

def load_images_from_folder(folder_path: str):
    """Load multiple images in an array from a folder in a given path. Images need to be ordered by number

    Args:
        path (string): path to folder containing JPEGs forming the OCT scan

    Returns:
        list[array]: array of loaded B-scans images contained in folder
        return empty list of one or more image could not be loaded
    """
    folder_path += '/'
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path,f)) if f[-4:] == '.jpg' or f[-4:] == '.png']

    # Longest file name
    max_len = len(max(files, key=len))
    # Recompute files number as three digits
    files_extended = ['0' * (max_len - len(f)) + f for f in files]
    # Sort files by number
    files = [x for _, x in sorted(zip(files_extended, files))]

    imgs = list()
    for f in files:
        try:
            # Tries to load each image
            imgs.append(mpimg.imread(folder_path + f))
        except BaseException:
            # Return an empty list of images if one scan could not be loaded
            return list()

    return imgs


def sigmoid(x : float, l = 1.0):
    """ Computes results of sigmoid, with parameter l as lambda for controling slope

    Args:
        x (float): input value
        l (float, optional): Lambda, parameter that controls slope. Defaults to 1.0.

    Returns:
        float : sigmoid of input
    """
    return 1 / (1 + np.exp(-(l*x)))

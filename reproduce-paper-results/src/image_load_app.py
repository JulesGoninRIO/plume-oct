import os
import os.path
import numpy as np
import matplotlib.image as mpimg
import zipfile

# TODO : clean the code and document (bc old code not everything is useful anymore)


def load_images_from_folder(folder_path: str):
    """
    Load multiple images in an array from a folder in a given path. Images need to be ordered by number

    Args:
        path (string): path to folder containing JPEGs forming the OCT scan

    Returns:
        list[array]: array of loaded B-scans images contained in folder
        return empty list of one or more image could not be loaded
    """
    # folder_path += '\\'
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        if f[-4:] == ".jpg" or f[-4:] == ".png"
    ]

    # Longest file name
    max_len = len(max(files, key=len))
    # Recompute files number as three digits
    files_extended = ["0" * (max_len - len(f)) + f for f in files]
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


def load_array_from_folder(folder_path: str) -> np.ndarray:
    """Load grayscale OCT volume as numpy array from a given string path

    Args:
        folder_path (str): folder path as a string

    Returns:
        np.ndarray: OCT cbe as numpy array, if a loading error occurs, returns an empty array
    """

    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        if f[-4:] == ".jpg" or f[-4:] == ".png"
    ]

    # Longest file name
    max_len = len(max(files, key=len))
    # Recompute files number as three digits
    files_extended = ["0" * (max_len - len(f)) + f for f in files]
    # Sort files by number
    files = [x for _, x in sorted(zip(files_extended, files))]

    try:
        image_shape = mpimg.imread(folder_path + files[0]).shape
    except BaseException:
        return np.empty((1, 1))

    volume = np.empty((len(files), image_shape[0], image_shape[1]))
    for index, file in enumerate(files):
        if os.path.isfile(folder_path + file):
            try:
                # Tries to load each image
                volume[index] = mpimg.imread(folder_path + file)
            except BaseException:
                # Return an empty list of images if one scan could not be
                # loaded
                return np.empty((1, 1))
        else:
            return np.empty((1, 1))

    return volume.astype("uint8")


def load_images_from_zip(zip_path):
    """
    Load images into an array from a zip file

    Args:
        zip_path (string): path to zip file

    Returns:
        img (list): list of B-scans loaded from zip folder
    """
    # Unzip file into variable
    imgzip = zipfile.ZipFile(zip_path)
    # Get info list // see Python API for details
    infolist = imgzip.infolist()
    # Filter files only in JPG or PNG format
    infolist = [
        f for f in infolist if f.filename[-4:] == ".jpg" or f.filename[-4:] == ".png"
    ]
    imgs = []
    for f in infolist:
        imgs.append(mpimg.imread(imgzip.open(f)))
    return imgs


def load_array_from_zip(zip_path):
    """
    Load images into an array from a zip file

    Args:
        zip_path (string): path to zip file

    Returns:
        array: array of B-scans loaded from zip folder
    """
    # Unzip file into variable
    imgzip = zipfile.ZipFile(zip_path)
    # Get info list // see Python API for details
    infolist = imgzip.infolist()
    # Filter files only in JPG or PNG format
    infolist = [
        f for f in infolist if f.filename[-4:] == ".jpg" or f.filename[-4:] == ".png"
    ]

    if os.path.isfile(zip_path):
        volume = []
        for index, file in enumerate(infolist):
            try:
                # Tries to load each image
                volume.append(mpimg.imread(imgzip.open(file)))
            except BaseException:
                # Return an empty list of images if one scan could not be
                # loaded
                return np.empty((1, 1))
    else:
        return np.empty((1, 1))

    return np.array(volume)


def load_frames_from_folder(folder_path, frames) -> list:
    """
    Load specific images from an OCT scan data.

    Args:
        folder_path (str): path to oct scan data.
        frames (list[int]): list of frames to load

    Returns:
        list : list of loaded frames as numpy array
    assert len(frames) != 0, " No frames provided"
    """
    folder_path += "/"
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    # Longest file name
    max_len = len(max(files, key=len))
    # Recompute files number as three digits
    files_extended = ["0" * (max_len - len(f)) + f for f in files]
    # Sort files by number
    files = [x for _, x in sorted(zip(files_extended, files))]

    img = list()

    try:
        # Tries to load image
        for frame in frames:
            # Tries to fill list with all valid frames
            if frame >= 0 and frame < len(files):
                img.append(mpimg.imread(folder_path + files[frame]))
    except BaseException:
        # Return empty list
        return list()
    return img

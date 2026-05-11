import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os 
import matplotlib.image as mpimg
import cv2
from configparser import ConfigParser
from argparse import ArgumentParser
import json

def load_images_from_folder(folder_path: str) -> list:
    """
    Load multiple images in an array from a folder in a given path.

    Args:
        folder_path (string): path to folder containing JPEGs forming the OCT scan

    Returns:
        list[array]: array of loaded B-scans images contained in folder
        return empty list if one or more images could not be loaded
    """

    folder_path += '/'
    files = [
        f for f in os.listdir(folder_path) if os.path.isfile(
            os.path.join(
                folder_path,
                f)) if f[-4:] == '.jpg' or f[-4:] == '.png']

    max_len = len(max(files, key=len))  # Longest file name
    files_extended = ['0' * (max_len - len(f)) + f for f in files]  # Recompute files number as three digits
    files = [x for _, x in sorted(zip(files_extended, files))]  # Sort files by number
    imgs = list()
    for f in files:
        try:
            imgs.append(mpimg.imread(folder_path + f))  # Tries to load each image
        except BaseException:
            return list()  # Return an empty list of images if one scan could not be loaded

    return imgs

def get_sub_misalignment(header, misalignment):
    """
    Get subset of misalignment data corresponding to the provided header.

    Args:
        header (pd.DataFrame): header data
        misalignment (pd.DataFrame): misalignment data

    Returns:
        pd.DataFrame: Subset of misalignment data
    """
    sub_misalignment = misalignment.iloc[header.index]
    sub_misalignment = sub_misalignment.replace(-9999.0, np.nan)
    sub_misalignment.dropna(axis=1, inplace=True)
    sub_misalignment.reset_index(drop=True, inplace=True)
    return sub_misalignment

def string_to_string_list(s):
    """Convert a string representation of a list to a Python list."""
    s = s.strip('[]')
    split_string = s.split()
    result = '[' + ', '.join(split_string) + ']'
    return result

def string_to_float_list(s):
    """Convert a string representation of a list of floats to a Python list."""
    s = s[1:-1]
    s = s.replace(',', ' ')
    nbs = s.split(' ')
    return [int(nb) for nb in nbs if nb]

def get_annot_scans_name(paths_images):
    """
    Get names of annotated scans from the provided image paths.

    Args:
        paths_images (str): Path to the folder containing images

    Returns:
        list: List of scan names
    """
    filenames = next(os.walk(paths_images), (None, None, []))[1]
    names_scans = []

    for filemames in filenames:
        names_scans.append((filemames.split("_")[2].split("D")[1]))

    return names_scans

def joint_header_misalignment(headers, misalignment, paths_images):
    """
    Join header and misalignment data based on the annotated scan names.

    Args:
        headers (pd.DataFrame): Header data
        misalignment (pd.DataFrame): misalignment data
        paths_images (str): Path to the folder containing images

    Returns:
        tuple: Tuple containing modified headers and misalignment data
    """
    name_scan = get_annot_scans_name(paths_images)

    # Get info only for the annotated scan (match scan name)
    headers_mod = headers[[x in name_scan for x in headers.dataset_uiid]]

    # Match and modify the misalignment signal to the new header
    misalignment = get_sub_misalignment(headers_mod, misalignment)

    return headers_mod, misalignment

def sub(row, a=2):
    """Subtract a constant value 'a' from each element in the row."""
    return [x - a for x in row]

def first_scan(row):
    """Extract unique values from the row, considering -1 as 0."""
    return np.unique([0 if x == -1 else x for x in row])

def pre_pro_annot(annotation):
    """
    Preprocess annotation data.

    Args:
        annotation (pd.DataFrame): Annotation data

    Returns:
        pd.DataFrame: Preprocessed annotation data
    """
    annotation.frames = annotation.frames.apply(string_to_float_list)
    annotation.frames = annotation.frames.apply(sub)
    annotation.frames = annotation.frames.apply(first_scan)
    annotation = annotation.rename(columns={'uuid': 'dataset_uiid'})
    return annotation

def fundus_along_misalignment(fundus, compound, label_PLUME, x_max, peak_list=[], label_annot=None,
                     title=None, color_PLUME="slategrey", show_curve=True, color_annot="seagreen"):
    """
    Plot fundus reconstruction alongside multiple user-specified score curves.

    Args:
        fundus (np.ndarray): Fundus data.
        compound (list): Compound measure to plot alongside fundus.
        label_PLUME (str): Label to identify PLUME score.
        x_max (float): Maximum of x-axis for plotting compound.
        peak_list (list, optional): List of indices of peaks to plot hlines. Defaults to empty list.
        label_annot (str, optional): Label to identify annotations. Defaults to None.
        title (str, optional): Title for the plot. Defaults to None.
        color_PLUME (str, optional): Color for PLUME score. Defaults to "slategrey".
        show_curve (bool, optional): Whether to show the score curve. Defaults to True.
        color_annot (str, optional): Color for annotations. Defaults to "seagreen".
    """
    assert len(fundus.shape) == 2, "Provided array has the wrong number of dimensions"

    height, width = fundus.shape[0], fundus.shape[1] * 2
    figure, axis = plt.subplots(1, 1, figsize=(15, 15))

    size = compound[compound != -9999.0].shape[0] + 1
    axis.imshow(fundus, cmap='gray')

    # Ticks for grid & y-labels
    if size == 128:
        yticks = 2 + 1
    if size == 256:
        yticks = 4 + 1
    if size == 320:
        yticks = 8 + 1
    else:
        yticks = 2 + 1

    # Horizontal grid
    axis.vlines(width / 2 + 5, ymin=0, ymax=height, color='k')
    x_max=x_max-x_max%5
    xticks = list(range(0,int(x_max)+5,5))
    for i, x in enumerate(np.linspace(width / 2 + 5, width, len(xticks))):
        axis.vlines(x, ymin=0, ymax=height, color='k', alpha=0.3)
        # x tick
        xtick = str(int(xticks[i]))
        # show x-tick with no more than 4 digits
        axis.text(x - 10, -25, xtick[:min(len(xtick), 5)])

    # Vertical grid
    for y in np.linspace(0, height, yticks):
        axis.hlines(y, xmin=width / 2 + 5, xmax=width, color='k', alpha=0.3)

    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    # Plot score curve
    n = size - len(compound)
    # Plot scaled compound value
    if show_curve:
        axis.plot([(i * (width / 2) * (1/x_max)) + width / 2 + 5 for i in compound],[(i + n) * (height / (len(compound) + 2 * n)) 
            for i in range(len(compound))],color=color_PLUME,label=label_PLUME)

    # Plot y axis by warpping it to the correct dimensions for easier readability
    plt.yticks(
        np.linspace(
            0, height, yticks), labels=np.linspace(
            0, size, yticks, dtype=int))
    axis.get_xaxis().set_visible(False)

    # Plot peak list
    if len(peak_list) > 0:
        # TODO : artificial shift of 1 unit for hlines to properly align peaks with misalignment score, investigate why
        # probably due to the length the score is plotted to (128, 256, 320) compared to score length (127, 255, 319)
        plt.hlines(np.array([peak_list]) * (height / size) + 1, xmin=0,xmax=1200,
           color=color_annot, alpha=1, label=label_annot, linestyles='dotted')
   
    if show_curve:
        plt.legend(loc="upper right")

    ax = plt.gca()
    ax.invert_yaxis()
    matplotlib.rc('font', size=22)


def compute_fundus(volume, width=512, height=512):
    """
    Compute the fundus by resizing the mean of the input volume along the vertical axis.

    Args:
        volume (np.ndarray): OCT data to analyze, in the form of a 3D numpy array.
        width (int, optional): Desired width of the fundus reconstruction. Defaults to 512.
        height (int, optional): Desired height of the fundus reconstruction. Defaults to 512.

    Returns:
        np.ndarray: Fundus image
    """
    # Resize the mean of the input volume along the vertical axis
    fundus = cv2.resize(volume.mean(axis=1), (width, height))

    return fundus

def compute_fundus_along_misalignment(headers: pd.DataFrame, misalignment: pd.DataFrame, n_b_scans: str, name_annot: str, output_dir: str):
    """
    Compute the fundus along with misalignment (Peak-to-Overlaid Curve) and save the result as an image.

    Args:
        headers (pd.DataFrame): DataFrame containing header information.
        misalignment (pd.DataFrame): DataFrame containing misalignment information.
        n_b_scans (str): n_b_scans information.
        name_annot (str): Name identifier for annotation.
        output_dir (str): Output directory for saving the image.
    """
    for idx, row in misalignment.iterrows():
        compound_metric = row
        score = '{:f}'.format(headers.score_displacement[idx])

        # Figure name: quality score _ index in dataframe _ patient_id
        figname = "score_" + str(score) + "_PID" + str(headers["Patient_id"][idx]) + "_" + name_annot

        # Load volume
        volume = np.array(load_images_from_folder(headers['OCT_pathway'][idx]))

        # Plot figure
        fundus = compute_fundus(volume)
        peak_list = headers.frames[idx]
        fundus_along_misalignment(fundus, compound_metric, 'PLUME-OCT',
                         x_max=max(np.max(compound_metric), 30), peak_list=peak_list,
                         label_annot=name_annot,
                         title="total displacement=" + str(headers.score_displacement[idx]))

        #Create directoty if not already there 
        if not os.path.exists(f'{output_dir}/enface'):
            os.mkdir(f'{output_dir}/enface')
        if not os.path.exists(f'{output_dir}/enface/{n_b_scans}'):
            os.mkdir(f'{output_dir}/enface/{n_b_scans}')
        plt.savefig(f"{output_dir}/enface/{n_b_scans}/{figname}.png")
        plt.close()

def main(header_csv, misalignment_csv, paths_images,annot_csv, n_b_scans, name_annot, outpur_dir):
    headers=pd.read_csv(header_csv)
    misalignment=pd.read_csv(misalignment_csv)
    headers,misalignment=joint_header_misalignment(headers, misalignment, paths_images)
    annotation = pd.read_csv(annot_csv)
    annotation=pre_pro_annot(annotation)
    final = headers.merge(annotation, on = "dataset_uiid")
    compute_fundus_along_misalignment(final,misalignment,n_b_scans, name_annot, outpur_dir)
    
if __name__ == "__main__":
    
    parser =  ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/config_enface.json")
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        
        config =json.load(config_file)
        header_csv=config["header_csv"]
        misalignment_csv = config["misalignment_csv"]
        paths_images=config["paths_images"]
        annot_csv = config["annot_csv"]
        name_annot=config["name_annot"]
        n_b_scans = config["n_b_scans"]
        output_dir=config["output_dir"]
        
    main(header_csv, misalignment_csv, paths_images,annot_csv, n_b_scans, name_annot, output_dir)
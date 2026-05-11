# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import json
import ast
from configparser import ConfigParser
from argparse import ArgumentParser
import os

def get_annot_scans_name(paths_images: str) -> list:
    """
    Function to extract the names of scans of interest used for annotation.

    Parameters:
    ----------
    paths_images : str
        Pathway to the directory containing the directory with each B-scans (config parameter).

    Returns:
    -------
    names_scans : list
        List of strings with the names of each scan used for annotation.
    """
    filenames = next(os.walk(paths_images), (None, None, []))[1]
    names_scans = []

    for filenames in filenames:
        names_scans.append((filenames.split("_")[2].split("D")[1]))

    return names_scans

def get_sub_misalignment(header: pd.DataFrame, misalignment: pd.DataFrame) -> pd.DataFrame:
    """
    Align sub_misalignment to the header df. 

    Parameters:
    ----------
    header : pd.DataFrame
        Dataframe containing header information.
    misalignment : pd.DataFrame
        Dataframe containing misalignment signal.

    Returns:
    -------
    sub_misalignment : pd.DataFrame
        Modified misalignment dataframe with NaN values replaced and unnecessary columns dropped.
    """
    sub_misalignment = misalignment.iloc[header.index]
    #replace missing value with nan (#TODO : check if necessary)
    sub_misalignment = sub_misalignment.replace(-9999.0, np.nan)
    sub_misalignment.dropna(axis=1, inplace=True)
    sub_misalignment = pd.DataFrame({"Peaks_Size": sub_misalignment.apply(list, axis=1)})
    return sub_misalignment

def joint_header_misalignment(headers: pd.DataFrame, misalignment: pd.DataFrame, paths_images: str) -> pd.DataFrame:
    """
    Combine header and misalignment data based on annotated scan names.

    Parameters:
    ----------
    headers : pd.DataFrame
        Dataframe containing header information.
    misalignment : pd.DataFrame
        Dataframe containing misalignment signal.
    paths_images : str
        Pathway to the directory containing the directory with each B-scans (config parameter).

    Returns:
    -------
    final_PLUME : pd.DataFrame
        Merged dataframe containing header and misalignment information for annotated scans only.
    """
    name_scan = get_annot_scans_name(paths_images)

    # Get info only for the annotated scan (match scan name)
    headers_mod = headers[[x in name_scan for x in headers.dataset_uiid]]

    # Match and modify the misalignment signal to the new header
    misalignment = get_sub_misalignment(headers_mod, misalignment)

    # Merge both
    final_PLUME = headers_mod.join(misalignment)
    final_PLUME.reset_index(drop=True, inplace=True)

    return final_PLUME

def string_to_float_list(s: str) -> list:
    """
    Convert a string representation of a list of floats to an actual list of floats.

    Parameters:
    ----------
    s : str
        String representation of a list of floats.

    Returns:
    -------
    list
        List of floats.
    """

    s = s[1:-1]
    s = s.replace(',', ' ')
    nbs = s.split(' ')
    return [int(nb) for nb in nbs if nb]

def sub(row: list, a: int = 2) -> list:
    """
    Subtract a constant value from each element in the input list.

    Parameters:
    ----------
    row : list
        Input list of numbers.
    a : int
        Constant value to subtract from each element.

    Returns:
    -------
    list
        List of subtracted values.
    """
    return [x - a for x in row]

def first_scan(row: list) -> np.ndarray:
    """
    Change the first index to 0 if become -1 (because of the re-alignement with PLUME)

    Parameters:
    ----------
    row : list
        Input list.

    Returns:
    -------
    np.ndarray
        Array of unique values treating -1 as 0.
    """
    return np.unique([0 if x == -1 else x for x in row])


def modify_annotation(annotation: pd.DataFrame, name: str) -> None:
    """
    Modify the annotation dataframe for comparison with the PLUME outputs.
    
    Note: For a C-scans of n B-scans, only n-1 B-scan can be annotated (never the last).
    Additionally, the annotation starts from index 1 until n instead of 0 until n-1.
    Thus, to realign, we first convert the list of str to float, then subtract 2, and change the first index to 0 instead of -1.
    

    Parameters:
    ----------
    annotation : pd.DataFrame
        Annotation dataframe.
    name : str
        Name identifier.
    """
    # modification to match PLUME output (realign indexes)
    annotation.frames=annotation.frames.apply(string_to_float_list)
    annotation.frames=annotation.frames.apply(sub)
    annotation.frames=annotation.frames.apply(first_scan)

    # One hot encoding (0 if B-scans not mentionned, 1 if mentionned)
    annotation[name]=list_one_hot(annotation, "frames")

    annotation.rename(columns={"uuid":"dataset_uiid"}, inplace=True)
    annotation.drop(["n_b_scans", "frames"], axis=1, inplace=True)

def one_hot_encode(numbers: list, max_num: int) -> list:
    """
    One-hot encodes a list of numbers up to a specified maximum value.

    Parameters:
    ----------
    numbers : list
        List of peaks detected from the annotation.
    max_num : int
        Number of B-scans.

    Returns:
    -------
    list
        One-hot encoded list of size max_num where one is the position of the peaks detected.
    """
    encoded = [0] * (max_num)  # Create a list of zeros with length max_num + 1
    for number in numbers:
        encoded[number] = 1  # Set the corresponding index to 1
    return encoded

def list_one_hot(df: pd.DataFrame, frame: str) -> list:
    """
    Apply one-hot encoding to a dataframe column.

    Parameters:
    ----------
    df : pd.DataFrame
        Dataframe containing the column to be one-hot encoded.
    frame : str
        Name of the column to be encoded.

    Returns:
    -------
    list
        List of one-hot encoded lists.
    """
    full = []
    max_num = int(df["n_b_scans"][0] - 1)

    for index in range(len(df)):
        list_ = one_hot_encode(df[frame][index], max_num)
        full.append(list_)

    return full


def hist_plot(df: pd.DataFrame, name_1: str, name_2: str, n_b_scans: int, output_dir: str) -> None:
    """
    Plot and save a histogram of the peaks size distribution with PLUME agreement information.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing B-scans information.
    name_1 : str
        Name identifier for annotator 1.
    name_2 : str
        Name identifier for annotator 2.
    n_b_scans : int
        n_b_scans information.
    output_dir : str
        Output directory for saving the plot.
    """
    matplotlib.rc('font', size=18)
    # Define color palette
    #palette = sns.color_palette("RdYlBu", 2)
    palette = ["coral","c"]

    # Create histogram plot
    sns.histplot(df, x="Peaks_Size", hue="Agree", bins=70, palette=palette).set(xlabel="Displacement Landscape")
    # change the figure size
    plt.gcf().set_size_inches(6, 8)

    #percentage of agreement
    #df["Agree"]=df["Agree"].astype("int")
    #hist,bins=np.histogram(df.Peaks_Size,weights=df.Agree, bins=70)
    #hist=hist/np.histogram(df.Peaks_Size, bins=70)[0]
    #bins=np.round(bins.astype("float"),1)

    #to_plot=pd.DataFrame({"Agree":hist,"Disagree":1-hist},index=bins[:-1])
    #to_plot.plot(kind="bar",stacked=True, width= 1.01)

    # Set y-axis to log scale
    plt.yscale('log')

    # Remove spines for better aesthetics
    sns.despine()

    # Set plot title
    #title = f"Height peaks distribution of PLUME-OCT\nAgreement session between {name_1} and {name_2}\nWith n_b_scans of {n_b_scans}"
    #plt.title(title)
    

    # Create output directory structure if not exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(f'{output_dir}/png'):
        os.mkdir(f'{output_dir}/png')
    if not os.path.exists(f'{output_dir}/png/{n_b_scans}'):
        os.mkdir(f'{output_dir}/png/{n_b_scans}')

    #Save it
    plt.savefig(f"{output_dir}/png/{n_b_scans}/{name_1}_{name_2}_{n_b_scans}.png", bbox_inches="tight")
    plt.close()


def main(headers: str, misalignment: str, paths_images: str, n_b_scans: int, annot_1: str, annot_2: str, name_1: str, name_2: str, output_dir: str) -> None:
    """
    Main function to process PLUME and human annotations, merge the data, and generate plots and CSV files.

    Parameters:
    ----------
    headers : str
        Path to the PLUME headers CSV file.
    misalignment : str
        Path to the PLUME misalignment CSV file.
    paths_images : str
        Path to the directory containing the B-scans.
    n_b_scans : int
        n_b_scans information.
    annot_1 : str
        Path to the CSV file for human annotation 1.
    annot_2 : str
        Path to the CSV file for human annotation 2.
    name_1 : str
        Name identifier for annotator 1.
    name_2 : str
        Name identifier for annotator 2.
    output_dir : str
        Output directory for saving the generated files.
    """
    # Read PLUME data
    headers = pd.read_csv(headers)
    misalignment = pd.read_csv(misalignment)

    # Process PLUME data
    final_PLUME = joint_header_misalignment(headers, misalignment, paths_images)

    # Read and modify human annotations
    df_1 = pd.read_csv(annot_1).copy()
    df_2 = pd.read_csv(annot_2).copy()
    modify_annotation(df_1, name_1)
    modify_annotation(df_2, name_2)

    # Merge PLUME and human annotations
    final = pd.merge(df_1, final_PLUME, on="dataset_uiid", how="inner")
    final = pd.merge(df_2, final, on="dataset_uiid", how="inner")

    # Get to B-scan levels
    B_scans = final[["Peaks_Size", name_1, name_2]]
    B_scans = B_scans.apply(pd.Series.explode).reset_index()
    index = (np.where(B_scans[name_1] == B_scans[name_2]))
    B_scans["Agree"] = [True if x in index[0] else False for x in np.arange((n_b_scans - 1) * 20)]

    # Plot results
    hist_plot(B_scans, name_1, name_2, n_b_scans, output_dir)

    # Create output directory structure if not exists
    if not os.path.exists(f'{output_dir}/csv'):
        os.mkdir(f'{output_dir}/csv')
    if not os.path.exists(f'{output_dir}/csv/{n_b_scans}'):
        os.mkdir(f'{output_dir}/csv/{n_b_scans}')

    # Save B-scans to CSV
    B_scans.drop(["Agree", "index"], axis=1).to_csv(f"{output_dir}/csv/{n_b_scans}/B_scans_{name_1}_{name_2}_{n_b_scans}.csv", index=False)


if __name__ == "__main__":
    
    parser =  ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/config_comparison.json")
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        
        config =json.load(config_file)
        headers_csv=config["header_csv"]
        misalignment_csv = config["misalignment_csv"]
        paths_images=config["paths_images"]
        csv_annot_1 = config["csv_annot_1"]
        csv_annot_2 = config["csv_annot_2"]
        name_annot1 = config["name_annot1"]
        name_annot2= config["name_annot2"]
        n_b_scans = config["n_b_scans"]
        output_dir=config["output_dir"]
        
    main(headers_csv, misalignment_csv, paths_images, n_b_scans, csv_annot_1, csv_annot_2, name_annot1, name_annot2, output_dir)
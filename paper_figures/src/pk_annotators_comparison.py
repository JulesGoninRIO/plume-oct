import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns 
import pandas as pd
import ast
import json
import os
from sklearn.metrics import f1_score, precision_score, cohen_kappa_score
from configparser import ConfigParser
from argparse import ArgumentParser
warnings.filterwarnings("ignore", message="Boolean Series key will be reindexed to match DataFrame index")


def divide_peaks(df: pd.DataFrame, list_index: list):
    """
    Divide the DataFrame into four subsets based on peak size thresholds.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing peak size information.
    list_index : list
        List of threshold values for dividing the peaks into four categories.

    Returns:
    -------
        Four DataFrames representing small, medium, high-medium, and high peaks.
    """
    # TODO: Report the length of list_index (I've done test but forgot to report it)
   
    small = df[df.Peaks_Size < list_index[0]]
    medium = df[(df.Peaks_Size < list_index[1]) & (df.Peaks_Size > list_index[0])].reset_index(drop=True)
    high_medium = df[(df.Peaks_Size < list_index[2]) & (df.Peaks_Size >= list_index[1])].reset_index(drop=True)
    high = df[df.Peaks_Size >= list_index[2]].reset_index(drop=True)

    return small, medium, high_medium, high

def get_metrics(df: pd.DataFrame, name: str, col_PLUME: str) -> float:
    """
    Calculate a metric (e.g., F1-score) between two columns of the DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing annotation and PLUME columns.
    name : str
        Name identifier.
    col_PLUME : str
        Column identifier for PLUME.

    Returns:
    -------
    float
        Calculated metric value.
    """
    
    return f1_score(df[name].values, df[col_PLUME].values)

def extract_unique_dict(dict_: dict) -> dict:
    """
    Extract unique arrays from a dictionary of lists.

    Parameters:B_scans_C1_C2_320
    ----------
    dict_ : dict
        Dictionary with lists as values.

    Returns:
    -------
    dict
        Dictionary with unique arrays as values.
    """
    unique_dict = {}
    for key, value_list in dict_.items():
        unique_arrays = [list(x) for x in set(tuple(x) for x in value_list)]
        unique_dict[key] = unique_arrays
    return unique_dict

def extract_metrics_df(
    csv_files: list, directory_path: str, n_b_scans: str, list_: list, name_types: list
) -> dict:
    """
    Extract and calculate metrics from multiple annotation CSV files.

    Parameters:
    ----------
    csv_files : list
        List of CSV files to process.
    directory_path : str
        Directory path where the CSV files are located.
    n_b_scans : str
        n_b_scans information.
    list_ : list
        List of thresholds for peak size categories.
    name_types : list
        List of names/types for peak size categories.

    Returns:
    -------
    dict
        Dictionary containing unique arrays of calculated metrics.
    """
    my_dictionary = {}
    for csv in csv_files:
        all_ = pd.read_csv(f"{directory_path}/{n_b_scans}/{csv}", sep=",")
        name_1 = csv.split("_")[2]
        name_2 = csv.split("_")[3]
        small, medium, high_medium, high = divide_peaks(all_, list_)

        for idx_, df in enumerate([small, medium, high_medium, high]):
            val = get_metrics(df, name_1, name_2)
            my_dictionary.setdefault(f"{name_1}_{name_2}", []).append([val, name_types[idx_]])

    return extract_unique_dict(my_dictionary)

def drop_col_conflict(df: pd.DataFrame) -> list:
    """
    Drop columns from a DataFrame based on a conflict condition.
    (supress double colonne after merge)

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame to process.

    Returns:
    -------
    list
        List of columns to drop.
    """
    col_to_drop = []
    for col in df.columns:
        number = int(col.split("_")[2])
        if number % 2 != 0:
            col_to_drop.append(col)
    return col_to_drop

def convert_dict_to_pd(dict_: dict) -> pd.DataFrame:
    """
    Convert a dictionary to a pandas DataFrame and handle column conflicts.

    Parameters:
    ----------
    dict_ : dict
        Dictionary to convert.

    Returns:
    -------
    pd.DataFrame
        Processed DataFrame.
    """
    df = pd.DataFrame(dict_)
    exploded_df = df.apply(pd.Series.explode, axis=1)

    names=np.unique(exploded_df.columns)
    
    # Rename columns to avoid conflicts
    #exploded_df.columns = [f'{col}_{i%2}' for i, col in enumerate(exploded_df.columns)]
    
    # Add a type column (height Peaks info)
    #col_to_drop = drop_col_conflict(exploded_df)
    #exploded_df["type"] = exploded_df[exploded_df.columns[1]]
    #exploded_df.drop(col_to_drop, axis=1, inplace=True)
    res_array=np.zeros((exploded_df.shape[0]*len(names),3),dtype=np.dtype("U100"))
    for i in range(len(names)):
        tmp=exploded_df.iloc[:,[i*2,i*2+1]].astype(str).to_numpy()
        name_row=np.array([[exploded_df.columns[i*2]]*tmp.shape[0]]).reshape((tmp.shape[0],1))
        tmp=np.append(name_row,tmp,axis=1)
        res_array[i*tmp.shape[0]:i*tmp.shape[0]+tmp.shape[0]]=tmp
    df=pd.DataFrame(res_array,columns=["Annotators","F1_score","Peaks_Size"])
    df["F1_score"]=pd.to_numeric(df["F1_score"])
    return df

def flatten_array(array: list) -> list:
    """
    Flatten a nested list to a single list.

    Parameters:
    ----------
    array : list
        Nested list to flatten.

    Returns:
    -------
    list
        Flattened list.
    """
    return [item for sublist in array for item in sublist]

def df_to_plot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform DataFrame for plotting.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame to transform.

    Returns:
    -------
    pd.DataFrame
        Transformed DataFrame for plotting.
    """
    names = []
    values = []
    types = []

    for col in df.drop("type", axis=1).columns:
        names.append([col.split("_")[0] + "_" + col.split("_")[1]] * len(df[col]))
        values.append(df[col].values)
        types.append(df["type"].values)

    names = flatten_array(names)
    values = flatten_array(values)
    types = flatten_array(types)

    return pd.DataFrame(list(zip(names, values, types)), columns=("Annotators", "F1_score", "Peaks_Size"))


def plot_metric_score(df: pd.DataFrame, n_b_scans: str, output_dir: str, list_separate_peaks: list,
                       type_metric: str, names_type_peaks: list, hue_order: list) -> None:
    """
    Plot metric scores bewteen Annotators with respect to n_b_scans and peak separation.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing metric scores.
    n_b_scans : str
        n_b_scans information.
    output_dir : str
        Output directory for saving the plot.
    list_separate_peaks : list
        List of peak separation information.
    type_metric : str
        Type of metric to plot.
    names_type_peaks : list
        List of names for different peak types.
    hue_order : list
        Order of hues for better visualization.

    """
    matplotlib.rc('font', size=24)
    mean = df.groupby("Peaks_Size").mean()
    std_mean = df.groupby("Peaks_Size").sem()
    palette = sns.color_palette("colorblind", n_colors=len(np.unique(df.Annotators)))
    #sns.set_theme(style="whitegrid", palette=palette)
    sns.set(font_scale=1.5,style="whitegrid", palette=palette)
    g = sns.catplot(data=df, y=type_metric, x="Peaks_Size", hue="Annotators",
                    kind="bar", palette=palette, order=names_type_peaks, hue_order=hue_order).set(xlabel="B-scan displacement",ylabel="F1-score")

    # Access the underlying Matplotlib axes
    ax = g.ax
    

    # Calculate and plot horizontal bars for the mean value of each group
    plt.ylim([0, 1])

    for i, cat in enumerate(names_type_peaks):
        y = mean.loc[cat].values
        y_std = std_mean.loc[cat].values
        x = i * 0.25
        ax.axhline(y, xmin=x, xmax=x + 0.24, color="black", linestyle="--", linewidth=1)
        ax.axvline(i - 0.48, ymin=y - y_std, ymax=y + y_std, color="black", linestyle="--", linewidth=1)
        ax.axvline(i + 0.49, ymin=y - y_std, ymax=y + y_std, color="black", linestyle="--", linewidth=1)

    title = f"{type_metric}_between_PLUME_and_Annotators_with_n_b_scans_of_{n_b_scans}_split_{list_separate_peaks}"
    #plt.title(title)
    #plt.xticks(rotation=45)

    if not os.path.exists(f'{output_dir}/png'):
        os.mkdir(f'{output_dir}/png')
    if not os.path.exists(f'{output_dir}/png/{n_b_scans}'):
        os.mkdir(f'{output_dir}/png/{n_b_scans}')
    plt.gcf().set_size_inches(12, 6)
    plt.savefig(f"{output_dir}/png/{n_b_scans}/{title}.png", bbox_inches="tight")


     
def main (directory_path,names_type_peaks,list_separate_peaks, n_b_scans, outptut_dir,hue_order):
    
    csv_files = [file for file in os.listdir(f"{directory_path}/{n_b_scans}/") if file.endswith('.csv')]
    dic=extract_metrics_df(csv_files, directory_path, n_b_scans, list_separate_peaks,names_type_peaks)
    df=convert_dict_to_pd(dic)
    #final = df_to_plot(df)
    plot_metric_score(df, n_b_scans, output_dir, list_separate_peaks,"F1_score" ,names_type_peaks,hue_order)
    
    
if __name__ == "__main__":
    
    parser =  ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/config_f1_metric.json")
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        
        config =json.load(config_file)
        directory_path=config["directory_path"]
        names_type_peaks = config["names_type_peaks"]
        list_separate_peaks=config["list_separate_peaks"]
        n_b_scans = config["n_b_scans"]
        output_dir=config["output_dir"]
        hue_order=config["hue_order"]
        
    main(directory_path,names_type_peaks,list_separate_peaks, n_b_scans, output_dir,hue_order)
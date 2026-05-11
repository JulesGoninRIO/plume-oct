import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm
import json
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from PIL import Image, ImageFont, ImageDraw
from argparse import ArgumentParser
import pickle
import os

def get_paths_pids(cohort_built_path):
    """
    Get valid paths, to the thumbnail.png files, and their corresponding patient id.

    Parameters
    ----------
    cohort_built_path : str
        The path to the cohort built folder, containing a folder_per_patient/a folder/1 folder per experiment/info.json+thumbnail.png

    Returns
    -------
    paths : dataFrame
        A dataFrame containing the path to the thumbnail.png files and their corresponding patient id.
    """
    #find all paths to the folder containing the thumbnail.png and the info.json
    all_paths = glob.glob(cohort_built_path + "/*/*/*")
    res = []
    for path in tqdm(all_paths):
        info = json.load(open(path + "/info.json"))
        #ensure that we are dealing with the right experiment
        if not "_21013_" in info["parentFile"]["filename"]:
            continue
        patient_id = info["patient"]["patientId"]
        res.append([path + "/thumbnail.png", patient_id])
    return pd.DataFrame(res, columns=["path", "patient_id"])

def get_whole_pannel(df_paths_pids, PLUME_res,sort_by: str="tot_displacement", n_cols: int=80, width: int = 4000, displaying: str="patient_id"):
    df_paths_pids["patient_id"]=df_paths_pids["patient_id"].astype(int)
    PLUME_res["Patient_id"] = PLUME_res["Patient_id"].astype(int)
    # add the PLUME results to the paths_pids dataFrame
    df = df_paths_pids.merge(PLUME_res[["Patient_id",sort_by]], left_on="patient_id", right_on="Patient_id")
    df = df.drop(columns=["Patient_id"])

    n_rows = len(df)//n_cols + int(bool(len(df)%n_cols))
    width = int(width)
    height = int(width *  n_rows / n_cols + 1)
    thumb_width = width//n_cols
    thumb_height = height//n_rows
    print(width,height)

    grid_img=Image.new('L', (width, height))
    
    def get_distances(i,n_cols=n_cols,n_rows=n_rows):
        row, col = divmod(i,n_cols)
        distance_origin = np.sqrt(row**2+col**2)
        distance_center = np.sqrt((n_rows/2-row)**2+(n_cols/2-col)**2)
        return distance_center,distance_origin
    def get_sorted_indexes(n_cols=n_cols,n_rows=n_rows):
        indexes=list(range(n_cols*n_rows))
        indexes=sorted(indexes,key=lambda i: get_distances(i))
        return indexes
    def find_font(pid_ex,thumb_width=thumb_width,thumb_height=thumb_height,max_height_percentage=0.2,margin=0.1,font_size=75):
        #find the font size that fits the pid in the thumb, with a margin
        font = ImageFont.truetype("Pillow/Tests/fonts/DejaVuSans.ttf", font_size)
        width_font,height_font = font.getsize(pid_ex)
        while width_font > thumb_width*(1-margin) or height_font > thumb_height*(1-max_height_percentage):
            font_size-=1
            font = ImageFont.truetype("Pillow/Tests/fonts/DejaVuSans.ttf", font_size)
            width_font,height_font = font.getsize(pid_ex)
        return font
    indexes_sorted=get_sorted_indexes()
    for i,(_,row) in tqdm(enumerate(df.sort_values(by=sort_by).iterrows())):
        r,c= divmod(indexes_sorted[i],n_cols)
        img = Image.open(row["path"]).convert('L')  # Convert to grayscale
        img = img.resize((thumb_width,thumb_height), Image.ANTIALIAS)
        grid_img.paste(img, (c * thumb_width, r * thumb_height))
    if displaying:
        #convert to int, rounded float or str
        if df[displaying].astype(str).str.isnumeric().all():
            df[displaying]=df[displaying].astype(float)
            if df[displaying].apply(float.is_integer).all():
                df["to_display"]=df[displaying].astype(int).astype(str)
            else:
                df["to_display"]=df[displaying].round(3).astype(str)
        else:
            df["to_display"]=df[displaying].astype(str)
        margin=0.1
        font = find_font(df["to_display"].max(),margin=margin)
        grid_img_draw=ImageDraw.Draw(grid_img)
        for i,(_,row) in tqdm(enumerate(df.sort_values(by=sort_by).iterrows())):
            r,c= divmod(indexes_sorted[i],n_cols)
            # add the pid to the image as text
            grid_img_draw.text((c * thumb_width + thumb_width * margin/2, r * thumb_height), row["to_display"], font=font)
    grid_img.save("grid_img.png")


def get_enface_fig(df_paths_pids, PLUME_res, n_rows_tot: int=10, n_bins: int=10, n_rows: int=2, sort_by: str="tot_displacement"):
    """
    construct a mosaic figure of the enface views, sorted by PLUME results.

    Parameters
    ----------
    df_paths_pids : dataFrame
        A dataFrame containing the path to the thumbnail.png files and their corresponding patient id.
    PLUME_res : dataFrame
        A dataFrame containing the PLUME results.
    n_rows_tot : int, optional
        The total number of images chosen for each column, only the n_rows first will be displayed, by default 10
    n_bins : int, optional
        The number of bins to sample the images from, by default 10
    n_rows : int, optional
        The number of rows of images to display, by default 2
    sort_by : str, optional
        The column to sort the images by, by default "tot_displacement"
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the mosaic of enface views.
    """
    print("building figure")
    plt.rcParams.update({'font.size': 22})
    #set random seed for reproducibility
    np.random.seed(1)
    df_paths_pids["patient_id"]=df_paths_pids["patient_id"].astype(int)
    PLUME_res["Patient_id"] = PLUME_res["Patient_id"].astype(int)
    # add the PLUME results to the paths_pids dataFrame
    df = df_paths_pids.merge(PLUME_res[["Patient_id",sort_by,"SNR"]], left_on="patient_id", right_on="Patient_id")
    df = df.drop(columns=["Patient_id"])
    
    # sample the data to the desired number of images, keeping the same number of images per bin
    bins = np.logspace(np.log10(df[sort_by].min()), np.log10(df[sort_by].max()), n_bins+1)
    df = df.groupby(pd.cut(df[sort_by], bins=bins)).apply(lambda x: x.sample(n=n_rows_tot))

    # create the figure, with on top the score distribution on the whole cohort and on the bottom the mosaic of enface views, each bins in a column, and each column properly aligned on the score distribution
    size_up = 3
    width = 20

    left_margin = 0.07
    width_available=width*(1-left_margin)
    hspace = 0.2
    height_grid = n_rows * width_available/n_bins#ensure the square
    height = (size_up + height_grid)/(1-hspace)


    fig = plt.figure(figsize=(width,height))
    g = gridspec.GridSpec(2, 1,figure=fig,height_ratios=[size_up, height_grid])
    g.update(left=left_margin , right=0.98, bottom=0, top=1,wspace=0.0,hspace=hspace)
    ax = fig.add_subplot(g[0])
    hist = sns.histplot(x=PLUME_res[sort_by],ax=ax,element='step',log_scale=True).set(yticks=[1000,2000],xticks=[],xlabel="total displacement [mm]")
    ax.set_xlim([df[sort_by].min(),df[sort_by].max()])
    plt.xticks(bins,np.round(bins,1))
    gs = gridspec.GridSpecFromSubplotSpec(n_rows, n_bins, subplot_spec=g[1],wspace=0.1,hspace=0)
    #gs.update(left=0 , right=1, bottom=0 , top=1,wspace=0.0,hspace=0.01)
    #fig.subplots_adjust(wspace=0, hspace=0)
    for col,(group,values) in enumerate(df.groupby(pd.cut(df[sort_by], bins=bins))):
        for i,(_,row) in enumerate(values.sort_values(by="SNR",ascending=False).iloc[0:2,:].iterrows()):
            ax = fig.add_subplot(gs[i, col])
            img = np.array(plt.imread(row["path"]))[1:-1,:,:]
            ax.imshow(img)
            ax.axis("off")

    return fig

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/config_mosaic.json")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = json.load(config_file)
        cohort_built_path = config["thumbnails_paths"] #"/data/soin/octgwas/Results_cohortbuilder/enface_view/UKBB/"
        PLUME_results_path = config["PLUME_res_path"] #"/data/soin/octgwas/test_sauna/oct_quality_sauna/res/csv/128/UKBB_final_headers.csv"
        output_path = config["output_path"]
        if os.path.exists("df_paths_pids.pkl"):
            df_paths_pids = pickle.load(open("df_paths_pids.pkl","rb"))
        else:
            df_paths_pids = get_paths_pids(cohort_built_path)
            pickle.dump(df_paths_pids,open("df_paths_pids.pkl","wb"))
        PLUME_res = pd.read_csv(PLUME_results_path)
        fig = get_enface_fig(df_paths_pids, PLUME_res, n_rows_tot=6,n_bins=10,n_rows=2)
        #get_whole_pannel(df_paths_pids, PLUME_res)
        fig.savefig(output_path)

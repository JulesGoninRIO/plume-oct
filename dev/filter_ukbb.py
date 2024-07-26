import pandas as pd
import numpy as np
import glob

#read the dicom
import pydicom
from scipy.signal import find_peaks

import multiprocessing
from functools import partial
from tqdm import tqdm

import os

def get_score(f, shared_result):
    ds = pydicom.dcmread(f)
    data = ds.pixel_array
    mean = np.mean(data, axis=(0,2))
    mean = mean - np.min(mean)
    peaks, _ = find_peaks(mean, prominence=0.8, distance=50, height=np.max(mean)/3.)
    if len(peaks)>1:
        i_peaks=peaks[np.argsort(mean[peaks])][-2:]
        i_peaks = np.sort(i_peaks)
        min_between_peaks = np.min(mean[i_peaks[0]:i_peaks[1]])
        score = 2*min_between_peaks/(mean[i_peaks[0]]+mean[i_peaks[1]])
        score_distance = i_peaks[1]-i_peaks[0]
    else:
        score = np.nan
        score_distance = np.nan

    shared_result.append([f.split("/")[-1].split("_")[0],score,score_distance])

def parallalized_score(input_path):
    paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".dcm")]
    with multiprocessing.Pool() as pool:
        shared_result = multiprocessing.Manager().list()
        partial_get_score = partial(
            get_score,
            shared_result=shared_result,
        )
        with tqdm(
            total=len(paths),
            desc="Processing UKBB Patients",
            unit="patient",
        ) as pbar:
            for _ in pool.imap_unordered(
                partial_get_score, paths
            ):
                pbar.update(1)
    return pd.DataFrame(list(shared_result), columns=["patient_id", "score_intensity","score_distance"])

if __name__ == "__main__":
    DATA_DIR = "/data/soin/retina/UKbiobank_90947/Clean_UKBB_data/21013_dcm_right_eye/"
    scores=parallalized_score(DATA_DIR)
    scores.to_csv("../res/ukbb_scores_bands.csv", index=False)
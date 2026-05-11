import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

#read the dicom
import pydicom
from scipy.signal import find_peaks
import sys

def plot_score(f):
    ds = pydicom.dcmread(f)
    data = ds.pixel_array
    mean = np.mean(data, axis=(0,2))
    mean = mean - np.min(mean)
    peaks, _ = find_peaks(mean, prominence=0.8, distance=50, height=np.max(mean)/3)
    if len(peaks)>1:
        i_peaks=peaks[np.argsort(mean[peaks])][-2:]
        i_peaks = np.sort(i_peaks)
        min_between_peaks = np.min(mean[i_peaks[0]:i_peaks[1]])
        score = 2*min_between_peaks/(mean[i_peaks[0]]+mean[i_peaks[1]])
        score_distance = i_peaks[1]-i_peaks[0]
    else:
        score = np.nan
        score_distance = np.nan

    plt.plot(mean,label=f.split("/")[-1].split("_")[0]+str(score)[:4]+str(int(score_distance)))
    plt.plot(i_peaks, mean[i_peaks], "x")
    plt.savefig("score.png")


DATA_DIR = "/data/soin/retina/UKbiobank_90947/Clean_UKBB_data/21013_dcm_right_eye/"
patient_id = str(sys.argv[1])
end_file="_21013_0_0.dcm"
plot_score(DATA_DIR+patient_id+end_file)

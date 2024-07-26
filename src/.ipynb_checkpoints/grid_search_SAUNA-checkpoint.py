# Standard libraries
import os
import sys
import random
import ast

# External libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial.distance as distance
from sklearn.model_selection import ParameterGrid
from scipy.signal import find_peaks

# Project libraries
# Path to project folder in order to access src/ folder
# MODIFY IF NECESSARY
project_path = "T:\Studies\OCTGWAS\code\oct_quality"
assert os.path.exists(project_path), "Modify or give a valid path to the project (this notebook needs access to src/)"
sys.path.insert(1, project_path)
from src.image_load import *
from src.oct_plot import *
from src.slice_viewer import *
from src.sauna import *
from src.csv_handling import *
from src.sauna_figures import *
from src.compute_offset import *

# Include path to project to data & results
assert os.path.exists('T:'), 'No T: directory found, connect to hospital server or vpn'
sys.path.insert(1, 'T:')

def grid_search_SAUNA(POC_score, annotations, modality):
    df_256_rd_20 = pd.read_csv(POC_score)
    df_256_rd_20 = df_256_rd_20[df_256_rd_20["modality"] == int(modality)]
    discontinuities = pd.read_csv(annotations)

    headers = ["patient_id","date","study_uuid","dataset_uuid","modality","laterality","folder_path"]
    # Discontinuity dictionnary for annotation-peak association
    DISCONTINUITY = {}
    for UUID in discontinuities['uuid']:
        DISCONTINUITY[UUID] = ast.literal_eval(list(discontinuities[discontinuities["uuid"] == UUID]["frames"])[0])
    # Compound metric definition
    COMPOUND = {}
    for UUID in DISCONTINUITY.keys():
        filtered_df = df_256_rd_20[df_256_rd_20["dataset_uuid"] == UUID]

        # Assuming `headers` is a list of column names in your DataFrame, we can extract the compound metric columns
        # Here, I'm assuming that you want to select all columns from index `len(headers)` to the end of the DataFrame
        compound_metric = np.array(filtered_df.iloc[:, len(headers):])
        compound_metric = compound_metric[compound_metric != -9999.0]
        COMPOUND[UUID] = compound_metric
        
    # Scanning for peaks
    COMPOUND_peaks, COMPOUND_prop = {}, {}
    for UUID, score in COMPOUND.items():
        peak, prop = find_peaks(score, prominence=0.0, width=0.0, height=0.0)
        COMPOUND_peaks[UUID] = peak
        COMPOUND_prop[UUID] = prop

    # Peak association with annotation
    # Associate index of discontinuity with index of peak for every discontinuity in every patient. Initialization by assigning -1 to every discontinuity
    ASSOCIATION = {UUID : [(discont, -1) for discont in DISCONTINUITY[UUID]] for UUID in DISCONTINUITY.keys()}
    # Maximum neighbouring range to look for a peak to associate with 
    neighbour_range = 2
    # Go over every annotation
    for UUID, association in ASSOCIATION.items():
        for index, a_pair in enumerate(association):
            # Extend window range to find detected peak
            for neighbour in range(0, neighbour_range+1):
                discontinuity_neighbourhood = np.array([a_pair[0] + e for e in range(-neighbour, neighbour+ 1)])
                #filtering mask for peaks
                mask = np.isin(discontinuity_neighbourhood, COMPOUND_peaks[UUID])
                discontinuity_neighbourhood = discontinuity_neighbourhood[mask]
                if len(discontinuity_neighbourhood) != 0:
                    peak_max = np.argmax(COMPOUND[UUID][discontinuity_neighbourhood])
                    ASSOCIATION[UUID][index] = (a_pair[0], discontinuity_neighbourhood[peak_max])
                    break

    # Computing distances between annotations and peaks detected, -1 if no peaks were associated 
    distances = {}
    for UUID, association in ASSOCIATION.items():
        for a_pair in association:
            #Compute distance
            if a_pair[1] != -1:
                distance = np.abs(a_pair[0] - a_pair[1])
            else :
                distance = -1
            #Add entry to dictionnary
            if distance in distances.keys():
                distances[distance] += 1
            else :
                distances[distance] = 1

    # Define minimum and maximum of each property
    properties = ['peak_heights', 'prominences']
    param_names = ['height', 'prominence']
    PROPERTIES = {property : {'max' : 0, 'min' : 0, 'values' : []} for property in properties}
    param_grid = {name : [] for name in param_names}
    for index, property in enumerate(PROPERTIES.keys()):
        for UUID in COMPOUND_prop.keys():
            # Find min, max foreach property
            PROPERTIES[property]['max'] = 13   #max(PROPERTIES[property]['max'],np.max(COMPOUND_prop[UUID][property]))
            PROPERTIES[property]['min'] = 0    #min(PROPERTIES[property]['min'],np.max(COMPOUND_prop[UUID][property]))
        # Set different values for 
        PROPERTIES[property]['values'] = np.linspace(PROPERTIES[property]['min'], PROPERTIES[property]['max'], 50)
        param_grid[param_names[index]] = PROPERTIES[property]['values']

    models = list(ParameterGrid(param_grid))
    RESULTS = [{'F1' : 0, 'TP' : 0, 'FP' : 0, 'TN' : 0, 'FN' : 0, 'Precision' : 0, 'Recall' : 0} for i in range(len(models))]

    for index, model in enumerate(models):
        results = {'F1' : 0, 'TP' : 0, 'FP' : 0, 'TN' : 0, 'FN' : 0, 'Precision' : 0, 'Recall' : 0}
        height = model['height']
        prominence = model['prominence']
        
        #Compute peaks with set of parameters
        for UUID in COMPOUND.keys():
            peaks, properties = find_peaks(COMPOUND[UUID], height=height, prominence=prominence)
            #Compute everything that is not a peak
            non_peaks = [elem for elem in np.arange(0, len(COMPOUND[UUID])) if elem not in peaks]
            #Retrieve list of peaks that have been associated with annontation for each patient
            peak_association = [pair[1] for pair in ASSOCIATION[UUID]]
                    #For each peak detected, check if peak is associated with annotation, a TP or FP
            for peak in peaks:
                if peak in peak_association:
                    results['TP'] += (peak_association == peak).sum()
                if peak not in peak_association:
                    results['FP'] += 1
            
            #For each "non-peak", check if non-peak is TN or FN
            for non_peak in non_peaks:
                if non_peak in peak_association:
                    results['FN'] += (peak_association == non_peak).sum()
                if non_peak not in peak_association:
                    results['TN'] += 1
        
        #Compute performance
        if results['TP'] + results['FP'] != 0:
            results['Precision'] = results['TP'] / (results['TP'] + results['FP'])
        if results['TP'] + results['FP'] == 0:
            results['Precision'] = 0
        if results['TP'] + results['FN'] != 0:
            results['Recall'] = results['TP'] / (results['TP'] + results['FN'])
        if results['TP'] + results['FN'] == 0:
            results['Recall'] = 0
        if results['Precision'] + results['Recall'] != 0:
            results['F1'] = 2 * (results['Precision']*results['Recall'] / (results['Precision'] + results['Recall']))
        if results['Precision'] + results['Recall'] == 0:
            results['F1'] = 0
        RESULTS[index] = results.copy()

    # Extract F1s, Precisons, Recalls
    F = [r['F1'] for r in RESULTS]
    # Best F1 performing paramters
    m = np.argmax(F)
    best_f1_score = np.max(F)

    height = models[m]["height"]
    prominence = models[m]["prominence"]

    return height, prominence, best_f1_score



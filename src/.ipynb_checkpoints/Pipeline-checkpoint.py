# Imports
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
#   Standard Libraries
import glob
import os
import sys
import json
import shutil
import datetime
import csv
from time import gmtime, strftime
import time
import tqdm
from pynput.mouse import Button, Controller

#   External libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pydicom import dcmread
from src.image_load import *
from src.csv_handling import *
from src.compute_offset import *
from src.utility import *
from src.sauna import a_scan_normalization
from src.oct_plot import *
from src.sauna_figures import *


class Pipeline:
    def __init__(self, from_idx, to_idx, input_path, output_path, step, bins , gif , min_sample_size, modality, setup, datetime, data_csv,headers, data_json, date_time):
     
        self.step = step
        self.bins = bins
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.min_sample_size =min_sample_size
        self.gif = gif
        self.modality = modality
        self.datetime =date_time
        self.data_csv = data_csv
        self.headers =headers
        if not os.path.exists('T:/Studies'):
            self.setup = "SOIN"
            self.input_path =input_path
            self.output_path = output_path
        if os.path.exists('T:/Studies'):
            self.setup = "OL"
            self.input_path = input_path
            self.output_path = output_path
            self.data_json = json.load(open(self.input_path))
      
        

    
    def compute_POC_OL(self, patient_ids=[]):
        """From a JSON file provided by cohort builder for a cohort, this step compute POC results for each OCT and record data in a CSV file.

        Args:
            input_path (str): path to JSON file to access cohort data
            output_name (str): name of output CSVs
            output_folder (str, optional): path to store generated files. Defaults to "output/cohort_results/".

        Returns:
            list(str) : list of paths of results files
        """
        #self.output_path = self.output_path+"SAUNA_run_"+self.datetime
        #mouse = Controller()
        counter = 0
        # Load cohort recap file, generated using cohort builder tool after download cohort data

        # Name of results file, one for each modailty (length of OCTs)
        #output_file = self.output_path + "/POC_score.csv"

        # Common header for csv file
        # headers = ['patient_id', 'date', 'study_uuid', 'dataset_uuid', 'modality', 'laterality', 'folder_path']
        # Try to load data if it exists, if found, it allows for computing results and append them to existing CSVs
        # One file per modality (number of B-scans in an OCT, 128, 256 or 320)

        # Results computation section
        data=[]
        
        for patient, studies in tqdm.tqdm(self.data_json.items()):
                patient_key = patient
                if patient in patient_ids or not patient_ids:
                    if counter < self.from_idx or counter > self.to_idx:
                        continue
                    for study, datasets in studies.items():
                        for dataset in datasets:
                            
                            if 'OCT_CUBE' in dataset['info']['layerVariants']:
                                counter += 1
                                laterality = dataset['info']['laterality']
                                patient_id = (dataset['info']['patient']['patientId']) # unusedd
                                # study_date = datetime.datetime.strptime(dataset['info']['study']['studyDatetime'][:-5], '%Y-%m-%dT%H:%M:%S')
                                oct_path =  dataset['oct']['folder']
                                index = oct_path.find("OpthalmoLaus")
                                oct_path = oct_path[:index] + "input\\CohortBuilder_run_15_11_2022\\" + oct_path[index:]
                                list_files = [f for f in os.listdir(oct_path) if f[-4:] == '.jpg']
                                modality = len(list_files)
                                
                                if modality == self.modality or self.modality == None:
                                    volume = load_array_from_folder(oct_path)
                                    # A-scan normalization, each image has its column normalized separately 
                                    volume_N = a_scan_normalization(volume)
                                    # POC registration, pixel displacement
                                    dx, dy, matching_height = [], [], []
                                    
                                    for i in range(volume.shape[0]-1):
                                        dx_, dy_, matching_height_ = compute_offset('POC',volume_N[i],volume_N[i+1])
                                        dx.append(dx_)
                                        dy.append(dy_)
                                        matching_height.append(matching_height_)
                                        # Coumpound metric for POC
                                        # TODO : improve quality assessment metric (add noise quantification for e.g.)
                                    compound = np.abs(dx) + np.abs(dy)
                                    while len(compound) < 319:
                                        compound = np.append(compound, -9999)
                                # Specific headers for modality
                                   
                                    data_row = [patient_id,study , dataset['info']['uuid'], modality,laterality,oct_path] + compound.tolist()
                                    data.append(data_row)
                                # TODO : adapt code to record all data in ONE .csv file 
                                # (by assigning -9999 to cells not assigned for OCTs that are 128 or 256 scans long for e.g.)
                                 
                                    #mouse.click(Button.right)
        data=pd.DataFrame(data, columns = ["Patient_id", "study_id", "oct_id", "modality", "lateraility", "OCT_pathway"] + list(np.linspace(0,318, 319)))
        self.POC_data = data
        return data
      
    def compute_POC_SOIN(self):
        """From a JSON file provided by cohort builder for a cohort, this step compute POC results for each OCT and record data in a CSV file.

        Args:
            input_path (str): path to JSON file to access cohort data
            output_name (str): name of output CSVs
            output_folder (str, optional): path to store generated files. Defaults to "output/cohort_results/".

        Returns:
            list(str) : list of paths of results files
        """
        #mouse = Controller()
        counter = 0
        # Load cohort recap file, generated using cohort builder tool after download cohort data
        #self.output_path = self.output_path+"SAUNA_run_"+self.datetime

        # Name of results file, one for each modailty (length of OCTs)
        times = []
        output_file = self.output_path + "/POC_score.csv"
        # Common header for csv file
        # headers = ['patient_id', 'date', 'study_uuid', 'dataset_uuid', 'modality', 'laterality', 'folder_path']
        # Try to load data if it exists, if found, it allows for computing results and append them to existing CSVs
        # One file per modality (number of B-scans in an OCT, 128, 256 or 320)
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.headers + [i for i in range(0, 319)])
            for file in tqdm.tqdm(os.listdir(self.input_path)):
                counter += 1
                if counter < self.from_idx or counter > self.to_idx:
                    continue
                if file.endswith("dcm"):
                    ds = dcmread(self.input_path+file)
                    laterality = ds.ImageLaterality
                    patient_id = ds.PatientID
                    study_date = ds.StudyDate[:4]+'-'+ds.StudyDate[4:6]+'-'+ds.StudyDate[6:]
                    study_date = datetime.datetime.strptime(study_date, '%Y-%m-%d')
                    modality = ds.NumberOfFrames
                    volume = ds.pixel_array
                    # A-scan normalization, each image has its column normalized separately 
                    volume_N = a_scan_normalization(volume)
                    # POC registration, pixel displacement
                    dx, dy, matching_height = [], [], []
                    for i in range(volume.shape[0]-1):
                        dx_, dy_, matching_height_ = compute_offset('POC',volume_N[i],volume_N[i+1])
                        dx.append(dx_)
                        dy.append(dy_)
                        matching_height.append(matching_height_)
                    # Coumpound metric for POC
                    # TODO : improve quality assessment metric (add noise quantification for e.g.)
                    compound = np.abs(dx) + np.abs(dy)
                    while len(compound) < 319:
                        compound = np.append(compound, -9999)
                    # Specific headers for modality
                    data_row = [patient_id, study_date, ds.StudyID, ds.SOPInstanceUID, modality, laterality, self.input_path+file] + compound.tolist()
                    writer.writerow(data_row)  # Write row by row
                # TODO : adapt code to record all data in ONE .csv file 
                # (by assigning -9999 to cells not assigned for OCTs that are 128 or 256 scans long for e.g.)
        print(f"Data written to {output_file} successfully.")
        #mouse.click(Button.right)

        
    def Compute_quality_score(self, lambda_ = 0.1):
        """ From CSVs generated in Step1, this step apply a peak detection algorithm with preset parameters. Then, it aggregate the detected peaks as 
        a score of the sum of the height of the peaks. It will convert this metric by squashing it between 0 and 1 with a sigmoid function. It will generate
        a distribution of these score given a number of bins and sample each bin to record fundus reconstructions along POC score & detected peaks. 
        Additionnally, it can record a GIF animation of the OCTs to better view the content of the OCT.

        Args:
            input_path (str): path to CSVs files
            output_folder (str): path to store output
            bins (int, optional): number of bins to divide score distribution. Defaults to 35.
            min_sample_size (int, optional): sample size to draw from each bin. Defaults to 1.
            SAVE_GIF (bool, optional): option to save gif animation in x- and y-axis. Defaults to False.
            lambda_ (float, optional): parameter for the squashing function, sigmoid. Defaults to 0.1.

        Returns:
            int : reutrn 0 after every output has been generated
        
         
        """
          
        assert (self.POC_data .columns[:len(self.headers)] == self.headers).all(), "Headers of .csv file not matching"

        # From DataFrame, compute quality score from cumulated height derived from pixel displacement
        peaks = []
        cumulated_height = []
        for row in   self.POC_data .iterrows():
            # Load pixel displacement as ndarray
            POC_score = np.array(row[1][len(self.headers):].to_list())

            # Peak detection
            # Two set of parameters found, check data_exploration.ipynb
            # Set 1 : prominence = 11.0418
            # Set 2 : height = 5.5235
            peak, _ = find_peaks(POC_score, height=5.5235)
            peaks.append(peak)
            cumulated_height.append(np.sum(POC_score[peak]))
            
        
        # Compute quality score to achieve a score between 0 and 1
        # sigmoid is centered around median of cumulated height
        # a reasonable parameter (from testing) is l=0.1 for smooth distribution between ) 0 and 1
        # TODO : improve quality assessment metric (add noise quantification for e.g.)
        # TODO : improve quality metric, POC registration can estimate translation from rotation and scaling separately,
        # investigate to see how relevant it is for our case
        quality_score = [1-sigmoid(c - np.median(cumulated_height), l=lambda_) for c in cumulated_height]
        
        
        
       
        # Plot quality score distribution
        if self.bins is None:
            # Defaults to maximum number of filled bins
            for bins in range(100, 10, -1):
                n, bins, _ = plt.hist(quality_score, bins = bins)
                # Check if all bins have one OCT at least
                if np.count_nonzero(n == 0) == 0:
                    break
                plt.close()
        else :
            n, bins, _ = plt.hist(quality_score, bins = self.bins)
        plt.title('Distribution of quality score')
        plt.xlabel('Quality score ('+str(bins) + '#bins)')
        plt.ylabel('OCT Count')
        figname = self.output_path + '/quality_distribution' + '.png'
        plt.savefig(figname, facecolor='w')
        plt.close()
    
        # Creates a csv that contains information to reproduce the figures
        return quality_score, peaks

    def compute_fundus_along_POC(self, quality_score, peaks):
        # Step 2.2 : After generating distribution of score of each OCT, retrieve OCTs from each bin and save fundus reconstruction and x and y "slideshow"
        # GIFs that go through the OCT
        df = self.data_csv
        
        # Subdirectory for fundus_reuslts
        output_fundus = self.output_path + '/fundus_results/'
        output_GIF = self.output_path + '/animated'

        for row in df.iterrows():
            # Extract quality score and convert into string keeping only the 5 decimals after 0
            compound_metric = np.array(row[1][len(self.headers):].to_list())
            compound_metric = compound_metric[compound_metric != -9999.0]
            score = '{:f}'.format(quality_score[row[0]])[2:]
            # Figure name : quality score _ index in dataframe _ patient_id
            figname = score + "_ID" + str(row[0]) +"_PID" + str(row[1]['patient_id']) + "_UUID" + str(row[1]['dataset_uuid'])
            # Load volume 
            if self.setup == "SOIN":
                volume = dcmread(row[1]['folder_path']).pixel_array
            if self.setup == "OL":
                volume = np.array(load_images_from_folder(row[1]['folder_path']))
            # Plot figure
            fundus = compute_fundus(volume)
            fundus_along_POC(fundus, compound_metric, 'compound', x_max=max(np.max(compound_metric) + 10, 50), peak_list = peaks[row[0]] , title = "SAUNA score = " + score)
            plt.savefig(output_fundus + figname + '.png')
            plt.close()

            if self.gif:
                # TODO : adapt aspect of OCT for each modality
                if row[1]['modality'] == 320:
                    aspect = 0.3
                if row[1]['modality'] == 256:
                    aspect = 0.6
                if row[1]['modality'] == 128:
                    aspect = 0.6
                # Animated x
                # TODO : not recommended for many OCTs, optimize or modify to lower computation time
                generate_gif_v2(volume, axis=0, fps=5, name= path_fundus[bin] + '/' +  figname + '_x.gif')
                # Animated y
                #generate_gif(volume, axis=2, aspect=aspect, start=0, end=0, step=1, name= path_fundus[bin] + '/' + figname + '_y.gif')
        return 0
# Imports
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
#   Standard Libraries
import os
import json
import tqdm

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from pydicom import dcmread

#  External libraries
from src.utility import *
from src.compute import a_scan_normalization, main_poc_reg
from src.oct_plot import compute_fundus, fundus_along_POC
import matplotlib.pyplot as plt
 

from concurrent.futures import ThreadPoolExecutor


class SAUNA:
    
    
    def __init__(self, input_path, output_path, modality):
        
        self.modality = modality
        self.headers = ["Patient_id", "modality", "laterality", "OCT_pathway", "dataset_uiid", "z_factor", "x_factor", "y_factor"]
        self.input_path = input_path
        self.output_path = output_path
     
    def compute_POC_parallel(self, ol):
        data = []
        POCs = []

        if ol:
            self.compute_CB_parallel(data, POCs)
        else:
            self.compute_SOIN_parallel(data, POCs)

        self.POC_sig = pd.DataFrame(POCs)
        self.headers = pd.DataFrame(data, columns=self.headers)
        
        
     
    

    def compute_CB_parallel(self, data, POCs):
        # TODO: add argument

        json_file = f"{self.input_path}/OL_browsed.json"
        data_json = json.load(open(json_file))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_CB_dataset, dataset) for datasets in data_json.values() for dataset in datasets]
            for future in tqdm.tqdm(futures, total=len(futures), desc="Processing CB datasets"):
                result = future.result()
                if result:
                    dataset, patient_id, laterality, oct_path, modality, z_factor, x_factor, y_factor = result
                    list_files = [f for f in os.listdir(oct_path) if f[-4:] == '.jpg']
                    modality = len(list_files)

                    volume = load_array_from_folder(oct_path)
                    self.POC_volume(patient_id, laterality, oct_path, modality, z_factor, x_factor, y_factor, volume,
                                    dataset['info']['uuid'], data, POCs)

    def compute_SOIN_parallel(self, data, POCs):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_SOIN_dataset, file) for file in os.listdir(self.input_path)]
            for future in tqdm.tqdm(futures, total=len(futures), desc="Processing SOIN datasets"):
                result = future.result()
                if result:
                    oct_path, laterality, patient_id, modality, dataset_uuid, volume, z_factor, x_factor, y_factor = result
                    self.POC_volume(patient_id, laterality, oct_path, modality, z_factor, x_factor, y_factor, volume,
                                    dataset_uuid, data, POCs)

    def process_CB_dataset(self, dataset):
        for dataset_info in dataset:
            if 'OCT_CUBE' in dataset_info['info']['layerVariants']:
                laterality = dataset_info['info']['laterality']
                oct_path = dataset_info['oct']['folder']
                patient_id = dataset_info['info']['patient']['patientId']
                dataset_uuid = dataset_info['info']['uuid']
                z_factor, _, x_factor, y_factor = dataset_info['oct']['info']['spacing']

                index = oct_path.find("OpthalmoLaus")
                oct_path = oct_path.replace('\\', '/')
                oct_path = self.input_path + oct_path[index:]
                return dataset_info, patient_id, laterality, oct_path, None, z_factor, x_factor, y_factor
        return None

    def process_SOIN_dataset(self, file):
        oct_path = self.input_path + file
        ds = dcmread(oct_path)
        laterality = ds.ImageLaterality
        patient_id = ds.PatientID
        modality = ds.NumberOfFrames
        dataset_uuid = ds.StudyID
        volume = ds.pixel_array
        pixel_measure = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
        z_factor = pixel_measure.SliceThickness
        x_factor, y_factor = pixel_measure.PixelSpacing

        return oct_path, laterality, patient_id, modality, dataset_uuid, volume, z_factor, x_factor, y_factor   
        
        
    def compute_POC(self, ol):
        
        data = []
        POCs = []
        
        if (ol):
            self.compute_CB(data, POCs )
        else:
            self.compute_SOIN(data, POCs)
        
        self.POC_sig = pd.DataFrame(POCs)
        self.headers = pd.DataFrame(data, columns = self.headers)

    def compute_CB(self, data, POCs):
        #TODO : add argument
      
        json_file =f"{self.input_path}/OL_browsed.json"
        data_json = json.load(open(json_file))
       
        for patient, studies in  tqdm.tqdm(data_json.items()):
            for study, datasets in studies.items():
                for dataset in datasets:
                 
                    if 'OCT_CUBE' in dataset['info']['layerVariants']:
                        
                        laterality = dataset['info']['laterality']
                        oct_path =  dataset['oct']['folder']
                        patient_id = dataset['info']['patient']['patientId']
                        dataset_uuid =  dataset['info']['uuid']
                        z_factor, _, x_factor,y_factor= dataset['oct']['info']['spacing']
                       
                        index = oct_path.find("OpthalmoLaus")
                        oct_path=oct_path.replace('\\', '/')
                        oct_path = self.input_path+oct_path[index:]
                        list_files = [f for f in os.listdir(oct_path) if f[-4:]=='.jpg']
                        modality = len(list_files)
                       
                        volume = load_array_from_folder(oct_path)
                        self.POC_volume(patient_id, laterality, oct_path, modality, z_factor,x_factor,y_factor,volume ,  dataset_uuid,data ,POCs)
                        
    def compute_SOIN(self,data, POCs):
        
        for file in tqdm.tqdm(os.listdir(self.input_path)):
           
            oct_path = self.input_path+file
            ds = dcmread(oct_path)
            laterality = ds.ImageLaterality
            patient_id = ds.PatientID
            modality = ds.NumberOfFrames
            dataset_uuid= ds.StudyID
            volume = ds.pixel_array
            pixel_measure =ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
            z_factor= pixel_measure.SliceThickness
            x_factor,y_factor = pixel_measure.PixelSpacing 
            
            self.POC_volume(patient_id, laterality, oct_path, modality, z_factor,x_factor,y_factor,volume, dataset_uuid,data ,POCs)            
        
    def POC_volume(self, patient_id, laterality, oct_path, modality,z_factor,x_factor,y_factor,volume, dataset_uuid,data ,POCs):
        
        if modality == self.modality :
            volume_N =  a_scan_normalization(volume)
            
            # POC registration, pixel displacement
            dx, dy, matching_height = [], [], []
            for i in range(volume.shape[0]-1):
                dx_, dy_, matching_height_ = main_poc_reg(volume_N[i],volume_N[i+1])
                dx.append(dx_)
                dy.append(dy_)
                matching_height.append(matching_height_)
    
            # Compute metric for POC
            compound = np.abs(dx) + np.abs(dy)
            
            # (by assigning -9999 to cells not assigned for OCTs that are 128 or 256 scans long for e.g.)
            #while len(compound) < lim_scans:
                #compound = np.append(compound, -9999)
                
            data_row = [patient_id, modality, laterality, oct_path , dataset_uuid, z_factor,x_factor,y_factor]
            POCs.append(compound)
            data.append(data_row)
        
        
    def Compute_quality_score(self):

       # peaks = []
        heights = []
        
        for row in self.POC_sig.iterrows() :
            height = row[row != -9999.0]
            heights.append(np.sum(height))
            
        factors = ((self.headers["x_factor"]*self.headers["y_factor"])/self.headers["z_factor"])
       
        
       # if(self.quality_metrics=="mean"):
        #    quality_score = [1-sigmoid(c - np.mean(cumulated_height), lambda_) for c in cumulated_height]
        #else:
           # quality_score = [1-sigmoid(c - np.median(cumulated_height), lambda_) for c in cumulated_height]
        
        self.score=heights*factors
        self.area=heights
        

    def add_bool_quality (self, thr=0.4):
        self.quality=np.where(np.array(self.score)>=thr, True, False)

    def compute_fundus_along_POC(self, image_directory, ol):
        # Step 2.2 : After generating distribution of score of each OCT, retrieve OCTs from 
        #each bin and save fundus reconstruction and x and y "slideshow"
        # GIFs that go through the OCT

        for idx, row in self.POC_sig.iterrows() :
            
           
            score = self.score[idx]
            
            # Figure name : quality score _ index in dataframe _ patient_id
            figname = "score_" + str(score) +"_PID" + str(self.headers["Patient_id"][idx])+"_"+self.headers["dataset_uiid"][idx]
            # Load volume 
            
            if(ol):
                volume = np.array(load_images_from_folder(self.headers['OCT_pathway'][idx]))
            else : 
                ds=dcmread(self.headers['OCT_pathway'][idx])
                volume = ds.pixel_array
            #for OL
             #volume = np.array(load_images_from_folder(self.headers['OCT_pathway'][idx]))
            # Plot figure
            fundus = compute_fundus(volume)
            fundus_along_POC(fundus, row, 'compound', x_max=max(np.max(row) + 10, 50), title = "SAUNA score =" + str(self.score[idx]))
        
            plt.savefig(image_directory+ figname + '.png') 
            plt.close()

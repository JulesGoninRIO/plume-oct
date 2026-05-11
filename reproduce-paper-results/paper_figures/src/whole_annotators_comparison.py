import annotators_comparison as annot_comp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast
from configparser import ConfigParser
from argparse import ArgumentParser
import os

if __name__ == "__main__":
    
    parser =  ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/config_whole_comparison.json")
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        
        config =json.load(config_file)
        #find the key names of the annotators
        csv_annots_keys_name = [key.replace("csv_annot_","") for key in config.keys() if "csv_annot_" in key]
        name_annots_keys_name = [key.replace("name_annot_","") for key in config.keys() if "name_annot_" in key]
        #check that the names are the same
        csv_annots_keys_name.sort()
        name_annots_keys_name.sort()
        if csv_annots_keys_name != name_annots_keys_name:
            raise ValueError("The annotators keys names are not matching between csv and name")
        #get the values
        csv_annots = np.array([config["csv_annot_"+name] for name in csv_annots_keys_name])
        name_annots = np.array([config["name_annot_"+name] for name in name_annots_keys_name])
        #move the consensus annotators to their list
        is_consensus = np.array([name.startswith("C") for name in name_annots])
        csv_consensus = csv_annots[is_consensus]
        name_consensus = name_annots[is_consensus]
        csv_annots = csv_annots[~is_consensus]
        name_annots = name_annots[~is_consensus]
        #get the other parameters
        headers_csv=config["header_csv"]
        misalignment_csv = config["misalignment_csv"]
        paths_images=config["paths_images"]
        n_b_scans = config["n_b_scans"]
        output_dir=config["output_dir"]
        for csv_annot_1,name_annot1 in zip(csv_annots,name_annots):
            for csv_annot_2,name_annot2 in zip(csv_annots,name_annots):
                if name_annot1 < name_annot2:#to avoid comparing twice the same pair and avoid the comparison of the same annotator
                    annot_comp.main(headers_csv, misalignment_csv, paths_images, n_b_scans, csv_annot_1, csv_annot_2, name_annot1, name_annot2, output_dir)
        for csv_annot_1,name_annot1 in zip(csv_consensus,name_consensus):
            for csv_annot_2,name_annot2 in zip(csv_consensus,name_consensus):
                if name_annot1 < name_annot2:
                    annot_comp.main(headers_csv, misalignment_csv, paths_images, n_b_scans, csv_annot_1, csv_annot_2, name_annot1, name_annot2, output_dir)
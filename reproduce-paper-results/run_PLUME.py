#   Standard Libraries
import sys
import json
from argparse import ArgumentParser
from tqdm import tqdm
import pickle
from time import time
import pandas as pd
import numpy as np
import os

#   Project resources
from src.plume import PLUME


def main(input_path, output_path, n_b_scans, output_name, display_fundus, save_misalignment, save_pickles, ol):

    # Define a list of tasks to track progress
    tasks = 3
    # Create output directory structure if not exists
    if not os.path.exists(f"{output_path}/enface"):
        os.mkdir(f"{output_path}/enface")
    if not os.path.exists(f"{output_path}/enface/{n_b_scans}"):
        os.mkdir(f"{output_path}/enface/{n_b_scans}")

    # Create a tqdm progress bar with the total number of tasks
    with tqdm(total=tasks, file=sys.stdout) as pbar:

        # Task 1: Loading data
        start = time()
        #

        dataset = PLUME(input_path, output_path, n_b_scans, display_fundus)
        pbar.update(1)

        # Task 2 : Process the data
        if ol:
            dataset.compute_OL_parallel()
        else:
            dataset.compute_UKBB_parallel()
        pbar.update(1)

        # Task 3: Save data
        data = dataset.headers
        misalignment = dataset.misalignment_sig
        columns_to_save = ["Patient_id",
            "n_b_scans",
            "laterality",
            "SNR",
            "tot_displacement",
            "spatially_weighted_displacement"]
            

        # Create output directory structure if not exists
        # TODO : function to create directory and save files

        # pkl data
        if save_pickles:
            if not os.path.exists(f"{dataset.output_path}/pkl"):
                os.mkdir(f"{dataset.output_path}/pkl")
            if not os.path.exists(f"{dataset.output_path}/pkl/{n_b_scans}"):
                os.mkdir(f"{dataset.output_path}/pkl/{n_b_scans}")

            with open(
                f"{dataset.output_path}/pkl/{n_b_scans}/{output_name}_final.pkl", "wb"
            ) as f:
                pickle.dump(data, f)

        # csv data
        if not os.path.exists(f"{dataset.output_path}/csv"):
            os.mkdir(f"{dataset.output_path}/csv")
        if not os.path.exists(f"{dataset.output_path}/csv/{n_b_scans}"):
            os.mkdir(f"{dataset.output_path}/csv/{n_b_scans}")

        data.to_csv(
            f"{dataset.output_path}/csv/{n_b_scans}/{output_name}_headers.csv",
            index=False,
            columns=columns_to_save
        )
        if save_misalignment:
            misalignment.to_csv(
                f"{dataset.output_path}/csv/{n_b_scans}/{output_name}_misalignment.csv", index=False
            )
        pbar.update(1)

        # TODO : bad timing bc paralelization should close the thread of somethig like that
        finish = time()

        print(f"Process finished in {np.round(finish-start,2 )} seconds")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_UKBB.json")
    args = parser.parse_args()

    with open(args.config, "r") as config_file:

        config = json.load(config_file)

        n_b_scans = config["n_b_scans"]
        input_path = config["input"]
        output_path = config["output"]
        output_name = config["name_output"]
        display_fundus = config["display_fundus"]
        ol = config["ol"]
        save_misalignment = config["save_misalignment"]
        save_pickles = config["save_pickles"]

        main(input_path, output_path, n_b_scans, output_name, display_fundus, save_misalignment, save_pickles, ol)

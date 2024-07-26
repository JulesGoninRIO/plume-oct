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
from scipy.stats import norm
import concurrent.futures
import multiprocessing
from functools import partial
from tqdm import tqdm

#  External libraries
from src.utility import *
from src.compute import a_scan_normalization, main_misalignment_reg
from src.oct_plot import compute_fundus, fundus_along_misalignment
import matplotlib.pyplot as plt


class PLUME:

    def __init__(
        self, input_path: str, output_path: str, n_b_scans: str, display_fundus: bool
    ):
        """
        Initialize PLUME object.

        Args:
        _______
        - input_path (str): Path to the input data.
        - output_path (str): Path to the output data.
        - n_b_scans (str): Number of B-scans
        - display_fundus (bool): Whether to display and save en-face view
        """

        self.n_b_scans = n_b_scans
        self.input_path = input_path
        self.output_path = output_path
        self.display_fundus = display_fundus
        self.headers = [
            "Patient_id",
            "n_b_scans",
            "laterality",
            "OCT_pathway",
            "dataset_uiid",
            "z_factor",
            "x_factor",
            "y_factor",
            "tot_displacement",
            "spatially_weighted_displacement",
            "SNR"
        ]
        # shared_data for paralelization
        self.shared_data = multiprocessing.Manager().list()
        self.shared_misalignment = multiprocessing.Manager().list()

    def compute_OL_parallel(self):
        """
        Perform parallel computation for OL data.
        """

        json_file = f"{self.input_path}/OL_browsed.json"
        data_json = json.load(open(json_file))

        # Create a pool of processes
        with multiprocessing.Pool() as pool:
            # Use partial to create a function with fixed parameters
            partial_process_patient = partial(
                self.process_patient_OL,
                self.input_path,
                shared_data=self.shared_data,
                misalignment_shared=self.shared_misalignment,
            )

            with tqdm(
                total=len(data_json.values()),
                desc="Processing OL Patients",
                unit="patient",
            ) as pbar:
                for _ in pool.imap_unordered(
                    partial_process_patient, data_json.values()
                ):
                    pbar.update(1)

        self.misalignment_sig = pd.DataFrame(list(self.shared_misalignment))
        self.headers = pd.DataFrame(list(self.shared_data), columns=self.headers)

    def compute_UKBB_parallel(self):
        """
        Perform parallel computation for ukbb data.
        """
        with multiprocessing.Pool() as pool:
            partial_process_UKBB = partial(
                self.process_UKBB,
                shared_data=self.shared_data,
                misalignment_shared=self.shared_misalignment,
            )
            with tqdm(
                total=len(os.listdir(self.input_path)),
                desc="Processing UKBB Patients",
                unit="patient",
            ) as pbar:
                for _ in pool.imap_unordered(
                    partial_process_UKBB, os.listdir(self.input_path)
                ):
                    pbar.update(1)
        self.misalignment_sig = pd.DataFrame(list(self.shared_misalignment))
        self.headers = pd.DataFrame(list(self.shared_data), columns=self.headers)

    def process_UKBB(
        self,
        file: str,
        shared_data: multiprocessing.Manager().list,
        misalignment_shared: multiprocessing.Manager().list,
    ):
        """
        Process UKBB data for a given file.

        Args:
        _______
        - file (str): Filename to process.
        - shared_data (multiprocessing.Manager().list): Shared list for data.
        - misalignment_shared (multiprocessing.Manager().list): Shared list for misalignment.
        """
        oct_path = os.path.join(self.input_path, file)
        ds = dcmread(oct_path)
        laterality = ds.ImageLaterality
        patient_id = ds.PatientID
        n_b_scans = ds.NumberOfFrames
        dataset_uuid = ds.StudyID
        volume = ds.pixel_array
        pixel_measure = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
        z_factor = pixel_measure.SliceThickness
        x_factor, y_factor = pixel_measure.PixelSpacing

        bollean, misalignment, misalignment_weighted = self.misalignment_volume(volume, n_b_scans)
        if bollean:
            # compute signal to noise ratio
            snr = self.compute_SNR(volume)
            misalignment, data = self.compute(
                misalignment,
                misalignment_weighted,
                snr,
                n_b_scans,
                z_factor,
                x_factor,
                y_factor,
                patient_id,
                dataset_uuid,
                oct_path,
                laterality,
                False,
            )
            misalignment_shared.append(misalignment)
            shared_data.append(data)

    def process_patient_OL(
        self,
        base_path: str,
        patient: dict,
        shared_data: multiprocessing.Manager().list,
        misalignment_shared: multiprocessing.Manager().list,
    ):
        """
        Process OL data for a given patient.

        Args:
        _______
        - base_path (str): Base path for data.
        - patient (dict): Patient information.
        - shared_data (multiprocessing.Manager().list): Shared list for data.
        - misalignment_shared (multiprocessing.Manager().list): Shared list for misalignment.
        """
        for study, datasets in patient.items():
            for dataset in datasets:
                if "OCT_CUBE" in dataset["info"]["layerVariants"]:
                    laterality = dataset["info"]["laterality"]
                    oct_folder = dataset["oct"]["folder"]
                    patient_id = dataset["info"]["patient"]["patientId"]
                    dataset_uuid = dataset["info"]["uuid"]
                    z_factor, _, x_factor, y_factor = dataset["oct"]["info"]["spacing"]
                    oct_folder = oct_folder.replace("\\", "/")
                    index = oct_folder.find("OpthalmoLaus")
                    oct_path = os.path.join(base_path, oct_folder[index:])
                    list_files = [f for f in os.listdir(oct_path) if f[-4:] == ".jpg"]
                    n_b_scans = len(list_files)
                    volume = load_array_from_folder(oct_path)

                    bollean, misalignment, misalignment_weighted = self.misalignment_volume(volume, n_b_scans)

                    if bollean:
                        snr = self.compute_SNR(volume)
                        misalignment, data = self.compute(
                            misalignment,
                            misalignment_weighted,
                            snr,
                            n_b_scans,
                            z_factor,
                            x_factor,
                            y_factor,
                            patient_id,
                            dataset_uuid,
                            oct_path,
                            laterality,
                            True,
                        )
                        misalignment_shared.append(misalignment)
                        shared_data.append(data)

    def compute(
        self,
        misalignment,
        misalignment_weighted,
        snr_b_scans,
        n_b_scans: int,
        z_factor: int,
        x_factor: int,
        y_factor: int,
        patient_id: str,
        dataset_uuid: str,
        oct_path: str,
        laterality: str,
        ol: bool,
    ):
        """
        Compute displacement score and display en-face reconstruction

        Args:
        _______
        - misalignment: misalignment sginal
        - misalignment_weighted: Weighted misalignment signal
        - n_b_scans:  number of B-scan
        - z_factor: Z factor.
        - x_factor: X factor.
        - y_factor: Y factor.
        - patient_id: Patient ID.
        - dataset_uuid: Dataset UUID.
        - oct_path: Path to OCT data.
        - laterality: Laterality information (left or right)
        - ol: Boolean indicating OL data.
        """

        tot_displacement = self.Compute_quality_score(
            misalignment, z_factor, x_factor, y_factor
        )
        spatially_weighted_displacement = self.Compute_quality_score(
            misalignment_weighted, z_factor, x_factor, y_factor
        )
        score_snr = self.compute_total_snr(snr_b_scans)
        data = [
            patient_id,
            n_b_scans,
            laterality,
            oct_path,
            dataset_uuid,
            z_factor,
            x_factor,
            y_factor,
            tot_displacement,
            spatially_weighted_displacement,
            score_snr
        ]
        if self.display_fundus:
            self.compute_fundus_along_misalignment(
                tot_displacement,
                spatially_weighted_displacement,
                snr_b_scans,
                patient_id,
                dataset_uuid,
                oct_path,
                misalignment,
                misalignment_weighted,
                ol,
            )

        return misalignment, data

    def generate_gaussian(self, n_b_scans: int, factor: int):
        """
        Generate Gaussian distribution.

        Args:
        _______
        - n_b_scans (int): n_b_scans information.
        - factor (int): Factor for Gaussian distribution.
        """

        # TODO  add argment in config file
        scale = n_b_scans / 10
        center = n_b_scans / 2
        x = np.linspace(0, n_b_scans - 2, n_b_scans - 1)

        return factor * (scale) * norm.pdf(x, loc=center, scale=scale)

    def compute_SNR(self, volume: np.ndarray, cutoff_percentage=0.2, min_n_lines=2, db=True):
        """
        Compute signal to noise ratio, using the formula: std(signal) / mean(noise). (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6865101/).
        To get the region on interest (roi) and background area, take the cutoff_percentage of horizontal lines
        (since ~follow retina) with lowest/highest values as background/roi

        Args:
        _______
        - volume: Input volume.
        - cutoff_percentage: The cutoff_percentage of the y axis will be used as roi and background area
        - min_n_lines: if less than min_n_lines has zero variation use 0 as SNR
        - db: if true return the result in decibels
        """
        assert (
            len(volume.shape) == 3
        ), " Provided array has the wrong number of dimensions"
        # find the lines that have only identical values
        is_identical = np.std(volume, axis=2) == 0
        res = np.zeros(len(volume))+1
        for i in range(volume.shape[0]):
            # remove the lines composed of identical values (probably padding added to the start and end of the volume)
            bscan = volume[i, ~is_identical[i, :], :]
            if bscan.shape[0] < min_n_lines:
                # not enough lines left -> all identical -> let 1 as SNR
                continue
            # take as background the cutoff_percentage lines with the mean median value
            sorted_lines_i=np.argsort(np.mean(bscan, axis=1))
            background = bscan[sorted_lines_i[:max(int(cutoff_percentage * bscan.shape[0]), min_n_lines)]]
            # take as roi the lines with the cutoff_percentage highest mean value
            roi = bscan[sorted_lines_i[-max(int(cutoff_percentage * bscan.shape[0]), min_n_lines):]]
            # compute the snr
            res[i] = np.mean(roi) / np.std(background)
        if db:
            res = 20*np.log10(res)
        return res

    def misalignment_volume(self, volume, n_b_scans: int, factor=10):
        """
        Compute misalignment for a given volume.

        Args:
        _______
        - volume: Input volume.
        - n_b_scans: n_b_scans information.
        - factor: Factor for computation.

        Returns:
        ________
        - misalignment: the values for each B-scan.
        - misalignment_weighted: the values for each B-scan weighted by a gaussian distribution to give more importance to the center of the retina.
        """

        if n_b_scans == self.n_b_scans:
            volume_N = a_scan_normalization(volume)

            # misalignment registration, pixel displacement
            dx, dy, matching_height = [], [], []
            for i in range(volume.shape[0] - 1):
                dx_, dy_, matching_height_ = main_misalignment_reg(volume_N[i], volume_N[i + 1])
                dx.append(dx_)
                dy.append(dy_)
                matching_height.append(matching_height_)

            # Compute metric for misalignment
            misalignment = np.abs(dx) + np.abs(dy)
            gaussian = self.generate_gaussian(n_b_scans, factor)
            misalignment_weighted = misalignment * gaussian
            return True, misalignment, misalignment_weighted
        else:
            return False, None, None

    def Compute_quality_score(self, misalignment, z_factor: int, x_factor: int, y_factor: int):
        """
        Compute quality score for misalignment.

        Args:
        _______
        - misalignment: misalignment information.
        - z_factor: Z factor.
        - x_factor: X factor.
        - y_factor: Y factor.
        """
        heights = np.sum(misalignment)
        factors = (x_factor * y_factor) / (z_factor)
        tot_displacement = heights * factors

        return tot_displacement
    
    def compute_total_snr(self, snr_b_scans):
        """
        Get one SNR score for the whole C-scan

        Args:
        _______
        - snr_b_scans: Signal to Noise Ratio for all the b-scans.
        """
        return np.mean(snr_b_scans)

    def compute_fundus_along_misalignment(
        self,
        tot_displacement,
        spatially_weighted_displacement,
        snr_b_scans,
        patient_id,
        dataset_uuid,
        oct_path,
        misalignment,
        misalignment_weighted,
        ol,
    ):
        """
        Compute fundus information along misalignment.

        Args:
            - tot_displacement: Score displacement information.
            - spatially_weighted_displacement: Weighted score displacement information.
            - snr_b_scans: signal to noise ratio for all the b-scans.
            - patient_id: Patient ID.
            - dataset_uuid: Dataset UUID.
            - oct_path: Path to OCT data.
            - misalignment: misalignment information.
            - misalignment_weighted: Weighted misalignment information.
            - ol: Boolean indicating OL data.
        """

        # Figure name : quality score _ index in dataframe _ patient_id
        figname = f"tot_displacement_{str(np.round(tot_displacement, 2))}_weighted_s_{str(np.round(spatially_weighted_displacement,2))}_PID{str(patient_id)}_{dataset_uuid}"
        # Load volume

        if ol:
            volume = np.array(load_images_from_folder(oct_path))

        else:
            ds = dcmread(oct_path)
            volume = ds.pixel_array

        # Plot figure
        fundus = compute_fundus(volume)
        fundus_along_misalignment(
            fundus,
            misalignment,
            misalignment_weighted,
            "misalignment",
            "weigthed_misalignment",
            x_max=max(np.max(misalignment) + 10, 50),
            title="Tot displacement =" + str(tot_displacement),
            snr_b_scans=snr_b_scans,
        )

        plt.savefig(f"{self.output_path}/enface/{self.n_b_scans}/" + figname + ".png")
        plt.close()

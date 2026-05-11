# PLUME-OCT DICOM Demo

This folder contains a minimal Python script to run **PLUME-OCT** on an OCT DICOM file or on a folder containing OCT DICOM files.

PLUME-OCT stands for **Phase-Level Unified Metric Evaluation for OCTs**. It is an automated quality-control method for OCT cubes that quantifies misalignment between consecutive B-scans. It also computes a Signal-to-Noise Ratio metric and displays a fundus-like reconstruction from the OCT volume.

## Background

Optical coherence tomography (OCT) is widely used for in vivo retinal imaging and provides 3D retinal volumes, also called OCT cubes. These volumes are composed of multiple 2D B-scans.

Classical quality-control methods often evaluate the quality of individual B-scans. However, small eye or head movements during acquisition can cause misalignment between B-scans. These misalignments may affect the quality and reproducibility of 3D retinal biomarkers.

PLUME-OCT was developed to measure this volumetric coherence and provide automated OCT cube quality metrics without requiring reference images.

## What this script does

The script:

1. Loads an OCT DICOM file or a folder of OCT DICOM files.
2. Converts the OCT data into a 3D NumPy volume.
3. Computes PLUME-OCT metrics:
   - number of B-scans
   - total displacement
   - weighted displacement
   - SNR
4. Prints the results in the terminal.
5. Displays a fundus reconstruction with the PLUME results in the title.

## File

Main script:

```text
run_plume_dcm.py
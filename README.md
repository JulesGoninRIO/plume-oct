# **OCT Quality : Assess quality of an OCT at C-scan level**

# Background 

Python code & Jupyter notebooks  that encapsulate the work done on assessing quality of an OCT at the C-level. The project started on 02.2022 with G. Molas ,A. Bouillon on 09.2022, L Gürtler on 02.2023, has been taken on by A.Milloz and finalized by Y.Paychere. The overall aim is to develop a pre-processing tools to assign a score to OCT cube in order to define them as useable or not for further study.


This pipeline contains **3 different parts** of the project :<span style="color:lightblue"> *the annotation study*</span>, <span style="color:lightseagreen"> *PLUME algorithm* </span> and the code used to <span style="color:lightgreen"> *generate the figures*</span> in the paper.

## Files structure
```

├── config
│   ├── config_OL.json
│   └── config_UKBB.json
├── doc
│   ├── ANNOTATION GUIDELINES.pdf
│   └── requirements.txt
├── libs
│   └── requirements.txt
├── README.md
├── run_plume.py
├── src
│    ├── compute.py
│    ├── final_app.py
│    ├── image_load_app.py
│    ├── oct_plot.py
│    ├── prism.py
│    ├── utility.py
│    └── visu.ipynb
├── paper_figures
   ├── config
   │   ├── config_comparison.json
   │   ├── config_enface.json
   │   ├── config_f1_metric.json
   │   ├── config_mosaic.json
   │   ├── config_SNR.json
   │   └── config_whole_comparison.json
   ├── image
   │   ├── ANNOTATION_GUIDELINES.pdf
   │   └── annotators_reparition.png
   ├── README.md
   └── src
       ├── annotators_comparison.py
       ├── build_enface_distribution.py
       ├── pk_annotators_comparison.py
       ├── plot_enface_annot.py
       ├── snr_analysis.py
       └── whole_annotators_comparison.py
```

## Implementation

#### <span style="color:lightblue"> *ANNOTATIONS STUDY* : </span>

This part of the project consists of manually annotating a set of annotated OCTs to further explore PLUME algorithm capacity. To do so, four annotators manually annotated 20 OCTs (256 and 320 B-scans in resolution). To achieve this, annotators had access to an annotation app and manually annotated the images using the internally developed application.

`final_app.py` : Annotation App that could be use to annotate C-scans and also do a consensus session. (cf Figure ..)

`image_load_app.py`  : contains plotter function to visualize scans.

`oct_plot.py`  :contains plotter function to visualize the results.


#### How to use 

##### Annotation App

```
python final_app.py
```

#### <span style="color:lightseagreen"> *PLUME ALGORITHM* : </span>
This part  of the project is the main one  as it incorporates the PLUME algorithm, designed to automatically assess misalignment between B-scans from OCT. It provides two types of measurements: a displacement score and a weighted version of it.

`run_PLUME.py` initiates PLUME Class, run the algorithm and save the results. More specifically, this files takes as an input argument a config file that contains all the parameters of the class to run and load the results into pkl  and a csv format . 

`PLUME.py` contains the main function initiates PLUME class and compute the misalignment scores for each OCT cube.

`compute.py` contains functions for the pre-processing of the data as well as for the misalignment method.

`oct_plot.py` contains plotter function to visualize the results.

`utility.py` utils function.

`visu.ipynb` to visualize the results

output are store in `pkl/`, `csv/` and `enface/` directory  in the output path given in the config path (automatically generated). 

#### How to use 

Example (with a customized configuration file) The configuration files must be available in the `configs/` directory and named `config.json` .\
Simply run the following (as an example):
```
python run_PLUME.py --config config/config.json
``` 
You can also use the json file by default named `config_PLUME.json` and run the following : 
```
python run_PLUME.py 
``` 
## Figures
The code used to generate figures can be found in `paper_figures`, with a README describing it

## Libraries
In the following commands replace `path/to/env/` by the path where you want to store the environment.
``` 
conda create -p /path/to/env/envPLUME python=3.6 
conda activate /path/to/env/envPLUME
pip install -r libs/requirements.txt
```
## Configuration file 


#### <span style="color:lightseagreen"> *PLUME ALGORITHM* : </span>

- **`n_b_scans`[int]**: resolution of the scan (number of B-scans)
- **`input`[str]**: Path to the input directory where OCT are stored.
- **`ouput`[str]**: Path to the output directory where results will be stored.
- **`name_output`[str]**: A customizable name for the output
- **`display_fundus`[str]**: Boolean indicating whether to display the fundus. In your configuration, it is set to true
- **`ol`[str]**:  Boolean indicating the usage of OL
 # PLUME-OCT AND HUMAN ANNOTATIONS

 ## Background
 This experiments manual, a subproject of the PLUME-OCT initiative, focuses on validating the algorithm's capacity to detect misalignments in OCT images using C-scans for computational vision. 
 
 To confirm the algorithm's efficacy, we employed a bespoke annotation app, where four annotators manually annotated 40 scans (256 and 320 B-scans).Annotators worked in pairs, individually annotating B-scans, and then participated in a "consensus" session ([cf set up](image/annotators_reparition.png)). The subsequent code assessed inter-annotator differences, compared F1-scores based on peak height, and presented visualizations for a comprehensive evaluation of human and algorithmic capabilities. 
 
 This folder includes three main analyses: (i) a comparison among annotators, (ii) exploration of peak height distribution with inter-annotator agreement, and (iii) visualization of PLUME-OCT and human ability to detect misalignments. Note that these codes ares designed to be executed sequentially in the right order.

 On top of these analysis this folder contains the script used to generate a correlation between the displacement score and the SNR for a given output of PLUME-OCT.


[cf guideline for annotations ](image/ANNOTATION_GUIDELINES.pdf)


  
## Files structure
```
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

## Generate the figures 
0) create the environment for PLUME-OCT (from the root directory of the repository):

In the following commands replace `path/to/env/` by the path where you want to store the environment.
``` 
conda create -p /path/to/env/envPLUME python=3.6 
conda activate /path/to/env/envPLUME
pip install -r libs/requirements.txt
```
1) Go to paper_figures/src


2) `whole_annotators_comparison.py`: designed for comparing and visualizing peak height distributions in B-scans obtained from PLUME-OCT  against human annotations agreement. The output includes histogram plots and CSV file for further analysis.

#### How to use

 **With a customized configuration file**: 

```
python whole_annotators_comparison.py --config $pathway_to_the_config/config.json
```
**Using the default files structure and config file (config_whole_comparison.json)** : 

```
python  whole_annotators_comparison.py
```

#### Configuration file

- **`header_csv` [csv file]:** File containing the header csv file from PLUME-OCT.
  
- **`misalignment_csv` [csv file]:** File containing the misalignment signal from PLUME-OCT.
  
- **`paths_images` [str]:** Directory where sets of B-scans are stored in directories named as PID$PID_UUID$UUID. 

(details : This parameter is utilized to extract the appropriate set of images for analysis. By specifying this directory, the script can automatically select the relevant information from PLUME-OCT for analysis, even if the PLUME-OCT output contains additional scans that may not have been manually annotated. This simplifies the process of selecting only the annotated images for comparison.) 

- **`csv_annot_1` [csv file]:** Annotation file of one annotator (output from the application).
  
- **`csv_annot_2` [csv file]:** Annotation file of another annotator (output from the application).

- ...

- **`csv_annot_n` [csv file]:** Annotation file of another annotator (output from the application).

(the plots generated for the study contains 6 annotators)

- **`name_annot_1` [csv file]:** Name of the annotator 1, should begin by C in case the corresponding csv corresponds to a consensus.
  
- **`name_annot_2` [csv file]:** Name of the annotator 2, should begin by C in case the corresponding csv corresponds to a consensus.

- ...

- **`name_annot_n` [csv file]:** Name of the annotator n, should begin by C in case the corresponding csv corresponds to a consensus.

(the plots generated for the study contains 6 annotators, including two consensus)

- **`n_b_scans` [number]:** n_b_scans of the C-scans (number of B-scans).
  
- **`output_dir` [str]:** Directory to save results.

### Annotators and Peak Size  :

3) `pk_annotators_comparison.py`: Compare F1 metric scores between human annotators along with peak height. The script processes and analyzes data from PLUME-OCT outputs and manual annotations, generating visualizations to assess agreement between annotators according to the class of the peak height from PLUME-OCT. 


#### How to use

 **With a customized configuration file**: 

```
python pk_annotators_comparison.py --config $pathway_to_the_config/config.json
```
**Using the default files structure and config file (config_f1_metric.json)** : 

```
python  pk_annotators_comparison.py
```

#### Configuration file

- **`directory_path` [str]**: Path to the directory containing csv outputs from annotators_comparison.py.

- **`names_type_peaks` [list of str]** : Names of the peak categories for analysis.

- **`list_separate_peaks` [list of int]** : Thresholds for separating peak categories.

- **`n_b_scans` [int]**: n_b_scans of C-scans (number of B-scans).

- **`output_dir` [str]**: Directory to save results and visualizations.

- **`hue_order` [list of str]** : Order of annotator names for consistent plotting.

⚠️ If `names_type_peaks` has 'n' values, then `list_separate_peaks` should have 'n-1' values. Additionally, the `hue_order` should correspond to the names of each CSV file in the `directory_path`. For example, if it contains files like 'annot_A1_A2_x.csv' and 'annot_A3_A4_x.csv', then `hue_order` is ['A1_A2', 'A3_A4'].


### Comparison bewteen anntotators :

4) `plot_enface_annot.py`: designed to generate enface plots along with PLUME-OCT misalignment and human annoatation. It processes OCT data, computes fundus images, and plots fundus reconstruction alongside with the misalignment (from PLUME-OCT) and human annotation.

Nb : What we refer to as "human annotation" is essentially a record of when a human classifies the corresponding B-scan as misaligned (represented as lines) and nothing when not.


#### How to use

 **With a customized configuration file**: 

```
python  plot_enface_annot.py --config $pathway_to_the_config/config.json
```
**Using the default files structure and config file (config_f1_metric.json)** : 

```
python  plot_enface_annot.py
```

#### Configuration file

- **`header_csv` [csv]**:File containing the header csv file from PLUME-OCT.

- **`misalignment_csv` [csv]**: File containing the misalignment from PLUME-OCT.

- **`paths_images` [str]**: Directory where sets of B-scans are stored in directories named as PID$PID_UUID$UUID.

- **`annot_csv` [csv]**: Annotation file of one annotator (output from the application).

- **`n_b_scans` [int]**: n_b_scans of C-scans (number of B-scans).

- **`name_annot` [str]** : Name of the annotators linked to the `annot_csv`. 

 - **`output_dir`[str]**: where to save the results


### Comparison bewteen SNR and total displacement :

5) `snr_analysis.py`: this file computes a 2D density histogram between the total displacement and the SNR, based on results returned by PLUME-OCT. It also returns the associated Pearson r value.


#### How to use

 **With a customized configuration file**: 

```
python  snr_analysis.py --config $pathway_to_the_config/config.json
```
**Using the default files structure and config file (config_SNR.json)** : 

```
python  snr_analysis.py
```

#### Configuration file

- **`header_csv` [csv]**: File containing the header csv file from PLUME-OCT.

- **`output_dir`[str]**: where to save the results

### Visualization of total displacement score distribution :

6) `build_enface_distribution.py`: this file display the distribution of the total displacement in the UK biobank and a few representative OCTs below it.


#### How to use

 **With a customized configuration file**: 

```
python  build_enface_distribution.py --config $pathway_to_the_config/config.json
```
**Using the default files structure and config file (config_mosaic.json)** : 

```
python  build_enface_distribution.py
```

#### Configuration file

- **`thumbnails_paths` [csv]**: path to the UKBB built with CohortBuilder, and SVG of the thumbnails converted to PNG using `../dev/convert_svg_png.py`

- **`PLUME_res_path`[str]**: where the output of PLUME (the header) for the UK biobank is stored

- **`output_path`[str]**: where to save the results

##  Output structure

Codes automatically creates subfolder  (csv, png, enface, ..) inside the output directory. 

```
├── output_directory
    ├── csv
        ├──n_b_scans
    ├── pkl
        ├──n_b_scans
    ├── enface
        ├──n_b_scans
    SNR_density_plot.png
```
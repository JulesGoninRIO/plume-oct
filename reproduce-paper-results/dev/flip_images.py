import numpy as np
#to read the dicom
import pydicom
import sys
from pydicom.uid import RLELossless
import glob

from argparse import ArgumentParser
import json
from tqdm import tqdm

def flip_images(path, output_folder):
    """
    Args:
        path: str, path to the dicom file

    Returns:
        None
    """
    ds = pydicom.dcmread(path)
    #flip the image
    data = ds.pixel_array
    assert len(data.shape) == 3
    data_flipped = np.flip(data, axis=1).copy()
    # change patient id
    #new_value_f = ds[0x0010, 0x0020].value + "999"
    #new_value = new_value_f + "^" + new_value_f
    #ds[0x0010, 0x0010].value = new_value
    ds.compress(RLELossless, data_flipped)
    ds.save_as(output_folder + "/" + path.split("/")[-1])
    #display_dicom(path.replace(".dcm", "_flipped.dcm"))

def display_dicom(path):
    """
    display the dicom file

    Args:
        path: str, path to the dicom file

    Returns:
        None
    """
    import napari
    ds = pydicom.dcmread(path)
    data = ds.pixel_array
    viewer = napari.Viewer()
    viewer.add_image(data)
    napari.run()

if __name__ == "__main__":
    parser =  ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config file", default="config_pids.json")
    with open(parser.parse_args().config) as f:
        config = json.load(f)
    dcm_folder = config["dcm_folder"]
    ids = config["pids"]
    output_folder = config["output_folder"]
    pbar = tqdm(ids)
    for i in pbar:
        # find the path which is in dcm_foler and starts with i
        path = glob.glob(dcm_folder + f"/{i}*.dcm")[0]
        if len(path) == 0:
            print(f"Patient {i} not found")
            continue
        flip_images(path, output_folder)
        pbar.set_description("Processing %s" % i)
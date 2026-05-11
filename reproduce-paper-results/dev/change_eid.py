import pydicom
import glob
def change_eid(path):
    dicoms=glob.glob(path + "*.dcm")
    for p in dicoms:
        ds = pydicom.dcmread(p)
        # change patient id
        new_value_f = ds[0x0010, 0x0020].value + "000"
        new_value = new_value_f + "^" + new_value_f
        ds[0x0010, 0x0010].value = new_value
        ds.save_as(p)

change_eid("/data/soin/octgwas/to check/")
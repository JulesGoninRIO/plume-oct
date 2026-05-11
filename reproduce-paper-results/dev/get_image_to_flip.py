import pandas as pd
import numpy as np

file = "/data/soin/octgwas/octgwas/res/filtered_out.csv"
df = pd.read_csv(file)
df = df[df.reason=="unspecified thickness"]
to_print = '", "'.join(df.eid.astype(int).astype("string"))
to_print = '"'+to_print+'"'

f=open("ied_to_flip.txt","w")
f.write(to_print)
f.close()
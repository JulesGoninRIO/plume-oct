import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as clr

df=pd.read_csv("../res/ukbb_scores_bands.csv")
#df=df[df.score_distance>75]
res_plume=pd.read_csv("../res/csv/128/UKBB_final_headers.csv")
df=res_plume.merge(df,left_on="Patient_id",right_on="patient_id")[["Patient_id","tot_displacement","SNR","score_intensity","score_distance"]]
#df.score_distance.hist(log=True,bins=100)
sns.histplot(
        x=df.score_distance, y=df.tot_displacement,
        bins=100, log_scale=(True, True),
        cbar=True, norm=clr.LogNorm(),vmin=None, vmax=None
    )
plt.savefig("hist.png")
df=df[(df.score_distance>100) & (df.tot_displacement<2) & (df.score_intensity<0.1)]
df["eid"] = df["Patient_id"]
df["columns"] = "all"
df["reason"] = "OCT without retina"
df=df[["eid","columns","reason"]]
df.to_csv("OCTs_to_exclude.csv",index=False)
print(df)
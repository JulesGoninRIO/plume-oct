import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("../res/csv/128/UKBB_final_headers.csv")
on="tot_displacement"
n_bins=10
n_tot=8000

df[on].hist(log=True,bins=100)
plt.savefig("../res/distribution"+str(on)+".png")
if on=="SNR":
    bins = np.linspace(min(df[on]),max(df[on]),n_bins)
else:
    bins = np.logspace(np.log10(min(df[on])*0.9), np.log10(max(df[on])*1.1), n_bins)
df=df.sample(frac=1).reset_index()
df["bins"]=pd.cut(df[on], bins)
res=[]
for x,y in df.groupby(by="bins"):
    res.extend(y["Patient_id"].head(int(n_tot/n_bins)).sort_values().astype(int).astype(str))
    #print(x,"\n",y[["Patient_id","SNR","tot_displacement","spatially_weighted_displacement"]].head(5).sort_values(by=on))
res='["'+'","'.join(res)+'"]'

text_file = open("../res/ied.txt", "w")
 
#write string to file
text_file.write(res)
 
#close file
text_file.close()
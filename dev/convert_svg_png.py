from cairosvg import svg2png
import glob
from tqdm import tqdm
cohort_built_path = "/data/soin/octgwas/Results_cohortbuilder/enface_view/UKBB/"
all_paths = glob.glob(cohort_built_path + "/*/*/*/*.svg")
for p in tqdm(all_paths):
    svg2png(url=p,write_to=p.replace(".svg",".png"))

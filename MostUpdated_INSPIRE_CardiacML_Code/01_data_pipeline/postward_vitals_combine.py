import pandas as pd, glob
files = sorted(glob.glob('postop_ward_vitals_part_*.csv.gz'))
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
df.to_csv("postop_ward_vitals_combined.csv", index=False)
df.to_csv("postop_ward_vitals_combined.csv", index=False)
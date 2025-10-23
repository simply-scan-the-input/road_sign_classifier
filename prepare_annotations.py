import os
import pandas as pd
import glob

# paths
base_path = r"C:\Users\48668\OneDrive\Pulpit\DataEngeneering\SeventhSemester\DNNDA\Project\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images"
signnames_path = "signnames.csv"

# load class names
signnames = pd.read_csv(signnames_path)

# collect all annotation csv files
csv_files = glob.glob(os.path.join(base_path, "*/GT-*.csv"))

df_list = []
for csv_file in csv_files:
    tmp = pd.read_csv(csv_file, sep=';')
    folder = os.path.basename(os.path.dirname(csv_file))
    tmp['Folder'] = folder
    df_list.append(tmp)

# combine all folders
df = pd.concat(df_list, ignore_index=True)

# build file paths
df['FilePath'] = df['Folder'] + '/' + df['Filename']

# join with class names
df = df.merge(signnames, left_on='ClassId', right_on='ClassId', how='left')

# select only relevant columns
df = df[['FilePath', 'ClassId', 'SignName']]

# save
df.to_csv('annotations.csv', index=False)

print(f"✅ annotations.csv created! Total samples: {len(df)}")
print("Example rows:")
print(df.head())

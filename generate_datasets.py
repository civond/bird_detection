import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

data_dir = "Data/"
label_path = "Data/metadata.csv"
df = pd.read_csv(label_path)

df['item_dir'] = data_dir + df['itemid'].astype(str)+'.wav'

train, validate, test = np.split(df.sample(frac=1, random_state=42),
                                 [int(.8*len(df)), int(.9*len(df))])
temp = [train, validate, test]

for i, df_item in enumerate(temp):
    item = ["train", "validate", "test"][i]  # Get the corresponding item name
    temp_path = os.path.join(data_dir, item)
    
    # Check if directory exists. If not, create it.
    if os.path.isdir(temp_path) == False:
        os.mkdir(temp_path)
        print(f"\t{temp_path} not found. Creating...")
    else:
        print(f"\t{temp_path} already exists...")
    
    # Save DataFrame as CSV in the corresponding directory
    csv_filename = os.path.join(temp_path, f"{item}.csv")
    df_item.to_csv(csv_filename, index=False)
    print(f"\t{item} saved to {csv_filename}")
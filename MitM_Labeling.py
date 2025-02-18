import pandas as pd
import csv
from tqdm import tqdm  # Progress bar

# 1️. Setting file path
raw_data_file = "../Data_Resources/ARP_MitM_Kitsune/ARP_MitM_dataset.csv/ARP_MitM_dataset.csv"
label_file = "../Data_Resources/ARP_MitM_Kitsune/ARP_MitM_labels.csv/ARP_MitM_labels.csv"
output_file = "../Data_Resources/ARP_MitM_Kitsune/ARP_MitM_dataset.csv/ARP_MitM_dataset_final.csv"

# 2️. Generate headers (23 features × 5 time windows = 115)
time_windows = ["100ms", "500ms", "1.5s", "10s", "1min"]
features = [
    "SrcMAC_IP_w", "SrcMAC_IP_mu", "SrcMAC_IP_sigma", "SrcMAC_IP_max", "SrcMAC_IP_min",
    "SrcIP_w", "SrcIP_mu", "SrcIP_sigma", "SrcIP_max", "SrcIP_min",
    "Channel_w", "Channel_mu", "Channel_sigma", "Channel_max", "Channel_min",
    "Socket_w", "Socket_mu", "Socket_sigma", "Socket_max", "Socket_min",
    "Jitter_mu", "Jitter_sigma", "Jitter_max"
]
header = [f"{feature}_{window}" for window in time_windows for feature in features]

# 3️. Get raw data
df_data = pd.read_csv(raw_data_file, header=None) # (read without headers)

# Adding headers
df_data.columns = header

# 4️. Loading label data
df_labels = pd.read_csv(label_file)

# Clean up IDs and labels (IDs start at 1, so convert to a 0-based index)
df_labels.rename(columns={"Unnamed: 0": "ID", "x": "Label"}, inplace=True)
df_labels.index = df_labels["ID"] - 1  # Adjust to start at 0

# 5️. Adding labels to a dataset
df_data["Label"] = df_labels["Label"].values

# 6. Save a new CSV (progress bar is only output to the console)
num_rows = len(df_data)

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    # Writing headers
    writer.writerow(df_data.columns)

    # Build data (progress bar console output only)
    with tqdm(total=num_rows, desc="Saving CSV", unit="row") as pbar:
        for row in df_data.itertuples(index=False, name=None):
            writer.writerow(row)
            pbar.update(1)  # Update the progress bar

print(f"Final CSV file created with headers and labels added: {output_file}")

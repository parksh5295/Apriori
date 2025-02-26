import pandas as pd
import itertools
from collections import defaultdict

data_df = pd.read_csv("output-dataset_ESSlab.csv")

Anomal_df = data_df.loc[data_df.reconnaissance == 1 or data_df.infection == 1 or data_df.action == 1]

Anomal_df



# Data preprocessing: Convert to 'on' or 'off' except reconnaissance, infection, action
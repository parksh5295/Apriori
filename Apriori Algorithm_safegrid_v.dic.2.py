import pandas as pd
import itertools
from collections import defaultdict

data_df = pd.read_csv("output-dataset_ESSlab.csv")

Anomal_df = data_df.loc[data_df.reconnaissance == 1 or data_df.infection == 1 or data_df.action == 1]

Anomal_df



# 데이터 전처리: reconnaissance, infection, action을 제외하고 'on' 또는 'off'로 변환
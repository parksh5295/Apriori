import pandas as pd

# CSV 파일 읽기
data = pd.read_csv('../Data_Resources/ARP_MitM_Kitsune/ARP_MitM_dataset.csv/ARP_MitM_dataset_final.csv')

# 가장 위 row 한 줄만 출력
print(data.head(1))
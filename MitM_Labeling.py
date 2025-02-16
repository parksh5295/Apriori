import csv

with open("../Data_Resources/ARP_MitM_Kitsune/ARP_MitM_dataset.csv/ARP_MitM_dataset.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        if i >= 1:  # 3줄 읽고 종료
            break
        print(row)
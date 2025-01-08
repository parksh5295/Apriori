import pandas as pd

# 두 CSV 파일 읽기
related_groups_df1 = pd.read_csv('related_groups_anomal_1.csv', header=None)
related_groups_df2 = pd.read_csv('related_groups_nomal_1.csv', header=None)

# 차집합을 구할 리스트를 저장할 집합
difference_groups_set = set()

# 각 그룹을 비교하여 A - B 차집합 구하기
for index1, group1 in related_groups_df1.iterrows():
    group1_set = set(group1.dropna().tolist())  # 그룹 1을 집합으로 변환
    for index2, group2 in related_groups_df2.iterrows():
        group2_set = set(group2.dropna().tolist())  # 그룹 2를 집합으로 변환
        
        # A - B 차집합 구하기
        difference = group1_set - group2_set
        if difference:  # 차집합이 비어있지 않은 경우
            difference_groups_set.update(difference)  # 집합에 추가

# 결과를 리스트로 변환하고 DataFrame으로 변환
difference_groups_df = pd.DataFrame(list(difference_groups_set))

# 현재 Python 파일과 동일한 위치에 CSV 파일로 저장
difference_groups_df.to_csv('Difference_groups_anomal2nomal.csv', index=False, header=False)

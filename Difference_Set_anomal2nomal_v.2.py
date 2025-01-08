import pandas as pd

# 두 CSV 파일 읽기
related_groups_df1 = pd.read_csv('related_groups_anomal_2.csv', header=None)
related_groups_df2 = pd.read_csv('related_groups_nomal_2.csv', header=None)

# 차집합을 구할 리스트를 저장할 리스트
difference_groups = []

# 각 그룹을 비교하여 A - B 차집합 구하기
for index1, group1 in related_groups_df1.iterrows():
    group1_set = set(group1.dropna().tolist())  # 그룹 1을 집합으로 변환
    for index2, group2 in related_groups_df2.iterrows():
        group2_set = set(group2.dropna().tolist())  # 그룹 2를 집합으로 변환
        
        # A - B 차집합 구하기
        difference = group1_set - group2_set
        if difference:  # 차집합이 비어있지 않은 경우
            # 리스트 형태로 추가하기 전에 중복 체크
            difference_group = list(difference)
            if difference_group not in difference_groups:
                difference_groups.append(difference_group)  # 중복되지 않는 경우만 추가

# 결과를 DataFrame으로 변환
difference_groups_df = pd.DataFrame(difference_groups)

# 현재 Python 파일과 동일한 위치에 CSV 파일로 저장
difference_groups_df.to_csv('Difference_groups_anomal2nomal.csv', index=False, header=False)

# 각 리스트의 요소 개수별로 세기 위한 딕셔너리 초기화
count_by_size = {}

# 각 그룹의 요소 개수 세기
for group in difference_groups:
    size = len(group)  # 그룹의 요소 개수
    if size in count_by_size:
        count_by_size[size] += 1
    else:
        count_by_size[size] = 1

# 결과 출력
sorted_counts = sorted(count_by_size.items())
for size, count in sorted_counts:
    print(f"Element count {size}: {count} groups")
import pandas as pd

data = pd.read_csv('./output-dataset_ESSlab.csv')

# 'reconnaissance', 'infection', 'action' 열 중 하나라도 1인 경우
data['anomal'] = data[['reconnaissance', 'infection', 'action']].any(axis=1).astype(int)

# 'reconnaissance', 'infection', 'action' 열을 제외한 열 선택
cols_to_change = data.columns.difference(['reconnaissance', 'infection', 'action'])
# 0보다 큰 값을 1로 변경
data[cols_to_change] = data[cols_to_change].applymap(lambda x: 1 if x > 0 else 0)

# 'anomal' 값이 1인 행만 필터링
anormal_rows = data[data['anomal'] == 1]
# 'anomal'이 1인 열들 선택, 'reconnaissance', 'infection', 'action' 열 제외
anormal_columns = anormal_rows.columns.difference(['anomal', 'reconnaissance', 'infection', 'action'])

# 각 column에서 값이 1인 row의 인덱스 리스트 만들기
anormal_lists = {col: anormal_rows.index[anormal_rows[col] == 1].tolist() for col in anormal_columns}

# 퍼센트 기준 설정
percent = 70 / 100

# 관련성 있는 리스트를 저장할 리스트
related_groups = []
considered = set()  # 이미 고려된 열들을 저장할 집합

# 모든 열에 대해 반복
for a in list(anormal_lists.keys()):
    if a in considered:
        continue
    current_group = {a}  # 현재 그룹 초기화
    considered.add(a)  # 현재 열을 고려된 집합에 추가
    related_groups.append(current_group.copy())  # 초기 그룹 추가
    new_group_found = True
    
    while new_group_found:
        new_group_found = False
        for b in list(anormal_lists.keys()):
            if b not in current_group:
                # A와 B의 관련성 확인
                a_count = len(anormal_lists[a])
                b_count = len(anormal_lists[b])
                combined_count = len(set(anormal_lists[a]) & set(anormal_lists[b]))

                # a_count 또는 b_count가 0이 아닐 경우에만 조건 확인
                if a_count > 0 and b_count > 0:
                    if combined_count / a_count >= percent and combined_count / b_count >= percent:
                        current_group.add(b)
                        considered.add(b)  # 추가된 열을 고려된 집합에 추가
                        new_group_found = True
                        related_groups.append(current_group.copy())  # 중간 그룹 추가

# 결과 확인
print([list(group) for group in related_groups])

# 그룹의 요소 개수별로 세기 위한 딕셔너리 초기화
count_by_size = {}

# related_groups의 각 그룹에 대해 요소 개수 세기
for group in related_groups:
    size = len(group)  # 그룹의 요소 개수
    if size in count_by_size:
        count_by_size[size] += 1
    else:
        count_by_size[size] = 1

# 결과 출력
sorted_counts = sorted(count_by_size.items())

# 출력
for size, count in sorted_counts:
    print(f"Element count {size}: {count} groups")

# related_groups의 각 그룹을 DataFrame으로 변환
related_groups_df = pd.DataFrame(related_groups)

# 현재 Python 파일과 동일한 위치에 CSV 파일로 저장
related_groups_df.to_csv('related_groups_anomal_2.csv', index=False, header=False)
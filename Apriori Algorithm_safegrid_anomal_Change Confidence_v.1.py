import pandas as pd
import numpy as np

# Step 1: Load the dataset and create the 'anomal' column
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

# 초기 confidence 설정
confidence = 0.7  # 70%
percent = confidence

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

# Step 2: Group size count
count_by_size = {}
for group in related_groups:
    size = len(group)  # 그룹의 요소 개수
    if size in count_by_size:
        count_by_size[size] += 1
    else:
        count_by_size[size] = 1

# 결과 출력
sorted_counts = sorted(count_by_size.items())
for size, count in sorted_counts:
    print(f"Element count {size}: {count} groups")

# related_groups의 각 그룹을 DataFrame으로 변환
related_groups_df = pd.DataFrame(related_groups)

# Step 3: Evaluation and confidence adjustment
# Load the evaluation parameter
eva_parameter = pd.read_csv('./related_groups_anomal_2.csv', header=None)

# Create binary values and 'anomal' column in test_file
test_file = pd.read_csv('./output-dataset_ESSlab.csv')
test_file_binary = test_file.applymap(lambda x: 1 if isinstance(x, (int, float)) and x > 0 else 0)
anomal_columns = ['reconnaissance', 'infection', 'action']
anomal_index = [test_file.columns.get_loc(col) for col in anomal_columns if col in test_file.columns]
test_file['anomal'] = test_file_binary.iloc[:, anomal_index].any(axis=1).astype(int)

# Create lists of positive columns
test_lists = []
for index, row in test_file_binary.iterrows():
    positive_columns = row[row == 1].index.tolist()
    test_lists.append(positive_columns)

# Calculate confusion matrix
def calculate_confusion_matrix(eva_row, test_list, anomal):
    is_subset = set(eva_row).issubset(set(test_list))
    TP, FP, TN, FN = 0, 0, 0, 0
    
    if anomal == 1:  # anomal = 1 case
        if is_subset:
            TP = 1
        else:
            FP = 1
    else:  # anomal = 0 case
        if is_subset:
            FN = 1
        else:
            TN = 1
            
    return TP, FP, TN, FN

# Store results
confusion_matrices = []
for index, eva_row in enumerate(eva_parameter.values):
    eva_list = eva_row[~pd.isnull(eva_row)].tolist()
    for test_index, test_list in enumerate(test_lists):
        anomal_value = test_file['anomal'].iloc[test_index]
        confusion_result = calculate_confusion_matrix(eva_list, test_list, anomal_value)
        confusion_matrices.append(confusion_result)

# Convert results to DataFrame
confusion_matrix_df = pd.DataFrame(confusion_matrices, columns=['TP', 'FP', 'TN', 'FN'])
confusion_matrix_df['eva_row'] = eva_parameter.index.repeat(len(test_lists))
confusion_matrix_df['test_row'] = [test_index for test_index in range(len(test_lists)) for _ in range(len(eva_parameter))]

# Summary of TP, TN, FP, FN for each eva_row
summary = confusion_matrix_df.groupby('eva_row').sum().reset_index()

# Function to calculate metrics
def calculate_metrics(row):
    TP = row['TP']
    FP = row['FP']
    TN = row['TN']
    FN = row['FN']
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    return pd.Series({
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    })

# Calculate metrics
metrics = summary.apply(calculate_metrics, axis=1)
summary = pd.concat([summary, metrics], axis=1)

# Weight definition
weights = {
    'Accuracy': 0.45,
    'Precision': 0.35,
    'Recall': 0.15,
    'F1-Score': 0.05
}

# Calculate total score
summary['Total_Score'] = summary.apply(lambda row: (row['Accuracy'] * weights['Accuracy'] +
                                                     row['Precision'] * weights['Precision'] +
                                                     row['Recall'] * weights['Recall'] +
                                                     row['F1-Score'] * weights['F1-Score']), axis=1)

# Function to adjust confidence based on total score
def adjust_confidence(total_score, initial_confidence=0.7):
    confidence = initial_confidence
    adjustments = []
    score_range = 10.0  # Starting with 10% adjustments

    while score_range >= 0.000001:
        for adjustment in np.arange(-score_range, score_range + 0.01, 0.01):  # 0.01 단위로 조정
            adjusted_confidence = confidence + adjustment / 100
            adjustments.append((adjusted_confidence, total_score))

        score_range /= 10  # Move to finer adjustments
    
    return adjustments

# Collect adjustments
all_adjustments = []
for index, row in summary.iterrows():
    total_score = row['Total_Score']
    adjustments = adjust_confidence(total_score, confidence)
    all_adjustments.extend(adjustments)

# Convert to DataFrame and save final results only
adjustments_df = pd.DataFrame(all_adjustments, columns=['Adjusted_Confidence', 'Total_Score'])
adjustments_df.to_csv('safegrid_confidence_adjustments_1.csv', index=False)

# Output the results
print(adjustments_df)

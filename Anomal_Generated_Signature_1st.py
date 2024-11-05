import pandas as pd

# 파일 로딩
data = pd.read_csv('./output-dataset_ESSLab.csv')
examples = pd.read_csv('./related_groups_anomal_c0.9.csv')

# 'reconnaissance', 'infection', 'action' 열 중 하나라도 1인 경우 anomal 컬럼에 1을 할당
data['anomal'] = data[['reconnaissance', 'infection', 'action']].any(axis=1).astype(int)

# 'reconnaissance', 'infection', 'action' 열을 제외한 나머지 컬럼 선택
cols_to_change = data.columns.difference(['reconnaissance', 'infection', 'action'])

# 0보다 큰 값을 1로 변경
data[cols_to_change] = data[cols_to_change].applymap(lambda x: 1 if x > 0 else 0)

# Anomalous 데이터만 필터링 (anomal == 1)
anomalous_data = data[data['anomal'] == 1]

# `examples`에서 24~32번째 열만 선택하여 checkpoint로 사용
checkpoint_columns = examples.columns[23:32]  # 24번째부터 32번째까지 (0-based indexing)
checkpoint_examples = examples[checkpoint_columns]

# checkpoint 예시 출력 (디버깅용)
print("Checkpoint columns:")
print(checkpoint_examples.head())

# Confusion Matrix 계산 함수
def calculate_confusion_matrix(eva_list, test_list, anomal):
    is_subset = set(test_list).issubset(set(eva_list))
    TP, FP, TN, FN = 0, 0, 0, 0
    
    if anomal == 1:  # anomal = 1인 경우
        if is_subset:
            TP = 1
        else:
            FP = 1
    else:  # anomal = 0인 경우
        if is_subset:
            FN = 1
        else:
            TN = 1
            
    return TP, FP, TN, FN

# Confusion Matrix 저장
confusion_matrices = []

# Anomalous 데이터에 대해 각 행과 관련된 특성을 비교하여 Confusion Matrix 계산
for index, row in anomalous_data.iterrows():
    eva_list = row[cols_to_change][row[cols_to_change] == 1].index.tolist()  # 1인 특성들
    for example_idx, example_row in checkpoint_examples.iterrows():
        # checkpoint 예시에서 관련된 특성들을 리스트로 변환
        test_list = example_row.dropna().astype(str).tolist()  # nan 값 제거하고 리스트로 변환
        anomal_value = 1  # Anomal = 1
        confusion_result = calculate_confusion_matrix(eva_list, test_list, anomal_value)
        confusion_matrices.append(confusion_result)

# 결과 DataFrame 생성
confusion_matrix_df = pd.DataFrame(confusion_matrices, columns=['TP', 'FP', 'TN', 'FN'])

# 총합 계산
summary = confusion_matrix_df.sum(axis=0)

# Precision, Recall 계산
precision = summary['TP'] / (summary['TP'] + summary['FP']) if (summary['TP'] + summary['FP']) > 0 else 0
recall = summary['TP'] / (summary['TP'] + summary['FN']) if (summary['TP'] + summary['FN']) > 0 else 0

# 결과 출력
print("Confusion Matrix Summary:")
print(summary)
print(f"Precision: {precision}")
print(f"Recall: {recall}")

confusion_matrix_df.to_csv('Result_1st_Signature_anomal_c0.9.csv', index=False)

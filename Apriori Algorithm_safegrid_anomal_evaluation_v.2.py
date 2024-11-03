import pandas as pd

# 데이터 로드
test_file = pd.read_csv('./output-dataset_ESSlab.csv')
eva_parameter = pd.read_csv('./related_groups_anomal_2.csv', header=None)

# test_file에서 0보다 큰 값을 1로 변환하고 anomal column 생성
test_file_binary = test_file.applymap(lambda x: 1 if isinstance(x, (int, float)) and x > 0 else 0)

# anomal column 추가
anomal_columns = ['reconnaissance', 'infection', 'action']  # test_file의 column 이름으로 맞춰야 함
anomal_index = [test_file.columns.get_loc(col) for col in anomal_columns if col in test_file.columns]
test_file['anomal'] = test_file_binary.iloc[:, anomal_index].any(axis=1).astype(int)

# test_file의 각 row에서 1인 column 이름 리스트 만들기
test_lists = []
for index, row in test_file_binary.iterrows():
    positive_columns = row[row == 1].index.tolist()
    test_lists.append(positive_columns)

# Confusion Matrix 계산
def calculate_confusion_matrix(eva_row, test_list, anomal):
    is_subset = set(eva_row).issubset(set(test_list))
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

# 결과 저장을 위한 리스트
confusion_matrices = []

# 각 eva_parameter의 row에 대해
for index, eva_row in enumerate(eva_parameter.values):
    eva_list = eva_row[~pd.isnull(eva_row)].tolist()  # NaN 값을 제외한 리스트
    for test_index, test_list in enumerate(test_lists):
        anomal_value = test_file['anomal'].iloc[test_index]  # 현재 row의 anomal 값
        confusion_result = calculate_confusion_matrix(eva_list, test_list, anomal_value)
        
        '''
        # eva_row, test_row 인덱스 출력
        print(f"eva_row: {index}, test_row: {test_index}, result: {confusion_result}")
        '''
        
        confusion_matrices.append(confusion_result)

# 결과 DataFrame으로 변환
confusion_matrix_df = pd.DataFrame(confusion_matrices, columns=['TP', 'FP', 'TN', 'FN'])
confusion_matrix_df['eva_row'] = eva_parameter.index.repeat(len(test_lists))
confusion_matrix_df['test_row'] = [test_index for test_index in range(len(test_lists)) for _ in range(len(eva_parameter))]

# 각 eva_row별 TP, TN, FP, FN 총합 계산 및 출력
summary = confusion_matrix_df.groupby('eva_row').sum().reset_index()
summary = summary[['eva_row', 'TP', 'FP', 'TN', 'FN']]  # test_row 제외
print("\nSummary of TP, TN, FP, FN for each eva_row:")
print(summary)

# 결과를 CSV 파일로 저장
confusion_matrix_df.to_csv('confusion_matrix_results_anomal_2.csv', index=False)
summary.to_csv('confusion_matrix_summary_anomal.csv', index=False)

# 결과 출력
print("Confusion matrix saved to 'confusion_matrix_results_anomal_2.csv'")

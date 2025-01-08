import pandas as pd


test_file = pd.read_csv('./output-dataset_ESSlab.csv')
eva_parameter = pd.read_csv('./related_groups_anomal_2.csv', header=None)

# 2. test_file에서 0보다 큰 값을 1로 변환하고 anomal column 생성
test_file_binary = test_file.applymap(lambda x: 1 if isinstance(x, (int, float)) and x > 0 else 0)

# 'reconnaissance', 'infection', 'action' 열 중 하나라도 1인 경우 anomal = 1
anomal_columns = ['reconnaissance', 'infection', 'action']  # 이 부분은 test_file의 column 이름으로 맞춰야 함
anomal_index = [test_file.columns.get_loc(col) for col in anomal_columns if col in test_file.columns]
test_file['anomal'] = test_file_binary.iloc[:, anomal_index].any(axis=1).astype(int)

# 3. test_file의 각 row에서 1인 column 이름 리스트 만들기
test_lists = []
for index, row in test_file_binary.iterrows():
    positive_columns = row[row == 1].index.tolist()
    test_lists.append(positive_columns)

# 4. Confusion Matrix 계산
def calculate_confusion_matrix(eva_row, test_list, anomal):
    is_subset = set(eva_row).issubset(set(test_list))  # Eva가 Test에 subset인지 확인
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

# eva_parameter의 각 row에 대해
for index, eva_row in enumerate(eva_parameter.values):
    eva_list = eva_row[~pd.isnull(eva_row)].tolist()  # NaN 값을 제외한 리스트
    for test_index, test_list in enumerate(test_lists):
        anomal_value = test_file['anomal'].iloc[test_index]  # 현재 row의 anomal 값
        confusion_matrices.append(calculate_confusion_matrix(eva_list, test_list, anomal_value))

# 5. Confusion Matrix 결과 DataFrame으로 저장
confusion_matrix_df = pd.DataFrame(confusion_matrices, columns=['TP', 'FP', 'TN', 'FN'])
confusion_matrix_df['eva_row'] = eva_parameter.index.repeat(len(test_lists))  # 각 row에 대해 eva_parameter의 인덱스 추가
confusion_matrix_df['test_row'] = [test_index for test_index in range(len(test_lists)) for _ in range(len(eva_parameter))]  # 각 row에 대해 test_file의 인덱스 추가

# 결과를 CSV 파일로 저장
confusion_matrix_df.to_csv('confusion_matrix_results_anomal.csv', index=False)

# 결과 출력
print("Confusion matrix saved to 'confusion_matrix_results_anomal.csv'")


'''
# Confusion Matrix 계산
for index, eva_row in enumerate(eva_parameter.values):
    eva_list = eva_row[~pd.isnull(eva_row)].tolist()  # NaN 값을 제외한 리스트
    for test_index, test_list in enumerate(test_lists):
        anomal_value = test_file['anomal'].iloc[test_index]  # 현재 row의 anomal 값
        confusion_result = calculate_confusion_matrix(eva_list, test_list, anomal_value)
        
        # 디버깅 출력
        print(f"eva_row: {index}, test_row: {test_index}, result: {confusion_result}")
        
        confusion_matrices.append(confusion_result)
'''
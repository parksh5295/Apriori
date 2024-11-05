import pandas as pd

# 1. CSV 파일 읽어오기
file_path = './anomal_evaluation_summary_metrics_c0.8.csv'
df = pd.read_csv(file_path)

# 2. Accuracy 열을 기준으로 내림차순으로 정렬
df_sorted = df.sort_values(by='Precision', ascending=False)

# 3. Accuracy가 가장 높은 상위 5개의 행 선택
top_5 = df_sorted.head(5)

# 4. Precision 열의 평균값 계산
precision_mean = top_5['Precision'].mean()

# 5. Precision 평균값을 새로운 행으로 추가
precision_mean_row = pd.DataFrame({'eva_row': ['Precision Mean'], 
                                   'TP': [None], 
                                   'FP': [None], 
                                   'TN': [None], 
                                   'FN': [None], 
                                   'Accuracy': [None], 
                                   'Precision': [precision_mean], 
                                   'Recall': [None], 
                                   'F1-Score': [None], 
                                   'TNR': [None], 
                                   'FPR': [None]})

# top_5 DataFrame과 Precision 평균값 행을 합침
top_5_with_mean = pd.concat([top_5, precision_mean_row], ignore_index=True)

# 6. 새로운 CSV 파일에 저장
output_file_path = 'top_5_precision_anomal_c0.8.csv'
top_5_with_mean.to_csv(output_file_path, index=False)

# 7. 결과 출력
print("상위 5개 Accuracy 점수 및 Precision 평균값:")
print(top_5_with_mean)
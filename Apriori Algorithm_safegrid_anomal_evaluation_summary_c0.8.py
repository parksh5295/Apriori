import pandas as pd

# summary CSV 파일 불러오기
summary = pd.read_csv('./confusion_matrix_summary_anomal_c0.8.csv')

# 지표 계산 함수
def calculate_metrics(row):
    TP = row['TP']
    FP = row['FP']
    TN = row['TN']
    FN = row['FN']
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0
    fpr = FP / (TN + FP) if (TN + FP) > 0 else 0
    
    return pd.Series({
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'TNR': tnr,
        'FPR': fpr
    })

# 각 eva_row에 대해 지표 계산
metrics = summary.apply(calculate_metrics, axis=1)

# 결과를 summary DataFrame에 추가
summary = pd.concat([summary, metrics], axis=1)

# 결과를 CSV 파일로 저장
summary.to_csv('anomal_evaluation_summary_metrics_c0.8.csv', index=False)

# 결과 출력
print(summary)

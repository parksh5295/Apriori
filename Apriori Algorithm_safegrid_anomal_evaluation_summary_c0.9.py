import pandas as pd

# Loading summary CSV file
summary = pd.read_csv('./confusion_matrix_summary_anomal_c0.9.csv')

# Indicator calculation function
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

# Calculate metrics for each eva_row
metrics = summary.apply(calculate_metrics, axis=1)

# Add results to summary DataFrame
summary = pd.concat([summary, metrics], axis=1)

# Save results as CSV file
summary.to_csv('anomal_evaluation_summary_metrics_c0.9.csv', index=False)

# Result output
print(summary)

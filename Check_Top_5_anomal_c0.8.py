import pandas as pd

# 1. Read CSV file
file_path = './anomal_evaluation_summary_metrics_c0.8.csv'
df = pd.read_csv(file_path)

# 2. Sort in descending order by the Accuracy column
df_sorted = df.sort_values(by='Precision', ascending=False)

# 3. Select top 5 rows with highest Accuracy
top_5 = df_sorted.head(5)

# 4. Calculate the average value of the Precision column
precision_mean = top_5['Precision'].mean()

# 5. Add precision average value as new row
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

# Combine top_5 DataFrame and Precision average rows
top_5_with_mean = pd.concat([top_5, precision_mean_row], ignore_index=True)

# 6. Save to new CSV file
output_file_path = 'top_5_precision_anomal_c0.8.csv'
top_5_with_mean.to_csv(output_file_path, index=False)

# 7. Result output
print("Top 5 Accuracy Scores and Precision Averages:")
print(top_5_with_mean)
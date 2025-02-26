import pandas as pd

# Load Data
test_file = pd.read_csv('./output-dataset_ESSlab.csv')
eva_parameter = pd.read_csv('./related_groups_anomal_c0.9.csv', header=None)

# Convert values ​​greater than 0 to 1 in test_file and create an anomal column
test_file_binary = test_file.applymap(lambda x: 1 if isinstance(x, (int, float)) and x > 0 else 0)

# Adding anomal column
anomal_columns = ['reconnaissance', 'infection', 'action']  # Must match the column name of test_file
anomal_index = [test_file.columns.get_loc(col) for col in anomal_columns if col in test_file.columns]
test_file['anomal'] = test_file_binary.iloc[:, anomal_index].any(axis=1).astype(int)

# Create a list of column names that are 1 in each row of test_file
test_lists = []
for index, row in test_file_binary.iterrows():
    positive_columns = row[row == 1].index.tolist()
    test_lists.append(positive_columns)

# Confusion Matrix calculation
def calculate_confusion_matrix(eva_row, test_list, anomal):
    is_subset = set(eva_row).issubset(set(test_list))
    TP, FP, TN, FN = 0, 0, 0, 0
    
    if anomal == 1:  # If anomal = 1
        if is_subset:
            TP = 1
        else:
            FP = 1
    else:  # If anomal = 0
        if is_subset:
            FN = 1
        else:
            TN = 1
            
    return TP, FP, TN, FN

# List for saving results
confusion_matrices = []

# For each row of eva_parameter
for index, eva_row in enumerate(eva_parameter.values):
    eva_list = eva_row[~pd.isnull(eva_row)].tolist()  # List excluding NaN values
    for test_index, test_list in enumerate(test_lists):
        anomal_value = test_file['anomal'].iloc[test_index]  # anomal value of current row
        confusion_result = calculate_confusion_matrix(eva_list, test_list, anomal_value)
        
        '''
        # eva_row, test_row index output
        print(f"eva_row: {index}, test_row: {test_index}, result: {confusion_result}")
        '''
        
        confusion_matrices.append(confusion_result)

# Convert to resulting DataFrame
confusion_matrix_df = pd.DataFrame(confusion_matrices, columns=['TP', 'FP', 'TN', 'FN'])
confusion_matrix_df['eva_row'] = eva_parameter.index.repeat(len(test_lists))
confusion_matrix_df['test_row'] = [test_index for test_index in range(len(test_lists)) for _ in range(len(eva_parameter))]

# Calculate and output TP, TN, FP, FN total for each eva_row
summary = confusion_matrix_df.groupby('eva_row').sum().reset_index()
summary = summary[['eva_row', 'TP', 'FP', 'TN', 'FN']]  # Excluding test_row
print("\nSummary of TP, TN, FP, FN for each eva_row:")
print(summary)

# Save results as CSV file
confusion_matrix_df.to_csv('confusion_matrix_results_anomal_c0.9.csv', index=False)
summary.to_csv('confusion_matrix_summary_anomal_c0.9.csv', index=False)

# Result output
print("Confusion matrix saved to 'confusion_matrix_results_anomal_c0.9.csv'")

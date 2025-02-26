import pandas as pd

# Loading files
data = pd.read_csv('./output-dataset_ESSLab.csv')
examples = pd.read_csv('./related_groups_anomal_c0.9.csv')

# Assign 1 to the anomal column if any of the 'reconnaissance', 'infection', or 'action' columns are equal to 1
data['anomal'] = data[['reconnaissance', 'infection', 'action']].any(axis=1).astype(int)

# Select all columns except the 'reconnaissance', 'infection', and 'action' columns
cols_to_change = data.columns.difference(['reconnaissance', 'infection', 'action'])

# Change values greater than 0 to 1
data[cols_to_change] = data[cols_to_change].applymap(lambda x: 1 if x > 0 else 0)

# Filter only anomalous data (anomal == 1)
anomalous_data = data[data['anomal'] == 1]

# Select only columns 24-32 from `examples` to use as a checkpoint
checkpoint_columns = examples.columns[23:32]  # 24th through 32nd (0-based indexing)
checkpoint_examples = examples[checkpoint_columns]

# Example checkpoint output (for debugging)
print("Checkpoint columns:")
print(checkpoint_examples.head())

# Confusion Matrix Calculation Functions
def calculate_confusion_matrix(eva_list, test_list, anomal):
    is_subset = set(test_list).issubset(set(eva_list))
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

# Save Confusion Matrix
confusion_matrices = []

# Calculate a Confusion Matrix by comparing the attributes associated with each row for anomalous data
for index, row in anomalous_data.iterrows():
    eva_list = row[cols_to_change][row[cols_to_change] == 1].index.tolist()  # Features with a value of 1
    for example_idx, example_row in checkpoint_examples.iterrows():
        # Convert the relevant attributes from the checkpoint example to a list
        test_list = example_row.dropna().astype(str).tolist()  # Remove nan values and convert to a list
        anomal_value = 1  # Anomal = 1
        confusion_result = calculate_confusion_matrix(eva_list, test_list, anomal_value)
        confusion_matrices.append(confusion_result)

# Generate the resulting DataFrame
confusion_matrix_df = pd.DataFrame(confusion_matrices, columns=['TP', 'FP', 'TN', 'FN'])

# Calculate totals
summary = confusion_matrix_df.sum(axis=0)

# Calculate Precision, Recall
precision = summary['TP'] / (summary['TP'] + summary['FP']) if (summary['TP'] + summary['FP']) > 0 else 0
recall = summary['TP'] / (summary['TP'] + summary['FN']) if (summary['TP'] + summary['FN']) > 0 else 0

# Output the results
print("Confusion Matrix Summary:")
print(summary)
print(f"Precision: {precision}")
print(f"Recall: {recall}")

confusion_matrix_df.to_csv('Result_1st_Signature_anomal_c0.9.csv', index=False)

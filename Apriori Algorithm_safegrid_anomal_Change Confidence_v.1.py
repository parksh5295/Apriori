import pandas as pd
import numpy as np

# Step 1: Load the dataset and create the 'anomal' column
data = pd.read_csv('./output-dataset_ESSlab.csv')

# If any of the columns 'reconnaissance', 'infection', or 'action' are equal to 1
data['anomal'] = data[['reconnaissance', 'infection', 'action']].any(axis=1).astype(int)

# Select columns except for the 'reconnaissance', 'infection', and 'action' columns
cols_to_change = data.columns.difference(['reconnaissance', 'infection', 'action'])
# Change values greater than 0 to 1
data[cols_to_change] = data[cols_to_change].applymap(lambda x: 1 if x > 0 else 0)

# Filter only rows with a value of 1 for 'anomal'
anormal_rows = data[data['anomal'] == 1]
# Select columns with 'anomal' equal to 1, excluding columns 'reconnaissance', 'infection', and 'action'
anormal_columns = anormal_rows.columns.difference(['anomal', 'reconnaissance', 'infection', 'action'])

# Create an indexed list of rows with a value of 1 in each column
anormal_lists = {col: anormal_rows.index[anormal_rows[col] == 1].tolist() for col in anormal_columns}

# Initial confidence settings
confidence = 0.7  # 70%
percent = confidence

# Lists to store relevant lists
related_groups = []
considered = set()  # A set to store columns that have already been considered

# Repeat for all columns
for a in list(anormal_lists.keys()):
    if a in considered:
        continue
    current_group = {a}  # Reset the current group
    considered.add(a)  # Add the current column to the considered set
    related_groups.append(current_group.copy())  # Add an initial group
    new_group_found = True
    
    while new_group_found:
        new_group_found = False
        for b in list(anormal_lists.keys()):
            if b not in current_group:
                # Check the relevance of A and B
                a_count = len(anormal_lists[a])
                b_count = len(anormal_lists[b])
                combined_count = len(set(anormal_lists[a]) & set(anormal_lists[b]))

                # Check condition only if a_count or b_count is non-zero
                if a_count > 0 and b_count > 0:
                    if combined_count / a_count >= percent and combined_count / b_count >= percent:
                        current_group.add(b)
                        considered.add(b)  # Add the added column to the considered set
                        new_group_found = True
                        related_groups.append(current_group.copy())  # Add an intermediate group

# Check the results
print([list(group) for group in related_groups])

# Step 2: Group size count
count_by_size = {}
for group in related_groups:
    size = len(group)  # Number of elements in the group
    if size in count_by_size:
        count_by_size[size] += 1
    else:
        count_by_size[size] = 1

# Output the results
sorted_counts = sorted(count_by_size.items())
for size, count in sorted_counts:
    print(f"Element count {size}: {count} groups")

# Convert each group in related_groups to a DataFrame
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

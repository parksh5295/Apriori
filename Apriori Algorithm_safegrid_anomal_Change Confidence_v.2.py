import pandas as pd
import numpy as np

# Total score calculation function
def calculate_total_score(TP, TN, FP, FN, weights):
    metrics = {}
    metrics['Accuracy'] = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    metrics['Precision'] = TP / (TP + FP) if (TP + FP) > 0 else 0
    metrics['Recall'] = TP / (TP + FN) if (TP + FN) > 0 else 0
    metrics['F1-Score'] = (2 * metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall']) \
                          if (metrics['Precision'] + metrics['Recall']) > 0 else 0
    total_score = sum(metrics[key] * weights[key] for key in weights)
    return total_score, metrics

# Confusion matrix calculation functions
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

# Apriori execution functions (traditional Apriori algorithm logic)
def run_apriori_algorithm(data, confidence):
    data['anomal'] = data[['reconnaissance', 'infection', 'action']].any(axis=1).astype(int)
    cols_to_change = data.columns.difference(['reconnaissance', 'infection', 'action'])
    data[cols_to_change] = data[cols_to_change].applymap(lambda x: 1 if x > 0 else 0)
    anormal_rows = data[data['anomal'] == 1]
    anormal_columns = anormal_rows.columns.difference(['anomal', 'reconnaissance', 'infection', 'action'])
    anormal_lists = {col: anormal_rows.index[anormal_rows[col] == 1].tolist() for col in anormal_columns}
    
    percent = confidence  # Using confidence as a ratio
    related_groups = []
    considered = set()
    for a in list(anormal_lists.keys()):
        if a in considered:
            continue
        current_group = {a}
        considered.add(a)
        related_groups.append(current_group.copy())
        new_group_found = True
        while new_group_found:
            new_group_found = False
            for b in list(anormal_lists.keys()):
                if b not in current_group:
                    a_count = len(anormal_lists[a])
                    b_count = len(anormal_lists[b])
                    combined_count = len(set(anormal_lists[a]) & set(anormal_lists[b]))
                    if a_count > 0 and b_count > 0:
                        if combined_count / a_count >= percent and combined_count / b_count >= percent:
                            current_group.add(b)
                            considered.add(b)
                            new_group_found = True
                            related_groups.append(current_group.copy())
    return [list(group) for group in related_groups]

# Main process
def main():
    # Load Data
    data = pd.read_csv('./output-dataset_ESSlab.csv')
    test_file = pd.read_csv('./output-dataset_ESSlab.csv')
    weights = {'Accuracy': 0.15, 'Precision': 0.65, 'Recall': 0.15, 'F1-Score': 0.05}
    confidence_scores = np.arange(0.1, 1.0, 0.05)
    results = []
    
    for confidence in confidence_scores:
        # Execute Apriori
        related_groups = run_apriori_algorithm(data.copy(), confidence)
        
        # Calculate Confusion Matrix
        test_file_binary = test_file.applymap(lambda x: 1 if isinstance(x, (int, float)) and x > 0 else 0)
        anomal_columns = ['reconnaissance', 'infection', 'action']
        anomal_index = [test_file.columns.get_loc(col) for col in anomal_columns if col in test_file.columns]
        test_file['anomal'] = test_file_binary.iloc[:, anomal_index].any(axis=1).astype(int)
        
        test_lists = []
        for _, row in test_file_binary.iterrows():
            positive_columns = row[row == 1].index.tolist()
            test_lists.append(positive_columns)
        
        confusion_matrices = []
        for eva_row in related_groups:
            for test_index, test_list in enumerate(test_lists):
                anomal_value = test_file['anomal'].iloc[test_index]
                confusion_matrices.append(calculate_confusion_matrix(eva_row, test_list, anomal_value))
        
        # Sum of Confusion matrix
        TP, FP, TN, FN = map(sum, zip(*confusion_matrices))
        
        # Calculate Total Score
        total_score, metrics = calculate_total_score(TP, TN, FP, FN, weights)
        results.append({'confidence': confidence, 'total_score': total_score, **metrics})
    
    # Choose the best confidence
    best_result = max(results, key=lambda x: x['total_score'])
    print(f"Best confidence: {best_result['confidence']}")
    print(f"Total Score: {best_result['total_score']}")
    print("Metrics:", {k: best_result[k] for k in weights.keys()})

# Execute
if __name__ == "__main__":
    main()

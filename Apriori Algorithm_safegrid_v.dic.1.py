import pandas as pd
import itertools
from collections import defaultdict

# Data preprocessing and application of Apriori algorithm
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Data preprocessing: Convert to 'on' or 'off' except reconnaissance, infection, action
def preprocess_data(data):
    transactions = []
    for index, row in data.iterrows():
        transaction = {item: 'on' if item > 0 else 'off' for item in row[1:]}  # Convert starting from the 1st column
        transactions.append(transaction)
    return transactions

# Support calculation function
def calculate_support(itemset, transactions):
    count = 0
    itemset = set(itemset)
    for transaction in transactions:
        if itemset.issubset(transaction.keys()):  # Check if itemset is in the transaction keys
            count += 1
    return count / len(transactions)

# Candidate Generation Function
def generate_candidates(itemsets, length):
    candidates = set()
    itemsets = list(itemsets)
    for item1 in itemsets:
        for item2 in itemsets:
            union = item1 | item2  # Use union operator for frozensets
            if len(union) == length:
                candidates.add(union)
    return candidates

# Apriori Algorithm
def apriori(transactions, min_support, min_confidence):
    # Initial itemset
    itemset = {frozenset([item]) for transaction in transactions for item in transaction.keys()}
    length = 1
    frequent_itemsets = set()

    while True:
        length += 1
        candidates = generate_candidates(itemset, length)
        frequent_itemsets_this_round = set()

        for candidate in candidates:
            support = calculate_support(candidate, transactions)
            if support >= min_support:
                frequent_itemsets_this_round.add(candidate)

        if not frequent_itemsets_this_round:
            break
        
        itemset = frequent_itemsets_this_round
        frequent_itemsets.update(itemset)

    # Create rules and calculate confidence
    rules = []
    for item in frequent_itemsets:
        for i in range(1, len(item)):
            for antecedent in itertools.combinations(item, i):
                antecedent = frozenset(antecedent)
                consequent = item.difference(antecedent)
                if len(consequent) > 0:
                    confidence = calculate_support(item, transactions) / calculate_support(antecedent, transactions)
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))
    return rules

# CSV file path settings
file_path = './output-dataset_ESSlab.csv'

# Data loading and preprocessing
data = load_data(file_path)
transactions = preprocess_data(data)

# Execute Apriori Algorithm
min_support = 0.1  # Minimum Support
min_confidence = 0.7  # Minimum confidence
rules = apriori(transactions, min_support, min_confidence)

# Result Output
print("\n== Found Rules ==")
for antecedent, consequent, confidence in rules:
    print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence}")
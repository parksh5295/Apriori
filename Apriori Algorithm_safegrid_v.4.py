import pandas as pd
import itertools

# Data preprocessing and application of Apriori algorithm
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Data preprocessing: Convert to 'on' or 'off' except reconnaissance, infection, action
def preprocess_data(data):
    transactions = []
    for index, row in data.iterrows():
        transaction = tuple(('on' if item > 0 else 'off' for item in row[1:]))  # Convert starting from the 1st column
        transactions.append(transaction)
    return tuple(transactions)

# Support calculation function
def calculate_support(itemset, transactions):
    count = 0
    itemset = set(itemset)
    for transaction in transactions:
        if itemset.issubset(transaction):
            count += 1
    return count / len(transactions)

# Candidate Generation Function
def generate_candidates(itemset, length):
    candidates = set()
    itemset = list(itemset)
    for item1 in itemset:
        for item2 in itemset:
            union = item1.union(item2)
            if len(union) == length:
                candidates.add(union)
    return tuple(candidates)

# Apriori Algorithm
def apriori(transactions, min_support, min_confidence):
    print("a")
    for trasaction in transactions:
        print("b")
        for item in trasaction:
            print("c")
            itemset = set(frozenset([item]))
            print(itemset)
            print("d")
            length = 1
            print("e")
            frequent_itemsets = set()
            print("f")

            while True:
                print("1")
                length += 1
                print("2")
                candidates = generate_candidates(itemset, length)
                print(candidates)
                print("3")
                frequent_itemsets_this_round = set()

                print("4")
                for candidate in candidates:
                    print("5")
                    support = calculate_support(candidate, transactions)
                    print("6")
                    if support >= min_support:
                        print("7")
                        frequent_itemsets_this_round.add(candidate)

                print("8")
                if len(frequent_itemsets_this_round) == 0:
                    break
        
                print("9")
                itemset = frequent_itemsets_this_round
                print("10")
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
    return tuple(rules)

# CSV file path settings
file_path = './output-dataset_ESSlab.csv'

# Data Loading and Preprocessing
data = load_data(file_path)
transactions = preprocess_data(data)

# Execute Apriori Algorithm
min_support = 0.004  # Minimum Support
min_confidence = 0.7  # Minimum confidence
rules = apriori(transactions, min_support, min_confidence)

# Result Output
print("\n== Found Rules ==")
for antecedent, consequent, confidence in rules:
    print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence}")
import csv
import pymysql
import itertools
import sys

# MySQL Database connection settings
def mysqlDbConnection(u, pw, h, p, d):
    try:
        conn = pymysql.connect(user=u, password=pw, host=h, port=p, database=d)
        print("DB Connection Success: {0}".format(h))
    except pymysql.Error as e:
        print("Error connecting to MySQL Platform : {}".format(e))
        sys.exit(1)
    return conn

# Data preprocessing and application of Apriori algorithm
dbConn = mysqlDbConnection('SafeGrid', 'safegrid001', 'localhost', 3306, 'safegrid_apriori_data')
cursor = dbConn.cursor()

# Data retrieval query
query = "SELECT * FROM DataDeck_Apriori"
cursor.execute(query)
data = cursor.fetchall()

# Connection terminated
cursor.close()
dbConn.close()

# Data preprocessing: Convert to 'on' or 'off' except reconnaissance, infection, action
transactions = []
for row in data:
    transaction = []
    for item in row[1:-3]:
        transaction.append((item, 'on' if item > 0 else 'off'))  # Excluding reconnaissance, infection, and action
    transactions.append(transaction)

# Support calculation function
def calculate_support(itemset, transactions):
    count = 0
    for transaction in transactions:
        if all(item in transaction for item in itemset):
            count += 1
    return count / len(transactions)

# Candidate Generation Function
def generate_candidates(itemset, length):
    return list(set([item1.union(item2) for item1 in itemset for item2 in itemset if len(item1.union(item2)) == length]))

# Apriori Algorithm
def apriori(transactions, min_support, min_confidence):
    itemset = [frozenset([item]) for transaction in transactions for item in transaction]
    length = 1
    while True:
        length += 1
        candidates = generate_candidates(itemset, length)
        frequent_itemset = []
        for item in candidates:
            support = calculate_support(item, transactions)
            if support >= min_support:
                frequent_itemset.append(item)
        if not frequent_itemset:
            break
        itemset = frequent_itemset

    # Create rules and calculate confidence
    rules = []
    for item in itemset:
        for i in range(1, len(item)):
            for antecedent in itertools.combinations(item, i):
                antecedent = frozenset(antecedent)
                consequent = item.difference(antecedent)
                if len(consequent) > 0:
                    confidence = calculate_support(item, transactions) / calculate_support(antecedent, transactions)
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))

    return rules

# Execute Apriori Algorithm
min_support = 0.1  # Minimum Support
min_confidence = 0.7  # Minimum confidence
rules = apriori(transactions, min_support, min_confidence)

# Result Output
print("\n== Found Rules ==")
for antecedent, consequent, confidence in rules:
    print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence}")
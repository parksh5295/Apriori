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
print("1")

# Connection terminated
cursor.close()
dbConn.close()

print("2")
# Data preprocessing: Convert to 'on' or 'off' except reconnaissance, infection, action
def preprocess_data(data):
    transactions = []
    for row in data:
        transaction = tuple((item, 'on' if item > 0 else 'off') for item in row[1:-3])
        transactions.append(transaction)
    return tuple(transactions)

transactions = preprocess_data(data)
print("3")

print("4")
# Support calculation function
def calculate_support(itemset, transactions):
    count = 0
    itemset = set(itemset)
    for transaction in transactions:
        if itemset.issubset(transaction):
            count += 1
    return count / len(transactions)
print("5")

print("6")
# Candidate Generation Function
def generate_candidates(itemset, length):
    print ("itemset: {}".format(itemset))
    print ("length: {}".format(length))
    print ("len(itemset): {}".format(len(itemset)))
    candidates = set()
    print ("g1")
    itemset = list(itemset)
    print ("g2")
    for item1 in itemset:
        print ("g3")
        for item2 in itemset:
            print ("g4")
            union = item1.union(item2)
            print ("g5")
            if len(union) == length:
                candidates.add(union)
            print ("g6")

            
    return tuple(candidates)
print("7")

print("8")

# Apriori Algorithm
def apriori(transactions, min_support, min_confidence):
    itemset = set(frozenset([item]) for transaction in transactions for item in transaction)
    length = 1
    frequent_itemsets = set()

    while True:
        length += 1
        print("a")
        candidates = generate_candidates(itemset, length)
        print("b")
        frequent_itemsets_this_round = set()
        
        print("c")
        for candidate in candidates:
            print("d")
            support = calculate_support(candidate, transactions)
            print("e")
            if support >= min_support:
                print("f")
                frequent_itemsets_this_round.add(candidate)
        
        print("g")
        if len(frequent_itemsets_this_round) == 0:
            break
        
        print("h")
        itemset = frequent_itemsets_this_round
        print("i")
        frequent_itemsets.update(itemset)

    print("9")

    # Create rules and calculate confidence
    rules = []
    for item in frequent_itemsets:
        for i in range(1, len(item)):
            list1 = itertools.combinations(item, i)
            for antecedent in list1:
                antecedent = frozenset(antecedent)
                consequent = item.difference(antecedent)
                if len(consequent) > 0:
                    confidence = calculate_support(item, transactions) / calculate_support(antecedent, transactions)
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))
    return tuple(rules)
print("10")

print("11")
# Execute Apriori Algorithm
min_support = 0.1  # Minimum Support
min_confidence = 0.7  # Minimum confidence
rules = apriori(transactions, min_support, min_confidence)
print("12")

# Result Output
print("\n== Found Rules ==")
for antecedent, consequent, confidence in rules:
    print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence}")

print("13")
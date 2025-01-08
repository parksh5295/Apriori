import pymysql
import itertools
import sys

def mysqlDbConnection(u, pw, h, p, d):
    try:
        conn = pymysql.connect(user=u, password=pw, host=h, port=p, database=d)
        print("DB Connection Success: {0}".format(h))
    except pymysql.Error as e:
        print("Error connecting to MySQL Platform : {}".format(e))
        sys.exit(1)
    return conn

def mysqlDbClose(_dbConn):
    try:
        _dbConn.close()
        print("DB Close Success")
    except pymysql.Error as e:
        print("Error closing from MySQL Platform")
        sys.exit(1)

# MySQL 연결 설정
dbConn = mysqlDbConnection('SafeGrid', 'safegrid001', 'localhost', 3306, 'sample')
cursor = dbConn.cursor()

# 데이터 불러오기 쿼리
query = "SELECT * FROM DataDeck_Apriori"
cursor.execute(query)
data = cursor.fetchall()

# 연결 종료
cursor.close()
mysqlDbClose(dbConn)

# 데이터 변환 및 전처리
transactions = []
for row in data:
    transaction = []
    for item in row:
        if item in ['reconnaissance', 'infection', 'action']:
            continue  # reconnaissance, infection, action은 제외
        transaction.append((item, 'on' if row[item] > 0 else 'off'))
    transactions.append(transaction)

# 지지도 계산 함수
def calculate_support(itemset, transactions):
    count = 0
    for transaction in transactions:
        if all(item in transaction for item in itemset):
            count += 1
    return count / len(transactions)

# 후보 생성 함수
def generate_candidates(itemset, length):
    return list(set([item1.union(item2) for item1 in itemset for item2 in itemset if len(item1.union(item2)) == length]))

# Apriori 알고리즘
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

    # 규칙 생성 및 confidence 계산
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

# Apriori 알고리즘 실행
min_support = 0.1  # 최소 지지도
min_confidence = 0.7  # 최소 confidence
rules = apriori(transactions, min_support, min_confidence)

# 결과 출력
for antecedent, consequent, confidence in rules:
    print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence}")
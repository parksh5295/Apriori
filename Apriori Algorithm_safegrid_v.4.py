import pandas as pd
import itertools

# 데이터 전처리 및 Apriori 알고리즘 적용
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# 데이터 전처리: reconnaissance, infection, action을 제외하고 'on' 또는 'off'로 변환
def preprocess_data(data):
    transactions = []
    for index, row in data.iterrows():
        transaction = tuple(('on' if item > 0 else 'off' for item in row[1:]))  # 1번째 열부터 변환
        transactions.append(transaction)
    return tuple(transactions)

# 지지도 계산 함수
def calculate_support(itemset, transactions):
    count = 0
    itemset = set(itemset)
    for transaction in transactions:
        if itemset.issubset(transaction):
            count += 1
    return count / len(transactions)

# 후보 생성 함수
def generate_candidates(itemset, length):
    candidates = set()
    itemset = list(itemset)
    for item1 in itemset:
        for item2 in itemset:
            union = item1.union(item2)
            if len(union) == length:
                candidates.add(union)
    return tuple(candidates)

# Apriori 알고리즘
def apriori(transactions, min_support, min_confidence):
    itemset = set(frozenset([item]) for transaction in transactions for item in transaction)
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

        if len(frequent_itemsets_this_round) == 0:
            break
        
        itemset = frequent_itemsets_this_round
        frequent_itemsets.update(itemset)

    # 규칙 생성 및 confidence 계산
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

# CSV 파일 경로 설정
file_path = './output-dataset_ESSlab.csv'

# 데이터 불러오기 및 전처리
data = load_data(file_path)
transactions = preprocess_data(data)

# Apriori 알고리즘 실행
min_support = 0.1  # 최소 지지도
min_confidence = 0.7  # 최소 confidence
rules = apriori(transactions, min_support, min_confidence)

# 결과 출력
print("\n== Found Rules ==")
for antecedent, consequent, confidence in rules:
    print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence}")
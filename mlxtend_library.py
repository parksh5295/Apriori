import pymysql
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# MySQL 연결 설정
connection = pymysql.connect(host='your_host',
                             user='your_username',
                             password='your_password',
                             db='your_database',
                             charset='utf8mb3',
                             cursorclass=pymysql.cursors.DictCursor)

# 데이터 불러오기 쿼리
query = "SELECT * FROM DataDeck_Apriori"

# 쿼리 실행 및 데이터 가져오기
with connection.cursor() as cursor:
    cursor.execute(query)
    data = cursor.fetchall()

# 연결 닫기
connection.close()

# 데이터 전처리: 데이터베이스에서 가져온 데이터를 DataFrame으로 변환
df = pd.DataFrame(data)

# 'reconnaissance', 'infection', 'action' 컬럼을 제외하고, 값에 따라 on 또는 off로 변환
columns_to_keep = [col for col in df.columns if col not in ['reconnaissance', 'infection', 'action']]
df[columns_to_keep] = df[columns_to_keep].applymap(lambda x: 'off' if x == 0 else 'on')

# Apriori 알고리즘 적용
frequent_itemsets = apriori(df[columns_to_keep], min_support=0.1, use_colnames=True)

# 연관 규칙 추출
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)

# 필터링: 'reconnaissance', 'infection', 'action' 관련 규칙 제외
filtered_rules = rules[~rules['antecedents'].apply(lambda x: any(item in x for item in ['reconnaissance', 'infection', 'action']))]

# 결과 출력
print(filtered_rules[['antecedents', 'consequents', 'support', 'confidence']])
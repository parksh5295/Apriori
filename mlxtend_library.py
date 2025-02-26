import pymysql
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# MySQL connection settings
connection = pymysql.connect(host='your_host',
                             user='your_username',
                             password='your_password',
                             db='your_database',
                             charset='utf8mb3',
                             cursorclass=pymysql.cursors.DictCursor)

# Data retrieval query
query = "SELECT * FROM DataDeck_Apriori"

# Run queries and get data
with connection.cursor() as cursor:
    cursor.execute(query)
    data = cursor.fetchall()

# Close connection
connection.close()

# Data Preprocessing: Convert data retrieved from database into DataFrame
df = pd.DataFrame(data)

# Switches to on or off depending on the value, except for the 'reconnaissance', 'infection', and 'action' columns.
columns_to_keep = [col for col in df.columns if col not in ['reconnaissance', 'infection', 'action']]
df[columns_to_keep] = df[columns_to_keep].applymap(lambda x: 'off' if x == 0 else 'on')

# Application of Apriori algorithm
frequent_itemsets = apriori(df[columns_to_keep], min_support=0.1, use_colnames=True)

# Association rule extraction
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)

# Filtering: Exclude rules related to 'reconnaissance', 'infection', and 'action'
filtered_rules = rules[~rules['antecedents'].apply(lambda x: any(item in x for item in ['reconnaissance', 'infection', 'action']))]

# Result Output
print(filtered_rules[['antecedents', 'consequents', 'support', 'confidence']])
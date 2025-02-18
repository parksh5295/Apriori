import json
import pandas as pd

# JSON 파일 읽기
data = []
#with open("./netML-Competition2020/0_test-challenge_set.json/0_test-challenge_set.json", "r") as file:
#with open("./netML-Competition2020/1_test-std_set.json/1_test-std_set.json", "r") as file:
with open("./netML-Competition2020/2_training_set.json/2_training_set.json", "r") as file:
    for line in file:
        try:
            # 각 줄을 JSON 객체로 변환
            json_obj = json.loads(line)
            data.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {line}")
            print(e)

# 데이터 확인
#print(data[:5])  # 첫 5개의 JSON 객체 출력

# data 리스트를 DataFrame으로 변환
df = pd.DataFrame(data)

# 데이터 확인
#print(df.head())

#df.to_csv("./netML-Competition2020/0_test-challenge_set.json/0_test-challenge_set.csv")
#df.to_csv("./netML-Competition2020/1_test-std_set.json/1_test-std_set.csv")
df.to_csv("./netML-Competition2020/2_training_set.json/2_training_set.csv")
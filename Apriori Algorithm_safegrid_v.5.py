import pandas as pd

data = pd.read_csv('./output-dataset_ESSlab.csv')

# If any of the 'reconnaissance', 'infection', or 'action' columns is 1
data['anomal'] = data[['reconnaissance', 'infection', 'action']].any(axis=1).astype(int)

# Select columns excluding 'reconnaissance', 'infection', and 'action' columns
cols_to_change = data.columns.difference(['reconnaissance', 'infection', 'action'])
# Change values ​​greater than 0 to 1
data[cols_to_change] = data[cols_to_change].applymap(lambda x: 1 if x > 0 else 0)

# Filter only rows with 'anormal' value of 1
anormal_rows = data[data['anomal'] == 1]
# Select columns where 'anomal' is 1
anormal_columns = anormal_rows.columns.difference(['anomal'])

# Create an index list of rows with value 1 in each column
anormal_lists = {col: anormal_rows.index[anormal_rows[col] == 1].tolist() for col in anormal_columns}

# Set percentage criteria
percent = 70 / 100

# List to store relevant lists
related_groups = []
considered = set()  # A set to store the columns that have already been considered

# Repeat for all columns
for a in list(anormal_lists.keys()):
    if a in considered:
        continue
    current_group = {a}  # Reset current group
    new_group_found = True
    
    while new_group_found:
        new_group_found = False
        for b in list(anormal_lists.keys()):
            if b not in current_group:
                # Check the relevance of A and B
                a_count = len(anormal_lists[a])
                b_count = len(anormal_lists[b])
                combined_count = len(set(anormal_lists[a]) & set(anormal_lists[b]))

                # Check condition only if a_count or b_count is non-zero
                if a_count > 0 and b_count > 0:
                    if combined_count / a_count >= percent and combined_count / b_count >= percent:
                        current_group.add(b)
                        considered.add(b)  # Add added columns to considered set
                        new_group_found = True

    related_groups.append(current_group)
    considered.add(a)  # Add current column to considered set

# Check results
print([list(group) for group in related_groups])
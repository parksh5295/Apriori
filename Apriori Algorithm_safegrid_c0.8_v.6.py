import pandas as pd

data = pd.read_csv('./output-dataset_ESSlab.csv')

# If any of the 'reconnaissance', 'infection', or 'action' columns is 1
data['anomal'] = data[['reconnaissance', 'infection', 'action']].any(axis=1).astype(int)

# Select columns excluding 'reconnaissance', 'infection', and 'action' columns
cols_to_change = data.columns.difference(['reconnaissance', 'infection', 'action'])
# Change values ​​greater than 0 to 1
data[cols_to_change] = data[cols_to_change].applymap(lambda x: 1 if x > 0 else 0)

# Filter only rows with 'anomal' value of 1
anormal_rows = data[data['anomal'] == 1]
# Select columns where 'anomal' is 1, exclude 'reconnaissance', 'infection', and 'action' columns
anormal_columns = anormal_rows.columns.difference(['anomal', 'reconnaissance', 'infection', 'action'])

# Create an index list of rows with value 1 in each column
anormal_lists = {col: anormal_rows.index[anormal_rows[col] == 1].tolist() for col in anormal_columns}

# Set percentage criteria
percent = 80 / 100

# List to store relevant lists
related_groups = []
considered = set()  # A set to store the columns that have already been considered

# Repeat for all columns
for a in list(anormal_lists.keys()):
    if a in considered:
        continue
    current_group = {a}  # Reset current group
    considered.add(a)  # Add current column to considered set
    related_groups.append(current_group.copy())  # Add initial group
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
                        related_groups.append(current_group.copy())  # Add middle group

# Check results
print([list(group) for group in related_groups])

# Initializing a dictionary to count by number of elements in a group
count_by_size = {}

# Count elements for each group in related_groups
for group in related_groups:
    size = len(group)  # Number of elements in group
    if size in count_by_size:
        count_by_size[size] += 1
    else:
        count_by_size[size] = 1

# Result output
sorted_counts = sorted(count_by_size.items())

# Output
for size, count in sorted_counts:
    print(f"Element count {size}: {count} groups")

# Convert each group in related_groups to DataFrame
related_groups_df = pd.DataFrame(related_groups)

# Save as CSV file in same location as current Python file
related_groups_df.to_csv('related_groups_anomal_c0.8.csv', index=False, header=False)
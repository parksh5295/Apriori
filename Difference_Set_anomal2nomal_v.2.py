import pandas as pd

# Reading two CSV files
related_groups_df1 = pd.read_csv('related_groups_anomal_2.csv', header=None)
related_groups_df2 = pd.read_csv('related_groups_nomal_2.csv', header=None)

# A set to store the list from which the difference is to be found
difference_groups = []

# Compare each group to find the difference between A and B
for index1, group1 in related_groups_df1.iterrows():
    group1_set = set(group1.dropna().tolist())  # Convert group 1 to set
    for index2, group2 in related_groups_df2.iterrows():
        group2_set = set(group2.dropna().tolist())  # Convert group 2 to set
        
        # Find the difference between A and B
        difference = group1_set - group2_set
        if difference:  # If the difference is not empty
            # Check for duplicates before adding them in list form
            difference_group = list(difference)
            if difference_group not in difference_groups:
                difference_groups.append(difference_group)  # Add only non-overlapping cases

# Convert results to list and convert to DataFrame
difference_groups_df = pd.DataFrame(difference_groups)

# Save as CSV file in same location as current Python file
difference_groups_df.to_csv('Difference_groups_anomal2nomal.csv', index=False, header=False)

# Initialize a dictionary to count the number of elements in each list
count_by_size = {}

# Count the number of elements in each group
for group in difference_groups:
    size = len(group)  # Number of elements in group
    if size in count_by_size:
        count_by_size[size] += 1
    else:
        count_by_size[size] = 1

# Result Output
sorted_counts = sorted(count_by_size.items())
for size, count in sorted_counts:
    print(f"Element count {size}: {count} groups")
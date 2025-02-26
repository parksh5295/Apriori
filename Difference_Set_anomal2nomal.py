import pandas as pd

# Reading two CSV files
related_groups_df1 = pd.read_csv('related_groups_anomal_1.csv', header=None)
related_groups_df2 = pd.read_csv('related_groups_nomal_1.csv', header=None)

# A set to store the list from which the difference is to be found
difference_groups_set = set()

# Compare each group to find the difference between A and B
for index1, group1 in related_groups_df1.iterrows():
    group1_set = set(group1.dropna().tolist())  # Convert group 1 to set
    for index2, group2 in related_groups_df2.iterrows():
        group2_set = set(group2.dropna().tolist())  # Convert group 2 to set
        
        # Find the difference between A and B
        difference = group1_set - group2_set
        if difference:  # If the difference is not empty
            difference_groups_set.update(difference)  # add to set

# Convert results to list and convert to DataFrame
difference_groups_df = pd.DataFrame(list(difference_groups_set))

# Save as CSV file in same location as current Python file
difference_groups_df.to_csv('Difference_groups_anomal2nomal.csv', index=False, header=False)

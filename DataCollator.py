
import pandas as pd

# Initialize empty lists to store dataframes
product_overview_dfs = []
other_ingredients_dfs = []

# List of filenames for product overview and other ingredients
product_overview_files = [
    'ProductOverview_1.csv', 'ProductOverview_2.csv', 'ProductOverview_3.csv', 
    'ProductOverview_4.csv', 'ProductOverview_5.csv', 'ProductOverview_6.csv', 'ProductOverview_7.csv'
]

other_ingredients_files = [
    'OtherIngredients_1.csv', 'OtherIngredients_2.csv', 'OtherIngredients_3.csv',
    'OtherIngredients_4.csv', 'OtherIngredients_5.csv', 'OtherIngredients_6.csv', 'OtherIngredients_7.csv'
]

# Read and store all ProductOverview CSVs
for file in product_overview_files:
    df = pd.read_csv(f'/Users/amystafford/Downloads/DSLD-full-database-CSV/{file}')
    product_overview_dfs.append(df)

# Read and store all OtherIngredients CSVs
for file in other_ingredients_files:
    df = pd.read_csv(f'/Users/amystafford/Downloads/DSLD-full-database-CSV/{file}')
    other_ingredients_dfs.append(df)

# Concatenate all product overview dataframes and reset index
product_overview_df = pd.concat(product_overview_dfs, ignore_index=True)
product_overview_df = product_overview_df[['DSLD ID', 'Serving Size', 'Product Type [LanguaL]','Supplement Form [LanguaL]']]

# Concatenate all other ingredients dataframes and reset index
other_ingredients_df = pd.concat(other_ingredients_dfs, ignore_index=True)
other_ingredients_df = other_ingredients_df[['DSLD ID', 'Other Ingredients']]

# Merge the datasets on DSLD ID
merged_df = pd.merge(product_overview_df, other_ingredients_df, on='DSLD ID', how='inner')

# Rename columns for clarity
merged_df.rename(columns={
    'Serving Size': 'Serving Size',
    'Product Type [LanguaL]': 'Product Type',
    'Other Ingredients': 'Other Ingredients',
     'Supplement Form [LanguaL]': 'Supplement Form'
}, inplace=True)

# Remove rows with missing values in any of the relevant columns
cleaned_df = merged_df.dropna(subset=['Serving Size', 'Product Type', 'Other Ingredients'])

# Save the cleaned dataframe to a CSV file
cleaned_df.to_csv('/Users/amystafford/Downloads/cleaned_dietary_supplements.csv', index=False)

# Display the cleaned dataframe
print(cleaned_df.head())


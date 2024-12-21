

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df = pd.read_csv('/Users/amystafford/Downloads/cleaned_dietary_supplements.csv')

# Check for missing values and drop rows with missing target values (Product Type)
df = df.dropna(subset=['Product Type', 'Other Ingredients', 'Supplement Form'])

# If necessary, you can remove unnecessary columns
# For simplicity, let's assume we keep the columns we need
df = df[['Product Type', 'Other Ingredients', 'Supplement Form']]

# Handle the text columns (Other Ingredients and Supplement Form)
# Encode the target column (Product Type)
label_encoder = LabelEncoder()
df['Product Type'] = label_encoder.fit_transform(df['Product Type'])

# Optional: Combine the text columns into one if needed 
df['Ingredients & Form'] = df['Other Ingredients'] + " " + df['Supplement Form']

# Step 2: Split the data into train and test sets
X = df['Ingredients & Form']  # Textual features
y = df['Product Type']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Vectorize the textual features using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Step 4: Train a classifier
model = Pipeline([
    ('tfidf', vectorizer),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
# 

#Data Viz
# 1. Distribution of Product Types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Product Type', order=df['Product Type'].value_counts().index)
plt.title('Distribution of Product Types')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.xlabel('Product Type')
plt.tight_layout()
plt.show()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# 3. Feature Importance from RandomForest
# Get feature importance from the trained RandomForest model
features = vectorizer.get_feature_names_out()
importances = model.named_steps['clf'].feature_importances_

# Sort the feature importances in descending order
sorted_idx = importances.argsort()[::-1]
top_n = 20  # Show the top 20 features
top_features = [features[i] for i in sorted_idx[:top_n]]
top_importances = importances[sorted_idx[:top_n]]

plt.figure(figsize=(10, 6))
sns.barplot(x=top_importances, y=top_features, palette='viridis')
plt.title('Top 20 Important Features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
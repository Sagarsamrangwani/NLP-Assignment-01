import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import wikipediaapi

# Function to get the data from Wikipedia
def get_wikipedia_content(title, lang='en'):
    user_agent = "Assignment_NLP/1.0 (sagarrangwani1@gmail.com) (https://github.com/Sagarsamrangwani)"  # Replace with your own user agent
    wiki_wiki = wikipediaapi.Wikipedia(lang, headers={'User-Agent': user_agent})
    page_py = wiki_wiki.page(title)

    if not page_py.exists():
        print(f"Page '{title}' does not exist.")
        return None

    return page_py.text

# Create dataset from Wikipedia
def create_wikipedia_dataset(medical_title, non_medical_title, num_samples=500):
    # Fetch Wikipedia
    medical_content = get_wikipedia_content(medical_title)
    non_medical_content = get_wikipedia_content(non_medical_title)

    # Create DataFrame from data
    data = pd.DataFrame({
        'text1': [medical_content] * num_samples + [non_medical_content] * num_samples,
        'label': ['medical'] * num_samples + ['non-medical'] * num_samples
    })

    # Change rows with None values
    data = data.dropna()

    return data

# Sample medical and non-medical Wikipedia articles
medical_title = "Medical_care"
non_medical_title = "History_of_India"

# Create new dataset
dataset = create_wikipedia_dataset(medical_title, non_medical_title)

# Data Preprocessing of dataset
dataset['text1'] = dataset['text1'].apply(lambda x: x.lower() if x else x)

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset['text1'])
y = dataset['label']

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Logistic Regression model as instructor told in class
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Example prediction on a new text
new_text = "This is a new text about medical topics."
new_text_vectorized = vectorizer.transform([new_text])
prediction = model.predict(new_text_vectorized)
print(f'Prediction for the new text: {prediction}')

import joblib

# Keep the trained model
model_filename = 'classifier_model.joblib'
joblib.dump(model, model_filename)

# Save the TF-IDF vectorizer
vectorizer_filename = 'vectorizer.joblib'
joblib.dump(vectorizer, vectorizer_filename)

print(f"Model and vectorizer saved as {model_filename} and {vectorizer_filename}")


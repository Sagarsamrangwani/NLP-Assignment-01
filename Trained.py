import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved trained model and vectorizer
model = joblib.load('classifier_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def preprocess_text(text):
    # Apply the same preprocessing steps as during training of model
    processed_text = text.lower() if text else ''
    return processed_text

def predict_category(text):
    # Preprocess the input text of data
    preprocessed_text = preprocess_text(text)

    # Vectorize the preprocessed text
    text_vectorized = vectorizer.transform([preprocessed_text])

    # Make predictions using the trained model
    prediction = model.predict(text_vectorized)

    return prediction[0]

if __name__ == "__main__":
    # Take input from the user
    user_input = input("Enter a text for prediction: ")

    # Make prediction on user input
    prediction = predict_category(user_input)
    print(f'Prediction for the input text: {prediction}')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, render_template

# Step 1: Load and preprocess the dataset
data = pd.read_csv('data/enron_spam_data.csv')  # Replace with your dataset path
data = data[['Spam/Ham', 'Subject', 'Message']]
data.columns = ['label', 'subject', 'body']

# Handle missing values by replacing NaN with empty strings
data['subject'] = data['subject'].fillna('')
data['body'] = data['body'].fillna('')

# Combine 'subject' and 'body' columns
X = data['subject'] + ' ' + data['body']
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Load the model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form submission
    email_subject = request.form.get('subject', '')
    email_body = request.form.get('body', '')

    # Preprocess and predict
    full_text = email_subject + ' ' + email_body
    tfidf_vector = vectorizer.transform([full_text])
    prediction = model.predict(tfidf_vector)

    # Return result as a response
    result = "Spam" if prediction[0] == 'spam' else "Not Spam"
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

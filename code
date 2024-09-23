import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

good_text = pd.read_csv('Good_text.csv')
bad_text = pd.read_csv('Bad_text_new.csv')

good_text['label'] = 0  # Non-malicious
bad_text['label'] = 1   # Malicious

combined_data = pd.concat([good_text, bad_text], ignore_index=True)
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

combined_data['Body'] = combined_data['Body'].astype(str)

stemmer = PorterStemmer()
combined_data['Body'] = combined_data['Body'].apply(lambda text: ' '.join([stemmer.stem(word) for word in text.split()]))

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(combined_data['Body'])

X_train, X_test, y_train, y_test = train_test_split(X, combined_data['label'], test_size=0.2, random_state=42)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

user_input = ["Dear , We would like to understand your payment experience for your recent purchase on Flipkart so that we can ensure a seamless experience for you in the future.Please fill in the survey form here.In case you are not comfortable answering any question, you may choose to skip it and move to the next one.Thanks, Flipkart Team""Dear , We would like to understand your payment experience for your recent purchase on Flipkart so that we can ensure a seamless experience for you in the future.Please fill in the survey form here.In case you are not comfortable answering any question, you may choose to skip it and move to the next one.Thanks, Flipkart Team"]# Replace with your input

user_input = ' '.join([stemmer.stem(word) for word in user_input[0].split()])
user_input_transformed = tfidf_vectorizer.transform([user_input])

user_predictions = nb_model.predict(user_input_transformed)

if user_predictions[0] == 1:
    print("Result: Malicious")
else:
    print("Result: Non-malicious")

y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(user_input)
print("TF-IDF vectorization",X)
print(f"Accuracy: {accuracy}")
print(report)
print("# Preprocess user input",user_input_transformed )
print("Stemming",combined_data['Body'])

plt.bar(["Non-Malicious", "Malicious"], user_predictions, color=['green', 'red'])
plt.title("User Input Classification Result")
plt.show()

evaluation_chart = [accuracy]  # You can add more metrics to this list if needed
metrics = ["Accuracy"]  # You can add more metric names to this list

plt.bar(metrics, evaluation_chart, color='blue')
plt.title("Model Evaluation Metrics")
plt.show()

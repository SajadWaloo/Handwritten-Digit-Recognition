import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

print("Loading the dataset...")
with open('/Users/mac/Downloads/fradulent_emails.txt', 'r', encoding='latin-1') as file:
    data = [line.strip().split(',', maxsplit=1) for line in file.readlines() if ',' in line]
print("Dataset loaded successfully.")

print("Preprocessing the data...")

emails = [str(email[1].strip()) for email in data]
labels = [email[0] for email in data]
print("Data preprocessed successfully.")

print("Converting text into numerical features...")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
print("Text converted into numerical features successfully.")

print("Splitting the data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
print("Data split into training and testing sets successfully.")

print("Training the Naive Bayes model...")

model = MultinomialNB()
model.fit(X_train, y_train)
print("Model trained successfully.")

print("Making predictions on the test set...")

y_pred = model.predict(X_test)
print("Predictions made successfully.")

print("Evaluating the model...")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Model evaluation completed successfully.")

print("Generating the bar plot for predicted label distribution...")

unique_labels, counts = np.unique(y_pred, return_counts=True)
plt.bar(unique_labels, counts, align='center')
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.title('Predicted Label Distribution')
plt.show()
print("Bar plot generated successfully.")

print("Model executed successfully.")

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset of song reviews from a CSV file
# The CSV file contains two columns: 'review' and 'sentiment'
song_reviews_df = pd.read_csv("song_reviews.csv")

# Remove rows with 'neutral' sentiment to simplify the classification task
# This step ensures that only 'positive' and 'negative' sentiments are included
song_reviews_df = song_reviews_df[song_reviews_df['sentiment'] != 'neutral']

# Convert text data (song reviews) into numerical feature vectors using the Bag of Words model
# 'max_features=2000' limits the number of unique words (features) to 2000 most frequent ones
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(song_reviews_df["review"])  # Features (review texts)
y = song_reviews_df["sentiment"]  # Labels (sentiments)

# Split the data into training and testing sets
# 'test_size=0.2' means 20% of the data will be used for testing, and 80% for training
# 'random_state=42' ensures the split is reproducible across different runs, since we don't have enough data, test resault will be different each time if we dont have this
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier (MultinomialNB) this is our algorithm
# This algorithm is well-suited for classification tasks involving discrete data, such as word counts
naive_bayes_model = MultinomialNB()

# Train the Naive Bayes classifier using the training data (features and labels)
naive_bayes_model.fit(X_train, y_train)

# Use the trained model to predict sentiments for the test data
predicted_sentiments = naive_bayes_model.predict(X_test)

# Evaluate the model's performance by calculating accuracy and generating a classification report
# 'accuracy_score' computes the ratio of correct predictions to the total number of predictions
# 'classification_report' provides precision, recall, and F1-score for each class
print(f"Accuracy: {accuracy_score(y_test, predicted_sentiments)}")
print(f"Classification Report:\n{classification_report(y_test, predicted_sentiments)}")


# Function to predict the sentiment of a new review text
def predict_sentiment(new_review_text):
    # Transform the input text into a feature vector using the trained vectorizer
    new_review_vector = vectorizer.transform([new_review_text])

    # Use the trained model to predict the sentiment (either 'positive' or 'negative')
    sentiment_prediction = naive_bayes_model.predict(new_review_vector)

    # Return the predicted sentiment (as a string)
    return sentiment_prediction[0]


# Test the sentiment prediction function with sample song reviews
print(predict_sentiment("I absolutely love this song! It's fantastic."))
print(predict_sentiment("I didn't enjoy this track. It's not good."))
print(predict_sentiment("The song is okay, but nothing special."))

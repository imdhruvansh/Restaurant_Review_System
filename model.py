import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')
corpus = []

for i in range(0, 1000):
    # Removing the special character from the reviews and replacing it with space character
    review = re.sub(pattern="[^a-zA-Z']", repl=' ', string=df['Review'][i])

    # Converting the review into lower case character
    review = review.lower()

    # if i == 6:
    #     print(review)
    # Tokenizing the review by words
    review_words = review.split()

    # if i == 6:
    #     print(review_words)

    stop_words = set(stopwords.words('english'))

    stop_words.remove('not')

    filtered_stopwords = [word for word in stop_words if "'t" not in word]

    # Removing the stop words using nltk stopwords
    review_words = [word for word in review_words if not word in set(
        filtered_stopwords)]
    
    
    lemmatizer = WordNetLemmatizer()  # Stemming the words

    review = [lemmatizer.lemmatize(word) for word in review_words]
    review = ' '.join(review)  # Joining the stemmed words

    corpus.append(review)  # Creating a corpus

# Creating Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

# Creating a pickle file for the CountVectorizer model
joblib.dump(cv, "cv.pkl")

# Model Building
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)

# Creating a pickle file for the Multinomial Naive Bayes model
joblib.dump(classifier, "model.pkl")
model = joblib.load(open('model.pkl', 'rb'))
cv = joblib.load(open('cv.pkl', 'rb'))
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy*100}%")

#Calculate and print other evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification Report:")
print(classification_report(y_test, predictions))


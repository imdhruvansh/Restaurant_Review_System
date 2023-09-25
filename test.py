import joblib

model = joblib.load(open('model.pkl', 'rb'))
cv = joblib.load(open('cv.pkl', 'rb'))

data = [text]
vectorizer = cv.transform(data).toarray()
prediction = model.predict(vectorizer)
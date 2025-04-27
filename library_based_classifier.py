from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# text for training (initially simple hand-written texts)
train_texts = [
    "أنا سعيد جدًا", 
    "أنا فرحان اليوم", 
    "أنا أشعر بالحزن", 
    "أنا مكتئب ومتعب",
    "يوم جميل وممتع",
    "أشعر بالألم والحزن"
]

# labels (happy or sad)
Y_train = [
    "^_^", 
    "^_^", 
    ":(", 
    ":(", 
    "^_^", 
    ":("
]

# convert texts to numbers (features)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)

# create a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, Y_train)

# Test on a new text
def predict_sentiment(text):
    X_test = vectorizer.transform([text])
    prediction = model.predict(X_test)
    return prediction[0]

# Example usage
# test_text = input("write a sentence to analyze the sentiment: ")
test_text = "سعيد صالح"
result = predict_sentiment(test_text)
print(f"emotions: {result}")
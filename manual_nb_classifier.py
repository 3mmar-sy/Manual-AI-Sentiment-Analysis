import math
train_texts = [
    "أنا سعيد جدا", 
    "أنا فرحان اليوم", 
    "أنا أشعر بالحزن", 
    "أنا مكتئب ومتعب",
    "يوم جميل وممتع",
    "أشعر بالألم والحزن"
]
Y_train = [
    "^_^", 
    "^_^", 
    ":(", 
    ":(", 
    "^_^", 
    ":("
]

########################################################### vectorizer = CountVectorizer()
vocab = set()
for text in train_texts:
    words = text.split()
    for word in words:
        vocab.add(word)
vocabulary = sorted(list(vocab))
    # [
    #     'أشعر'
    #     'أنا'
    #     'اليوم' 
    #     'بالألم' 
    #     'بالحزن' 
    #     'جد' 
    #     'جميل' 
    #     'سعيد' 
    #     'فرحان' 
    #     'مكتئب'
    #     'والحزن' 
    #     'ومتعب' 
    #     'وممتع' 
    #     'يوم'
    # ]

########################################################### X_train = vectorizer.fit_transform(train_texts)
X_train = []
for text in train_texts:
    vector = []
    for word_in_vocab in vocabulary:
        vector.append(text.split().count(word_in_vocab))
    X_train.append(vector)

    # [
    #     [0 1 0 0 0 1 0 1 0 0 0 0 0 0]
    #     [0 1 1 0 0 0 0 0 1 0 0 0 0 0]
    #     [1 1 0 0 1 0 0 0 0 0 0 0 0 0]
    #     [0 1 0 0 0 0 0 0 0 1 0 1 0 0]
    #     [0 0 0 0 0 0 1 0 0 0 0 0 1 1]
    #     [1 0 0 1 0 0 0 0 0 0 1 0 0 0]
    # ]

########################################################### model = MultinomialNB()
class_word_counts = {}
class_counts = {}
for cls in set(Y_train):                            # set(Y_train) = {"^_^", ":("}
    class_word_counts[cls] = [0] * len(X_train[0])  # len(X_train[0]) = 6 , #class_word_counts = {"^_^": [0, 0, 0, 0, 0, 0], ":(": [0, 0, 0, 0, 0, 0]}
    class_counts[cls] = 0                           # class_counts = {'^_^': 0, ':(': 0}
for features, label in zip(X_train, Y_train):       # zip = [([0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], '^_^'), ....)], 
                                                    # features = [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                                                    # label = '^_^'
    class_counts[label] += 1                        # class_counts = {'^_^': 3, ':(': 3}
    for idx, count in enumerate(features):          # enumerate(features) = [(0, 0), (1, 1), (2, 0), ....]
        class_word_counts[label][idx] += count      # class_word_counts = {'^_^': [0, 2, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1], ':(': [2, 2, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]}

########################################################### model.fit(X_train, train_labels)
word_probabilities = {}
total_words_in_class = {}
for cls in class_word_counts:
    word_probabilities[cls] = []                                    # word_probabilities = {'^_^': [], ':(': []}
    total_words_in_class[cls] = sum(class_word_counts[cls])         # total_words_in_class = {'^_^': 9, ':(': 9}
    for word_idx, word_count in enumerate(class_word_counts[cls]):  # enumerate(class_word_counts[cls]) = [(0, 0), (1, 2), (2, 1)...]
        prob = (word_count + 1) / (total_words_in_class[cls] + len(vocabulary)) # prob = (0+1)/(9+14) = 0.066667 (6%) # prob = (2+1)/(9+14) = 0.153846 (15%)
        word_probabilities[cls].append(prob)    # word_probabilities = {'^_^': [0.066667, 0.153846, ....], ':(': [....]}
class_probabilities = {}
for cls in class_counts:
    class_probabilities[cls] = class_counts[cls] / len(Y_train)   # 3/6 > class_probabilities = {'^_^': 0.5, ':(': 0.5}

########################################################### def predict_sentiment(text)
def predict_sentiment(text):
    global word_probabilities, class_probabilities, vocabulary
    words = text.split()                                    # words = ["سعيد", "صالح"]
    features = [words.count(word) for word in vocabulary]   # features = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    class_scores = {}
    for cls in class_probabilities:
        score = math.log(class_probabilities[cls])          # score = log(0.5) = -0.693147
        for idx, count in enumerate(features):              # enumerate(features) = [(0, 0), (1, 0), (2, 0), ..., (7, 1), ...]
            if count > 0:
                score += count * math.log(word_probabilities[cls][idx]) #idx_1: score += 0 * log(0.066667) = 0
                                                                        #idx_7: score += 1 * log(0.153846) = -1.871802
        class_scores[cls] = score                                       # class_scores = {':(': -3.828641396489095, '^_^': -3.1354942159291497}
    return max(class_scores, key=class_scores.get)


########################################################### Example usage
# test_text = input("write a sentence to analyze the sentiment: ")
test_text = "سعيد صالح"
result = predict_sentiment(test_text)
print(f"emotions: {result}")
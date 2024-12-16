import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Load data
data1 = pd.read_csv(r"F:\VS Code\Machine Learning Projects\Malayalam Tamil Foul Word detection\AWM_train.csv")
data2 = pd.read_csv(r"F:\VS Code\Machine Learning Projects\Malayalam Tamil Foul Word detection\AWM_dev.csv")
malayalam_dataset = pd.concat([data1, data2], axis=0)

from indicnlp.tokenize import indic_tokenize

malayalam_stopwords = {
    "അവൻ", "അവൾ", "അവർ", "അത്", "ഇത്", "ഞാൻ", "നാം", "നിങ്ങൾ", 
    "അവിടെ", "ഇവിടെ", "എന്നെ", "അതേ", "ഇല്ല", "അതിനു", "അതിന്", 
    "അവസാനം", "അവിടെ", "അവയുടെ", "ഇതിൽ", "ഇതിന്റെ", "അതിൽ", 
    "നമ്മുടെ", "അവിടെ", "വളരെ", "അല്ല", "ഒന്ന്", "ഒരു", "എവിടെ", 
    "എങ്ങനെ", "എപ്പോൾ", "എന്ത്", "കാരണം", "മറ്റു", "മറ്റെല്ലാം", 
    "പിന്നെ", "പിന്നീട്", "ഇപ്പോൾ", "ഈ", "ഈയുള്ള", "ആ", "ആയിരിക്കാം"
}

def simple_malayalam_stemmer(word):
    suffixes = [
    'കൾ', 'യുടെ', 'ത്തിന്', 'ങ്ങൾ', 'ത്തിൽ', 'ഉം', 'നാൽ', 'യെ', 
    'ഇൽ', 'വരെ', 'ഇക്ക്', 'ഓടെ', 'ഉടെ', 'പ്പടെ', 'ൽ', 'പോലെ', 
    'യുള്ള', 'ക്ക്', 'നെ', 'രണ്ടു', 'രീതിയിലുള്ള', 'നു', 'നുൽ',
    'ആയ', 'ടെ', 'ക്കു', 'ഉണ്ട്', 'പോലുള്ള', 'ല്‍', 'ഏറ്റം'
]

    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def malayalam_stemming(content):
    tokens = indic_tokenize.trivial_tokenize(content, lang='ml')
    stemmed_content = [simple_malayalam_stemmer(word) for word in tokens if word not in malayalam_stopwords]
    return ' '.join(stemmed_content)

# Preprocess the text
malayalam_dataset['Text'] = malayalam_dataset['Text'].apply(malayalam_stemming)

# Separate features and labels
X = malayalam_dataset['Text'].values
Y = malayalam_dataset['Class'].values

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert text data to numerical data using TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 5))
X_tfidf_train = vectorizer.fit_transform(X_train)
X_tfidf_test = vectorizer.transform(X_test)

# Train the model
#model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
model = SVC(kernel='linear', class_weight='balanced')
model.fit(X_tfidf_train, Y_train)

# Accuracy score on the training data
X_train_prediction = model.predict(X_tfidf_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score on the training data malayalam: ', training_data_accuracy)

# Accuracy score on the testing data
X_test_prediction = model.predict(X_tfidf_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on the testing data malayalam: ', test_data_accuracy)

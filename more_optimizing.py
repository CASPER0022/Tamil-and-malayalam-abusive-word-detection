import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#malayalam_dataset = pd.read_csv("F:\VS Code\Machine Learning Projects\Malayalam Tamil Foul Word detection\AWM_train.csv")

data1 = pd.read_csv(r"F:\VS Code\Machine Learning Projects\Malayalam Tamil Foul Word detection\AWM_train.csv")
data2 = pd.read_csv(r"F:\VS Code\Machine Learning Projects\Malayalam Tamil Foul Word detection\AWM_dev.csv")
malayalam_dataset = pd.concat([data1,data2],axis=0)
X = malayalam_dataset.drop(columns='Class', axis = 1)

Y = malayalam_dataset['Class']


from indicnlp.tokenize import indic_tokenize

malayalam_stopwords = {'എന്ന', 'ആണ്', 'വഴി', 'ഒരു', 'ഇതും', 'പോലെ', 'എന്നാണ്'}

def simple_malayalam_stemmer(word):
    suffixes = ['കൾ', 'യുടെ', 'ത്തിന്', 'ങ്ങൾ', 'ത്തിൽ']  
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def malayalam_stemming(content):
    # Tokenize content
    tokens = indic_tokenize.trivial_tokenize(content, lang='ml')

    # Remove stopwords and apply custom stemming
    stemmed_content = [simple_malayalam_stemmer(word) for word in tokens if word not in malayalam_stopwords]

    # Join back to a single string
    return ' '.join(stemmed_content)

# Example Usage
malayalam_dataset['Text'] = malayalam_dataset['Text'].apply(malayalam_stemming)

X = malayalam_dataset['Text'].values
Y = malayalam_dataset['Class'].values
X_list = X.tolist()
Y_list = Y.tolist()

#converting the texual data to numerical data

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(X_list)


#model = LogisticRegression()
#model = SVC(kernel='linear', class_weight='balanced')
model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
model.fit(X_tfidf, Y)

# accuracy score on the training data
X_train_prediction = model.predict(X_tfidf)
training_data_accuracy = accuracy_score(X_train_prediction, Y)
print('Accuracy score of the training data : ', training_data_accuracy)






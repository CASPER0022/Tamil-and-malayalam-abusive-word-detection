import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Load data
data1 = pd.read_csv(r"F:\VS Code\Machine Learning Projects\Malayalam Tamil Foul Word detection\AWT_train.csv")
data2 = pd.read_csv(r"F:\VS Code\Machine Learning Projects\Malayalam Tamil Foul Word detection\AWT_dev.csv")
malayalam_dataset = pd.concat([data1, data2], axis=0)

from indicnlp.tokenize import indic_tokenize

malayalam_stopwords = {
    "அவன்", "அவள்", "அவர்கள்", "அது", "இது", "நான்", "நாம்", "நீங்கள்",
    "அங்கே", "இங்கே", "என்னை", "அதை", "இல்லை", "அதற்காக", "அதற்கு",
    "இல்லை", "பின்னர்", "இப்போது", "இந்த", "அந்த", "பொறுப்பு", "மற்றவை",
    "ஒரு", "ஏன்", "எப்போது", "எப்படி", "காரணம்", "மற்றும்", "அந்த",
    "சில", "இல்லாமல்", "ஆனால்"
}

def simple_malayalam_stemmer(word):
    suffixes = ['ஆன்', 'ஆள்', 'க்கு', 'படி', 'ல்', 'ன்', 'ம்'
                 'வது', 'உம்', 'இல்', 'ல'
                 'மான', 'காரன்', 'பட்ட', 'வெள்ளி'
                 'ச்சி', 'அவர்கள்'
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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

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
print('Accuracy score on the training data tamil: ', training_data_accuracy)

# Accuracy score on the testing data
X_test_prediction = model.predict(X_tfidf_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on the testing data tamil: ', test_data_accuracy)

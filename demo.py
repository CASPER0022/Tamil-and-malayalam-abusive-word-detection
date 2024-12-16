import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import fasttext
from indicnlp.tokenize import indic_tokenize

# Load data
data1 = pd.read_csv(r"F:\VS Code\Machine Learning Projects\Malayalam Tamil Foul Word detection\AWM_train.csv")
data2 = pd.read_csv(r"F:\VS Code\Machine Learning Projects\Malayalam Tamil Foul Word detection\AWM_dev.csv")
malayalam_dataset = pd.concat([data1, data2], axis=0)

# Load fastText model for Malayalam (ensure you have downloaded 'cc.ml.300.bin' model)
model = fasttext.load_model('cc.ml.300.bin')

# Stopwords list
malayalam_stopwords = {
    "അവൻ", "അവൾ", "അവർ", "അത്", "ഇത്", "ഞാൻ", "നാം", "നിങ്ങൾ", 
    "അവിടെ", "ഇവിടെ", "എന്നെ", "അതേ", "ഇല്ല", "അതിനു", "അതിന്", 
    "അവസാനം", "അവിടെ", "അവയുടെ", "ഇതിൽ", "ഇതിന്റെ", "അതിൽ", 
    "നമ്മുടെ", "അവിടെ", "വളരെ", "അല്ല", "ഒന്ന്", "ഒരു", "എവിടെ", 
    "എങ്ങനെ", "എപ്പോള്", "എന്ത്", "കാരണം", "മറ്റു", "മറ്റെല്ലാം", 
    "പിന്നെ", "പിന്നീട്", "ഇപ്പോൾ", "ഈ", "ഈയുള്ള", "ആ", "ആയിരിക്കാം"
}

# Function to get the average word vector for a text using fastText
def get_fasttext_vector(text):
    tokens = indic_tokenize.trivial_tokenize(text, lang='ml')
    vectors = []
    
    for word in tokens:
        if word not in malayalam_stopwords:
            vectors.append(model.get_word_vector(word))  # Get word vector from fastText
    
    if vectors:
        return np.mean(vectors, axis=0)  # Return the average vector
    else:
        return np.zeros(model.get_dimension())  # Return a zero vector if no valid tokens are found

# Preprocess the text by converting each text to its fastText vector representation
malayalam_dataset['Text'] = malayalam_dataset['Text'].apply(get_fasttext_vector)

# Separate features and labels
X = np.array(list(malayalam_dataset['Text']))  # Convert list of vectors to numpy array
Y = malayalam_dataset['Class'].values

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the model using SVC
model = SVC(kernel='linear', class_weight='balanced')
model.fit(X_train, Y_train)

# Accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score on the training data Malayalam: ', training_data_accuracy)

# Accuracy score on the testing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on the testing data Malayalam: ', test_data_accuracy)

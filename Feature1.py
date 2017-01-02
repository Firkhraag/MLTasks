#Text Classification
#Bag Of Words
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

df_test=pd.read_csv('news_test.txt', header=0, delimiter="\t", quoting=3)
df_train=pd.read_csv('news_train.txt', header=0, delimiter="\t", quoting=3)

def text_to_words( raw_text ):
    #Function to convert a raw text to a string of words
    #the input is a single string (a raw text), and 
    #the output is a single string (a preprocessed text)

    #Remove non-letters        
    letters_only = re.sub("[^а-яА-Я]", " ", raw_text) 

    #Convert to lower case, split into individual words
    words = letters_only.lower().split()                             

    #In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    stops = set(stopwords.words("russian"))                  
	
    #Remove stop words
    meaningful_words = [w for w in words if not w in stops]   

    #Join the words back into one string separated by space, and return the result.
    return( " ".join( meaningful_words ))   

#Get the number of texts
num_train_texts = df_train["content"].size
num_train_titles = df_train["title"].size
num_test_texts = df_test["content"].size
num_test_titles = df_test["title"].size

#Initialize an empty list to hold the clean texts
clean_train_texts = []
clean_train_titles = []
clean_test_texts = []
clean_test_titles = []

#Loop over each text
for i in range(num_train_texts):
    clean_train_texts.append(text_to_words(df_train["content"][i]))
	
for i in range(num_train_titles):
    clean_train_titles.append(text_to_words(df_train["title"][i]))

for i in range(num_test_texts):
    clean_test_texts.append(text_to_words(df_test["content"][i]))

for i in range(num_test_titles):
    clean_test_titles.append(text_to_words(df_test["title"][i]))

for i in range(num_train_texts):
    clean_train_texts[i] = clean_train_texts[i] + " " + clean_train_titles[i]

for i in range(num_test_texts):
    clean_test_texts[i] = clean_test_texts[i] + " " + clean_test_titles[i]

#Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 8000)

#fit_transform() does two functions: First, it fits the model
#and learns the vocabulary; second, it transforms our training data
#into feature vectors. The input to fit_transform should be a list of strings.
train_data_features = vectorizer.fit_transform(clean_train_texts)

#Numpy arrays are easy to work with, so convert the result to an array
train_data_features = train_data_features.toarray()

print (train_data_features.shape)

#Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

#Fit the forest to the training set, using the bag of words as features and the type labels as the response variable
forest = forest.fit(train_data_features, df_train["type"])

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_texts)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe
output = pd.DataFrame(data={"type":result})

# Use pandas to write the comma-separated output file
output.to_csv("news_output.txt", index=False, quoting=3)

print("success")

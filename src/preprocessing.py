#We will clean the text and preapre if for the model, how can we do thi?
#     1) Convert all to lower case 
#     2) Remove special characters
#     3) Remove numbers
#     4) Remove stopwords

import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '',text) #removes numbers and punctuations
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)
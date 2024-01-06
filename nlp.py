import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()
def tokonize(sentence):
    return nltk.word_tokenize(sentence)
ignore=['?','.',',','!',',']
def stem(a):
    return [stemmer.stem(word.lower())  for word in a if word not in ignore]


# a="how long does shipping taken Bhavesh"
# a=tokonize(a)
# print(a)
# a=stem(a)
# # a=[stem(a1) for a1 in a] 
# print(a)

def bag_of_words(sentence,all_words):
    sentence=stem(sentence)
    bag=np.zeros(len(all_words),dtype=np.float32)
    for i,w in enumerate(all_words):
        if w in sentence:
            bag[i]=1.0
    return bag
a= ['tell', 'me', 'someth', 'funni', 'do', 'you']
allword=['hi', 'hey', 'how', 'are', 'you', 'is', 'anyon', 'there', 'hello', 'good', 'day', 'bye', 'see', 'you', 'later', 'goodby', 'thank', 'thank', 'you', 'that', "'s", 'help', 'thank', "'s", 'a', 'lot', 'which', 'item', 'do', 'you', 'have', 'what', 'kind', 'of', 'item', 'are', 'there', 'what', 'do', 'you', 'sell', 'do', 'you', 'take', 'credit', 'card', 'do', 'you', 'accept', 'mastercard', 'can', 'i', 'pay', 'with', 'paypal', 'are', 'you', 'cash', 'onli', 'how', 'long', 'doe', 'deliveri', 'take', 'how', 'long', 'doe', 'ship', 'take', 'when', 'do', 'i', 'get', 'my', 'deliveri', 'tell', 'me', 'a', 'joke', 'tell', 'me', 'someth', 'funni', 'do', 'you', 'know', 'a', 'joke']
bag=bag_of_words(a,allword)
# print(bag)
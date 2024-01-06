import torch
from model import NeuralNet
from nlp import bag_of_words,tokonize
import json
import random
with open('data.json','r') as F:
    intents=json.load(F)

FILE='data.pth'
data=torch.load(FILE)

input_size=data["input_size"]
hidden_size=data["hidden_size"]
output_size=data["output_size"]
all_words=data["allwords"]
model_state=data["modelstate"]
tags=data["tags"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()
bot="sam"

print("chat started type 'quit' to exit ")

def reply(sentence):


    sentence=tokonize(sentence)
    X=bag_of_words(sentence,all_words)
    X=X.reshape(1,X.shape[0])
    X=torch.from_numpy(X)

    output=model(X)
    _,predicted=torch.max(output,dim=1)
    tag=tags[predicted.item()]
    probs=torch.softmax(output,dim=1)
    probs=probs[0][predicted.item()]
    probs=probs.item() 
    if probs >0.75:
        for intent in intents["intents"]:
            if tag==intent["tag"]:
                return (f'{random.choice(intent["patterns"])}')
    else:
        return (" dont understand")    
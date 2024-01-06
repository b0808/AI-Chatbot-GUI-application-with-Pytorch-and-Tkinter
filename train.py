import json
import numpy as np
# from nlp import tokonize,stem
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nlp import bag_of_words, tokonize, stem
from model import NeuralNet
with open('data.json','r') as f:
    intents=json.load(f)
    # print(data) 
all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    qtag = intent['tag']
    # add to tag list
    tags.append(qtag)
    # print(tags.index(qtag))
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokonize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, qtag))
all_words=stem(all_words)
# print(xy)
# print(all_words)
all_words=sorted(set(all_words))
tags=sorted(set(tags))
# print(xy)
# print(tags)
xtrain=[]
ytrain=[]
for (pattern_sentence,qtag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    xtrain.append(bag)
    # print(xtrain)
    label=tags.index(qtag)
    ytrain.append(label)
# print(ytrain)
xtrain=np.array(xtrain)
ytrain=np.array(ytrain)
# print(ytrain)
class train(Dataset):
    def __init__(self):
        self.n_sample=len(xtrain)
        self.x_data=xtrain
        self.y_data=ytrain
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.n_sample
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(xtrain[0])
hidden_size = 8
output_size = len(tags)
dataset=train()
train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
model = NeuralNet(input_size, hidden_size, output_size)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)/
# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words
        labels = labels.to(dtype=torch.long)
        # print(labels)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

data={
    "modelstate":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "allwords":all_words,
    "tags":tags 
}     
FILE="data.pth"
torch.save(data,FILE)
print(f'{FILE}')
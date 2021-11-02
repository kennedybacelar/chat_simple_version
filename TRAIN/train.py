import numpy as np
import random, json, os, sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__),'../SRC'))
from SRC.nltk_utils import bag_of_words, tokenize, stem
from SRC.model import NeuralNet
from SRC.compute_topics import compute_topics


class ChatDataset(Dataset):

    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

def train():
  print('treinando')
  intents = compute_topics()
  all_words = []
  tags = []
  xy = []

  for intent in intents['intents']:
      tag = intent['tag']
      tags.append(tag)
      for pattern in intent['patterns']:
          w = tokenize(pattern)
          all_words.extend(w)
          xy.append((w, tag))


  ignore_words = ['?', '.', '!']
  all_words = [stem(w) for w in all_words if w not in ignore_words]

  all_words = sorted(set(all_words))
  tags = sorted(set(tags))

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  X_train = []
  y_train = []
  for (pattern_sentence, tag) in xy:
      bag = bag_of_words(pattern_sentence, all_words)
      X_train.append(bag)
      label = tags.index(tag)
      y_train.append(label)

  X_train = np.array(X_train)
  y_train = np.array(y_train)

  # Hyper-parameters 
  num_epochs = 1000
  batch_size = 8
  learning_rate = 0.001
  input_size = len(X_train[0])
  hidden_size = 8
  output_size = len(tags)
  print(input_size, output_size)


  dataset = ChatDataset(X_train, y_train)
  train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = NeuralNet(input_size, hidden_size, output_size).to(device)

  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # Train the model
  for epoch in range(num_epochs):
      for (words, labels) in train_loader:
          words = words.to(device)
          labels = labels.to(dtype=torch.long).to(device)
          
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

  print(f'final loss: {loss.item():.4f}')

  data = {
  "model_state": model.state_dict(),
  "input_size": input_size,
  "hidden_size": hidden_size,
  "output_size": output_size,
  "all_words": all_words,
  "tags": tags
  }

  FILE = 'data.pth'
  torch.save(data, FILE)

  print(f'training complete. file saved to {FILE}')

if __name__ == '__main__':
  train()
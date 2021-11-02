import random, json, os, sys, torch
sys.path.append(os.path.join(os.path.dirname(__file__),'../SRC'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../TRAIN'))

from SRC.model import NeuralNet
from SRC.nltk_utils import bag_of_words, tokenize
from SRC.compute_topics import compute_topics
from TRAIN import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE = "data.pth"

def load_dataPTH_file():
  data = torch.load(FILE)
  return data

def load_model():
  data = load_dataPTH_file()
  input_size = data["input_size"]
  hidden_size = data["hidden_size"]
  output_size = data["output_size"]
  model = NeuralNet(input_size, hidden_size, output_size).to(device)
  return model

def conversa(sentence):

  data = load_dataPTH_file()
  model = load_model()

  model_state = data["model_state"]
  all_words = data['all_words']
  tags = data['tags']

  model.load_state_dict(model_state)
  model.eval()

  intents = compute_topics()

  bot_name = "Sam"

  sentence = tokenize(sentence)
  X = bag_of_words(sentence, all_words)
  X = X.reshape(1, X.shape[0])
  X = torch.from_numpy(X).to(device)

  output = model(X)
  _, predicted = torch.max(output, dim=1)

  tag = tags[predicted.item()]

  probs = torch.softmax(output, dim=1)
  prob = probs[0][predicted.item()]
  if prob.item() > 0.8:
      for intent in intents['intents']:
          if tag == intent["tag"]:
              print(f"{bot_name}: {random.choice(intent['responses'])}")
              #return f"{bot_name}: {random.choice(intent['responses'])}"
  else:
      print(f"{bot_name}: I do not understand...")

if __name__ == '__main__':
  if not os.path.isfile(FILE):
    train.train()
  else:
    read_input = input('Would you like to retrain the model? Y/N\n').lower()
    if read_input == 'y':
      train.train()
  print("Let's chat! (type 'quit' to exit)")
  while(True):
    sentence = input('')
    if sentence == 'quit':
      break
    conversa(sentence)
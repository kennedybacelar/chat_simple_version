import os, sys

def compute_topics():

  response_topics = { 'intents': [] }
  files_to_be_loaded = { 'answers.txt', 'questions.txt' }
  _dir = 'TOPICS'

  for root,dirs,files in os.walk(_dir, topdown=True):
    if files_to_be_loaded.issubset(set(files)):
      with open(f'{root}/questions.txt') as file:
        patterns = [line.replace('\n', '') for line in file.readlines()]

      with open(f'{root}/answers.txt') as file:
        responses = [line.replace('\n', '') for line in file.readlines()]

      response_topics['intents'].append(
      {
        'tag': root,
        'patterns': patterns,
        'responses': responses
      }
    )

  return response_topics
import numpy as np
import matplotlib.pyplot as plt
import requests


def entrop(P):
    if P == 0.0 or P == 1.0:
        return 0.0
    else:
        return -P * np.log(P) - (1 - P) * np.log(1 - P)


filename = 'random_words.txt'
path = 'https://raw.githubusercontent.com/kupav/data-sc-intro/main/data/'
data = requests.get(path + filename)
assert data.status_code == 200
text = data.text.split('\n')[:-1]
words = set(text)
probabilities = np.zeros(len(words))
c = 0
for word in words:
    for element in text:
        if element == word:
            probabilities[c] += 1
    c += 1
probabilities /= len(text)
c = 0
for word in words:
    print(f'Information enthropy of word {word} is {np.round(entrop(probabilities[c]), 3)}')
    c += 1

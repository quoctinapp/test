import random
import json
import pickle
import numpy as np

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

with open("THE FUNCTIONS RUNNED/chatbot/data-trained.json", "r", encoding="utf-8") as file:
    intents_data = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

for intent in intents_data["questions"]:
    for key_word in intent["keys"]:
        word_list = nltk.word_tokenize(key_word.lower())
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('THE FUNCTIONS RUNNED/chatbot/words.pkl', 'wb'))
pickle.dump(classes, open('THE FUNCTIONS RUNNED/chatbot/classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x = np.array([x[0] for x in training])
train_y = np.array([x[1] for x in training])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))  
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))  

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=3, verbose=1) #200 - 5 - 1
model.save('THE FUNCTIONS RUNNED/chatbot/chatbot_model.h5')

print("Done")
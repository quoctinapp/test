import random
import json
import pickle
import numpy as np
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

with open("chatbot/data-trained.json", "r", encoding="utf-8") as file:
    intents_data = json.load(file)

words = pickle.load(open('THE FUNCTIONS RUNNED/chatbot/words.pkl', 'rb'))
classes = pickle.load(open('THE FUNCTIONS RUNNED/chatbot/classes.pkl', 'rb'))
model = load_model('THE FUNCTIONS RUNNED/chatbot/chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    return_list = []
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['questions']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['answers'])
            break
    return result

print("GO, Bot is running!")
while True:
    message = input("Moi ban nhap cau hoi: ")
    intents = predict_class(message)
    res = get_response(intents, intents_data)
    print(res)

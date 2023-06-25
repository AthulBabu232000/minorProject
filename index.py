import random
import json
import torch
import streamlit as st
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Naruto"

st.title("Chatbot")

counter = 0
counters=0

while True:
    with st.form(key=f'form-{counters}'):
        counters+=1
        user_input = st.text_input("",key=f"You ({counter}): ")
        formButton=st.form_submit_button(label="Send")
        while not formButton:
            pass
        if formButton:
            sentence = user_input
            sentence = tokenize(sentence)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]

            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        bot_response = f"{bot_name}: {random.choice(intent['responses'])}"
            else:
                bot_response = f"{bot_name}: I do not understand..."

            st.write(bot_response)

            # Check if the user wants to quit
            if user_input.lower() == 'quit':
                break
            if len(user_input)>0:
                counter+=1
           

            

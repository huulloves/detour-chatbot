from database import get_connection, setup_database, log_conversation

import random
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# load intent-based model
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
intent_model = NeuralNet(input_size, hidden_size, output_size).to(device)
intent_model.load_state_dict(model_state)
intent_model.eval()

def get_intent_response(user_history):
    # combine the last N user utterances for intent detection
    N = 3
    if len(user_history) < N:
        combined = " ".join(user_history)
    else:
        combined = " ".join(user_history[-N:])
    sentence = tokenize(combined)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = intent_model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    for intent in intents['intents']:
        if tag == intent["tag"]:
            return tag, prob.item(), intent.get('responses', [])
    return None, prob.item(), []

# load DialoGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def get_dialogpt_response(history):
    # history is a list of alternating "You: ..." and "Bot: ..." strings
    prompt = "\n".join(history) + "\nBot:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    response_ids = dialogpt_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[1] + 50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    response = tokenizer.decode(response_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    conn = get_connection()
    setup_database(conn)
    user_id = "user1"

    print("Let's chat! (type 'quit' to exit)")
    chat_history = []
    user_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        user_history.append(user_input)
        chat_history.append(f"You: {user_input}")

        tag, confidence, responses = get_intent_response(user_history)

        if tag and confidence > 0.8 and responses:
            response = random.choice(responses)
        else:
            if tag and confidence > 0.4:
                clarifying_prompt = (
                    "The conversation so far:\n" +
                    "\n".join(chat_history) +
                    "\nBot: Can you ask a clarifying question to better understand the user's intent?"
                )
                chat_history.append(f"Bot: {clarifying_prompt}")
                response = get_dialogpt_response(chat_history)
                chat_history.pop()  # remove clarifying prompt from history after use
            else:
                response = get_dialogpt_response(chat_history)

        print(f"Bot: {response}")
        chat_history.append(f"Bot: {response}")
        log_conversation(conn, user_id, tag if tag else "unknown", user_input, response)
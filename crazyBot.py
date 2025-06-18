from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# load DialoGPT-medium
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def get_dialogpt_response(history):
    prompt = "\n".join(history) + "\nBot:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    response_ids = model.generate(
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
    print("Let's chat! (type 'quit' to exit)")
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        chat_history.append(f"You: {user_input}")
        response = get_dialogpt_response(chat_history)
        print(f"Bot: {response}")
        chat_history.append(f"Bot: {response}")
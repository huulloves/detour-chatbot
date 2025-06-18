import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from database import get_connection, setup_database, log_conversation, get_conversations

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_mistral_response(history, user_input):
    prompt = ""
    for i in range(0, len(history), 2):
        prompt += f"[INST] {history[i]} [/INST]"
        if i+1 < len(history):
            prompt += f" {history[i+1]} "
    prompt += f"[INST] {user_input} [/INST]"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.95,
            top_k=40,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    user_id = "user1"
    try:
        conn = get_connection()
        setup_database(conn)
        print("Database setup complete.")
    except Exception as e:
        print(f"Database connection failed: {e}")
        exit(1)

    print("Let's chat! (type 'quit' to exit)")

    # fetch and flatten chat history
    chat_tuples = get_conversations(conn, user_id)
    # fast access to chat history to build prompt during current session
    chat_history = []
    # reverse to get chronological order (oldest first)
    for user_input, bot_response, _ in reversed(chat_tuples):
        chat_history.append(user_input)
        chat_history.append(bot_response)

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        bot_response = get_mistral_response(chat_history, user_input)
        print(f"Bot: {bot_response}")
        # persistent chat history stored to database
        log_conversation(conn, user_id, user_input, bot_response)
        chat_history.append(user_input)
        chat_history.append(bot_response)

    conn.close()
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def generate_response(self, user_input, stream=False):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        
        if stream:
            # Initialize the streamer for streaming output
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Define generation arguments
            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": 32768
            }
            
            # Run generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Collect the full response while streaming
            response_chunks = []
            for new_text in streamer:
                response_chunks.append(new_text)
                yield new_text
            
            response = ''.join(response_chunks)
        else:
            response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        if not stream:
            return response

# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot()

    print("=== Non-streaming mode ===")
    # First input (without streaming)
    user_input_1 = "How many r's in strawberries?"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print("----------------------")

    # Second input with /no_think
    user_input_2 = "Then, how many r's in blueberries? /no_think"
    print(f"User: {user_input_2}")
    response_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}") 
    print("----------------------")

    print("\n=== Streaming mode ===")
    # Third input with streaming enabled
    user_input_3 = "Really? /think"
    print(f"User: {user_input_3}")
    print("Bot: ", end="", flush=True)
    for text_chunk in chatbot.generate_response(user_input_3, stream=True):
        print(text_chunk, end="", flush=True)
    print("\n----------------------")

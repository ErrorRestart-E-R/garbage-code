import ollama
import json
import os
import sys

# Add current directory to path to import config
sys.path.append(os.getcwd())
import config

print(f"Connecting to: {config.OLLAMA_HOST}")
model_name = config.LLM_MODEL_NAME

try:
    client = ollama.Client(host=config.OLLAMA_HOST)
    
    print(f"Testing chat with model: {model_name}")
    response = client.chat(model=model_name, messages=[{'role': 'user', 'content': 'Hello'}])
    
    print("Chat response type:", type(response))
    print("Chat response content:", response)
    
    if hasattr(response, 'message'):
        print("response.message type:", type(response.message))
        print("response.message content:", response.message)
        if hasattr(response.message, 'content'):
             print("response.message.content:", response.message.content)
    
    # Check if it supports dict access
    try:
        print("Dict access response['message']:", response['message'])
    except Exception as e:
        print(f"Dict access failed: {e}")

except Exception as e:
    print(f"Error: {e}")

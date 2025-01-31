import requests
import json

# Your Together API key
api_key = "1c613ebcfc86ab9e33444bbb4bc4c3bb657d5dfcab4a7edf68adc8f5316ad55f"

# API endpoint
url = "https://api.together.xyz/v1/completions"

# Headers for the API request
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Payload for the API request
payload = {
    "model": "mistralai/Mistral-7B-Instruct-v0.2",  # Model name
    "prompt": "What is react",      # Your input prompt
    "max_tokens": 100,                              # Maximum number of tokens to generate
    "temperature": 0.7,                             # Controls randomness (0 = deterministic, 1 = creative)
    "top_p": 0.9,                                   # Controls diversity (0 = narrow, 1 = broad)
    "stop": ["\n"]                                  # Stop sequence (optional)
}

# Send the request to the Together API
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Check if the request was successful
if response.status_code == 200:
    # Parse the response
    result = response.json()
    generated_text = result["choices"][0]["text"]
    print("Generated Text:", generated_text)
else:
    print("Error:", response.status_code, response.text)
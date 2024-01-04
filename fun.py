import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Hugging Face Token
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

API_URL = "https://ghprpg1pq3gveb5b.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Authorization": f"Bearer {HF_TOKEN}",
	"Content-Type": "application/json"
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "What is the capital of France?",
})



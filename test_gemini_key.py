import google.generativeai as genai
import os

# Load API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY not set in environment.")
    exit(1)

genai.configure(api_key=api_key)

print("Available Gemini models for your API key:")
for m in genai.list_models():
    print("-", m.name) 
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

print("Testing Interactions API...")
try:
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=[
            {"type": "text", "text": "Say exactly this: 'The Interactions API is working perfectly!'"}
        ]
    )
    print("SUCCESS! Output:")
    print(">", interaction.output_text)
except Exception as e:
    print("ERROR:")
    print(e)

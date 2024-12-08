from openai import OpenAI
import os
from dotenv import load_dotenv

def test_openai_connection():
    # Load environment variables
    load_dotenv()

    # Initialize client
    client = OpenAI()

    try:
        # Simple test request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Hello, World!'"}],
            temperature=0.5,
        )
        
        # Print full response for debugging
        print("Full response object:")
        print(response)
        
        # Print just the message content
        print("\nMessage content:")
        print(response.choices[0].message.content)
        
        print("\nAPI connection successful!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_openai_connection() 
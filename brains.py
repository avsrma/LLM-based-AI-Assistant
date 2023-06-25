import json
import os
import openai

from dotenv import load_dotenv

def chatgpt(content, model="gpt-4-32k-deployment"):

    load_dotenv(".env")

    openai.api_type = "azure"
    openai.api_base = os.environ.get("OPENAI_API_BASE") 
    openai.api_version = "2023-03-15-preview"
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
                engine=model,
                messages=content
            )

    output = response['choices'][0]['message']['content']
    return output


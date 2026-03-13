from dotenv import load_dotenv
import os
load_dotenv()
from openai import OpenAI

client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.getenv('OPENROUTER_API_KEY'))

resp = client.chat.completions.create(
    model='nvidia/nemotron-3-super-120b-a12b:free',
    messages=[{'role': 'user', 'content': 'Reply with: ok'}],
    timeout=15
)
print('Model used:', resp.model)
print('Response:', resp.choices[0].message.content)

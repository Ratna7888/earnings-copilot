from dotenv import load_dotenv; import os; load_dotenv()
from google import genai
from openai import OpenAI

# Test Gemini
print('Testing Gemini...')
try:
    g = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    r = g.models.generate_content(model='gemini-2.0-flash', contents='Reply: ok')
    print('Gemini OK:', r.text.strip())
except Exception as e:
    print('Gemini FAIL:', str(e)[:100])

# Test OpenRouter
print('Testing OpenRouter...')
try:
    c = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.getenv('OPENROUTER_API_KEY'))
    r = c.chat.completions.create(
        model='openrouter/free',
        messages=[{'role':'user','content':'Reply: ok'}],
        timeout=15
    )
    print('OpenRouter OK:', r.choices[0].message.content.strip())
except Exception as e:
    print('OpenRouter FAIL:', str(e)[:100])

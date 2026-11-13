import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPENROUTER_API_KEY')
)

MODEL = 'openrouter/free'

def chat(messages, max_tokens=None, attempts=3, **kwargs):
    """Return the text reply from an OpenRouter free model, or raise.

    openrouter/free routes each call to a random free model. Reasoning models can
    spend the whole max_tokens budget thinking and return content=None, so ask with
    reasoning off; models that refuse that get low-effort reasoning and extra room.
    """
    last_error = None
    for _ in range(attempts):
        reasoning, budget = {'enabled': False}, max_tokens
        for _ in range(2):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=budget,
                    extra_body={'reasoning': reasoning},
                    **kwargs
                )
                content = resp.choices[0].message.content
                if content:
                    return content
                last_error = ValueError(f'{resp.model} returned an empty response')
                break
            except Exception as e:
                last_error = e
                if 'reasoning is mandatory' not in str(e).lower():
                    break
                reasoning = {'effort': 'low'}
                budget = max_tokens + 2000 if max_tokens else None
    raise last_error

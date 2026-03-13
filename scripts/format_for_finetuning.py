import json, random

with open('data/finetune/balanced_training_data.json') as f:
    data = json.load(f)

# Skip examples with missing/invalid output
clean = []
for ex in data:
    try:
        out = json.loads(ex['output'])
        if out.get('confidence') in ('HIGH', 'UNVERIFIABLE'):
            clean.append(ex)
    except:
        pass

print(f'Clean examples: {len(clean)}')

formatted = []
for ex in clean:
    formatted.append({
        'messages': [
            {
                'role': 'system',
                'content': 'You are a financial KPI extraction model. Extract metrics from SEC filing chunks as JSON. If a metric cannot be verified from the text, output {"confidence": "UNVERIFIABLE"}. Never invent numbers.'
            },
            {
                'role': 'user',
                'content': 'Filing chunk:\n' + ex['chunk_text'] + '\n\nExtract: ' + ex['question']
            },
            {
                'role': 'assistant',
                'content': ex['output']
            }
        ]
    })

random.shuffle(formatted)
split = int(len(formatted) * 0.9)
train, val = formatted[:split], formatted[split:]

with open('data/finetune/train.json', 'w') as f:
    json.dump(train, f, indent=2)
with open('data/finetune/val.json', 'w') as f:
    json.dump(val, f, indent=2)

print(f'Train: {len(train)}, Val: {len(val)}')

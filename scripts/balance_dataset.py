import json, re, random

with open('data/finetune/raw_training_data.json') as f:
    data = json.load(f)

# Separate HIGH and UNVERIFIABLE
high = [d for d in data if json.loads(d['output']).get('confidence') == 'HIGH']
unverifiable = [d for d in data if json.loads(d['output']).get('confidence') == 'UNVERIFIABLE']
other = [d for d in data if json.loads(d['output']).get('confidence') not in ('HIGH', 'UNVERIFIABLE')]

print(f'HIGH: {len(high)}')
print(f'UNVERIFIABLE: {len(unverifiable)}')
print(f'OTHER: {len(other)}')

# Keep all HIGH, sample 300 UNVERIFIABLE (2.6x ratio is fine)
random.shuffle(unverifiable)
balanced = high + unverifiable[:300] + other
random.shuffle(balanced)

print(f'\nBalanced dataset: {len(balanced)} examples')
print(f'HIGH ratio: {len(high)/len(balanced)*100:.1f}%')
print(f'UNVERIFIABLE ratio: {300/len(balanced)*100:.1f}%')

with open('data/finetune/balanced_training_data.json', 'w') as f:
    json.dump(balanced, f, indent=2)
print('\nSaved to data/finetune/balanced_training_data.json')

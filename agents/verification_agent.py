import re

def normalize_number(val):
    if val is None:
        return set()
    s = str(val).replace(',', '').replace('$', '').strip()
    variants = set()
    try:
        f = float(s)
        variants.add(str(int(f)))           # 50332
        variants.add(f'{f:.1f}')            # 50332.0
        variants.add(f'{f:,.0f}')           # 50,332
        variants.add(str(f))                # 50332.0
        # Also try divided by 1000 and multiplied (billion/million confusion)
        variants.add(str(int(f/1000)))      # 50
        variants.add(str(int(f*1000)))      # 50332000
    except:
        variants.add(s)
    return variants

def find_number_in_text(value, text):
    if value is None:
        return False
    text_clean = text.replace(',', '').replace('$', '')
    for variant in normalize_number(value):
        if variant and len(variant) >= 2:
            pattern = re.compile(r'\b' + re.escape(variant) + r'\b')
            if pattern.search(text_clean):
                return True
    return False

def verification_agent(state: dict) -> dict:
    kpis = state['extracted_kpis']
    chunks_by_id = {c['chunk_id']: c.get('text', '') for c in state['raw_chunks']}

    verified, flagged = [], []
    for kpi in kpis:
        chunk_text = chunks_by_id.get(kpi.get('chunk_id'), '')
        value = kpi.get('value')
        source_quote = kpi.get('source_quote', '')

        num_check = find_number_in_text(value, chunk_text)
        quote_check = (
            source_quote.lower()[:25] in chunk_text.lower()
            if source_quote and len(source_quote) > 10
            else False
        )
        metric_check = (
            kpi.get('metric', '').lower()[:8] in chunk_text.lower()
            if kpi.get('metric')
            else False
        )

        if num_check or quote_check or (metric_check and value):
            kpi['verified'] = True
            verified.append(kpi)
        else:
            kpi['verified'] = False
            kpi['flag_reason'] = 'Number not found in source chunk'
            flagged.append(kpi)

    score = len(verified) / len(kpis) if kpis else 0.0
    return {
        **state,
        'verified_kpis': verified,
        'unverified_flags': [f.get('question', '') for f in flagged],
        'verification_score': round(score, 2)
    }

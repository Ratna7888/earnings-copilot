def narrative_agent(state: dict) -> dict:
    flags = []
    score = state.get('verification_score', 1.0)

    if score < 0.5:
        flags.append(f'Low verification score ({score:.0%}) — treat numbers with caution')
    if not state.get('extracted_kpis'):
        flags.append('No KPIs extracted — filing may lack financial tables in indexed chunks')

    return {**state, 'narrative_flags': flags, 'sentiment_delta': 'neutral'}

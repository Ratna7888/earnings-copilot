import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from agents.state import EarningsState
from agents.extraction_agent import extraction_agent
from agents.verification_agent import verification_agent
from agents.narrative_agent import narrative_agent
from agents.llm import chat
load_dotenv()

def memo_writer_agent(state: dict) -> dict:
    kpi_summary = '\n'.join([
        f"- {k['metric']}: {k['value']} {k.get('unit', '')} ({k.get('period', '')}) [Source: {k.get('chunk_id', '?')}]"
        for k in state['verified_kpis']
    ]) or 'No verified KPIs available.'

    flags = '\n'.join(
        state.get('narrative_flags', []) + state.get('unverified_flags', [])
    ) or 'None'

    prompt = f"""Write a concise investment memo for {state['ticker']} ({state['fiscal_period']}).

VERIFIED KPIs (citation-grounded only):
{kpi_summary}

FLAGS: {flags}
Verification score: {state.get('verification_score', 0):.0%}

Structure:
1. Bull Thesis (1 paragraph)
2. Bear Risks (1 paragraph)
3. Verdict (2-3 sentences)

Only reference verified KPIs. Never invent numbers."""

    try:
        memo = chat([{'role': 'user', 'content': prompt}], timeout=60)
    except Exception as e:
        memo = f'Memo generation failed: {e}\n\nVerified KPIs:\n{kpi_summary}'

    return {**state, 'investment_memo': memo}

def build_graph():
    g = StateGraph(EarningsState)
    g.add_node('extract', extraction_agent)
    g.add_node('narrate', narrative_agent)
    g.add_node('verify', verification_agent)
    g.add_node('write', memo_writer_agent)
    g.set_entry_point('extract')
    # No filing for the requested period: stop rather than write a memo from nothing
    g.add_conditional_edges('extract', lambda s: END if s.get('error') else 'narrate')
    g.add_edge('narrate', 'verify')
    g.add_edge('verify', 'write')
    g.add_edge('write', END)
    return g.compile()

if __name__ == '__main__':
    graph = build_graph()
    print('Running pipeline for AAPL Q3 2024...')
    result = graph.invoke({
        'ticker': 'AAPL',
        'fiscal_period': 'Q3 2024',
        'raw_chunks': [],
        'extracted_kpis': [],
        'narrative_flags': [],
        'unverified_flags': []
    })
    print('\n=== INVESTMENT MEMO ===')
    print(result['investment_memo'])
    print(f'\nVerification score: {result["verification_score"]}')
    print(f'Verified KPIs: {len(result["verified_kpis"])}')
    if result['verified_kpis']:
        print('\nKPIs found:')
        for k in result['verified_kpis']:
            print(f"  {k['metric']}: {k['value']} {k.get('unit','')} ({k.get('period','')})")

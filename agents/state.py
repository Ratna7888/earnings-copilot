from typing import TypedDict, List, Optional

class EarningsState(TypedDict):
    ticker: str
    fiscal_period: str

    # Extraction agent output
    raw_chunks: List[dict]
    extracted_kpis: List[dict]

    # Narrative agent output
    sentiment_delta: Optional[str]
    narrative_flags: List[str]

    # Verification agent output
    verified_kpis: List[dict]
    unverified_flags: List[str]
    verification_score: Optional[float]

    # Final output
    investment_memo: Optional[str]
    error: Optional[str]

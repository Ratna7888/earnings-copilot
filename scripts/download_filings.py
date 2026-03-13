from sec_edgar_downloader import Downloader
import os

dl = Downloader("YourName", "your@email.com", "data/raw")

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "JNJ", "V",
    "WMT", "PG", "UNH", "HD", "BAC",
    "XOM", "CVX", "LLY", "AVGO", "COST"
]

for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    dl.get("10-K", ticker, after="2020-01-01", before="2024-12-31")
    dl.get("10-Q", ticker, after="2023-01-01", before="2024-12-31")

print("Done! Check data/raw/")
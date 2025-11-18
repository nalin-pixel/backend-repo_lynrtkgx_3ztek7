import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests

from database import db, create_document, get_documents
from schemas import Signal

app = FastAPI(title="Trading Signals API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignalQuery(BaseModel):
    asset_type: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    limit: int = 20


@app.get("/")
def read_root():
    return {"message": "Trading Signals Backend Running"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# Simple technical analysis based signal generator using public market data
# We will fetch quotes from Binance for crypto and TwelveData (demo) for forex where possible

BINANCE_API = "https://api.binance.com/api/v3"
TWELVEDATA_API = "https://api.twelvedata.com"
TWELVEDATA_KEY = os.getenv("TWELVEDATA_API_KEY")  # optional, improves reliability for forex


def ema(values, period):
    if not values or len(values) < period:
        return None
    k = 2 / (period + 1)
    ema_val = values[0]
    for price in values[1:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val


def rsi(values, period=14):
    if len(values) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 200):
    url = f"{BINANCE_API}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Binance error: {r.text}")
    data = r.json()
    closes = [float(c[4]) for c in data]
    return closes, float(data[-1][4]) if data else ([], None)


def fetch_forex_klines(symbol: str, interval: str = "1h", limit: int = 200):
    # Try TwelveData; if no key, use demo throttle
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": limit,
        "apikey": TWELVEDATA_KEY or "demo",
    }
    r = requests.get(f"{TWELVEDATA_API}/time_series", params=params, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TwelveData error: {r.text}")
    data = r.json()
    if "values" not in data:
        raise HTTPException(status_code=502, detail=f"TwelveData response invalid: {data}")
    values = list(reversed(data["values"]))  # oldest first
    closes = [float(v["close"]) for v in values]
    last_price = closes[-1] if closes else None
    return closes, last_price


def generate_signal(asset_type: str, symbol: str, timeframe: str):
    try:
        if asset_type == "crypto":
            closes, last_price = fetch_binance_klines(symbol, timeframe)
        elif asset_type == "forex":
            closes, last_price = fetch_forex_klines(symbol, timeframe)
        else:
            raise HTTPException(status_code=400, detail="asset_type must be 'crypto' or 'forex'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Calculate indicators
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200)
    rsi14 = rsi(closes, 14)

    if not all([ema50, ema200, rsi14, last_price]):
        raise HTTPException(status_code=422, detail="Not enough data to generate a high-confidence signal")

    # Basic rules for high-confidence signals:
    # - Bullish when EMA50 > EMA200 and RSI between 50 and 65
    # - Bearish when EMA50 < EMA200 and RSI between 35 and 50
    # Avoid overbought/oversold extremes to reduce false signals
    signal_type = "neutral"
    confidence = 0.5
    reasons = []

    if ema50 > ema200:
        reasons.append("EMA50 above EMA200 (uptrend)")
        if 50 <= rsi14 <= 65:
            signal_type = "buy"
            confidence = 0.8
            reasons.append("RSI in healthy bullish zone (50-65)")
    elif ema50 < ema200:
        reasons.append("EMA50 below EMA200 (downtrend)")
        if 35 <= rsi14 <= 50:
            signal_type = "sell"
            confidence = 0.8
            reasons.append("RSI in healthy bearish zone (35-50)")

    # Slight adjustment with price position vs EMA50
    if last_price and ema50:
        if signal_type == "buy" and last_price >= ema50:
            confidence += 0.05
            reasons.append("Price above EMA50 supports trend")
        if signal_type == "sell" and last_price <= ema50:
            confidence += 0.05
            reasons.append("Price below EMA50 supports trend")

    confidence = min(max(confidence, 0.0), 0.95)

    return Signal(
        asset_type=asset_type,
        symbol=symbol,
        timeframe=timeframe,
        signal_type=signal_type,
        confidence=confidence,
        reason="; ".join(reasons) if reasons else "No high-confidence setup",
        price=float(last_price),
    )


@app.post("/api/signals/generate", response_model=Signal)
def api_generate_signal(query: SignalQuery):
    if not query.asset_type or not query.symbol or not query.timeframe:
        raise HTTPException(status_code=400, detail="asset_type, symbol, timeframe are required")
    signal = generate_signal(query.asset_type, query.symbol.upper(), query.timeframe)
    try:
        create_document("signal", signal)
    except Exception:
        pass
    return signal


@app.post("/api/signals", response_model=List[Signal])
def list_signals(query: SignalQuery):
    filt = {}
    if query.asset_type:
        filt["asset_type"] = query.asset_type
    if query.symbol:
        filt["symbol"] = query.symbol.upper()
    if query.timeframe:
        filt["timeframe"] = query.timeframe
    limit = max(1, min(query.limit or 20, 100))

    try:
        docs = get_documents("signal", filt, limit)
        results: List[Signal] = []
        for d in docs:
            results.append(
                Signal(
                    asset_type=d.get("asset_type"),
                    symbol=d.get("symbol"),
                    timeframe=d.get("timeframe"),
                    signal_type=d.get("signal_type"),
                    confidence=float(d.get("confidence", 0.0)),
                    reason=d.get("reason", ""),
                    price=float(d.get("price", 0.0)),
                    generated_at=d.get("generated_at"),
                )
            )
        return results
    except Exception:
        # If DB not available, return empty list
        return []


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

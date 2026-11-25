"""Downloads daily OHLCV (adjusted) for the extended BIST universe + XU100 + FX, cut at 2026-09-10."""
import yfinance as yf, pandas as pd
CORE30 = "ASELS TUPRS BIMAS THYAO AKBNK KCHOL EREGL TCELL CCOLA MGROS SASA TRALT FROTO ENKAI PGSUS MPARK EKGYO GUBRF TURSG ENJSA TTKOM OYAKC AKSEN ISMEN ARCLK ECILC MAVI PETKM TOASO KRDMD".split()
EXTRA = ("GARAN ISCTR YKBNK HALKB VAKBN TSKB SKBNK SAHOL DOHOL AGHOL ALARK TKFEN SOKM AEFES ULKER TUKAS TABGD "
         "TAVHL CLEBI DOAS OTKAR TTRAK BRSAN ODAS ZOREN CWENE EUPWR ASTOR SMRTG GESAN AKSA HEKTS AYGAZ CIMSA BTCIM "
         "AKCNS SISE KOZAA ANSGR AGESA VESTL KONTR MIATK REEDR ALFAS ENERY TTRAK GRSEL").split()
syms = [t + ".IS" for t in dict.fromkeys(CORE30 + EXTRA)] + ["XU100.IS", "USDTRY=X", "EURTRY=X"]
d = yf.download(syms, start="2023-06-01", end="2026-09-11", progress=False, auto_adjust=True, group_by="ticker", threads=True)
rows = []
for s in syms:
    try:
        x = d[s].dropna(how="all").reset_index()
    except KeyError:
        continue
    if len(x) < 100:
        print("skip", s, len(x)); continue
    x["ticker"] = s.replace(".IS", "").replace("=X", ""); rows.append(x)
df = pd.concat(rows); df.columns = [c.lower() for c in df.columns]
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
df = df[df.date <= "2026-09-10"]
df.to_parquet("data/prices.parquet")
print(df.ticker.nunique(), "series;", df.groupby("ticker").date.min().max().date(), "latest first date")
print(df.groupby("ticker").size().sort_values().head(8).to_string())

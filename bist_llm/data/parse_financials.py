"""Parses KAP financial-report PDFs (data_econ.rar → raw_fin/data_econ/<TICKER>/<YYYYQn>.pdf) into data/fundamentals.csv.

Extracted (current-period column, thousand TL): revenue (Hasılat), EBIT (Esas faaliyet karı), D&A (cash-flow adjustment),
net profit (Dönem karı), cash, short/long-term borrowings; disclosure date = board-approval date in the report.
Reports are cumulative (YTD); quarterly values = YTD difference within the same fiscal year (Q1 = YTD).
2024Q4 has no 9M-2024 report in the archive, so that row holds full-year 2024 values (flagged in `source`).
Caveat: under TMS 29 each report is restated to its own period-end purchasing power, so YTD differences are approximate.
Usage: python data/parse_financials.py <dir with .txt or .pdf files>"""
import re, subprocess, sys
from pathlib import Path
import pandas as pd

TR = str.maketrans({"ı": "i", "İ": "i", "I": "i", "â": "a", "Â": "a", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
                    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c", "î": "i", "û": "u"})
MONTHS = {"ocak": 1, "subat": 2, "mart": 3, "nisan": 4, "mayis": 5, "haziran": 6, "temmuz": 7, "agustos": 8,
          "eylul": 9, "ekim": 10, "kasim": 11, "aralik": 12}
NUM = re.compile(r"\(?-?\d{1,3}(?:\.\d{3})+\)?|\(?-?\d{4,}\)?")
DATE = re.compile(r"(\d{1,2})\s+(ocak|subat|mart|nisan|mayis|haziran|temmuz|agustos|eylul|ekim|kasim|aralik)\s+(\d{4})")


def norm(s):
    return s.translate(TR).lower()


TOK = re.compile(r"\(\s*-?\d{1,3}(?:\.\d{3})+\s*\)|\(\s*-?\d{4,}\s*\)|-?\d{1,3}(?:\.\d{3})+|-?\d{4,}|(?<!\S)[-–](?!\S)")


def first_number(line):
    """Value of the first (current-period) amount column. A lone '-' means zero; '(x)' means negative."""
    s = re.sub(r"^\s*[-–•*]\s+", "", line)          # drop a leading bullet dash
    for m in TOK.finditer(s):
        tok = m.group()
        if tok in ("-", "–"):
            return 0
        neg = tok.startswith("(") or tok.startswith("-")
        v = int(re.sub(r"[^\d]", "", tok))
        return -v if neg else v
    return None


def find(lines, pattern, start=0, exclude=None):
    rx = re.compile(pattern)
    for i in range(start, len(lines)):
        t = lines[i].strip().lstrip("-–•* ").strip()
        if rx.search(t) and not (exclude and re.search(exclude, t)):
            v = first_number(lines[i])
            if v is not None:
                return v, i
    return None, None


def approval_date(lines, period_end):
    lo, hi = period_end, period_end + pd.Timedelta(days=110)
    def dates_in(text):
        out = []
        for d, m, y in DATE.findall(text):
            try:
                ts = pd.Timestamp(int(y), MONTHS[m], int(d))
            except ValueError:
                continue
            if lo < ts <= hi:
                out.append(ts)
        return out
    for i, l in enumerate(lines):                         # 1) "Finansal tabloların onaylanması" section
        if re.search(r"finansal tablolarin onaylanmasi", l):
            ds = dates_in(" ".join(lines[i:i + 7]))
            if ds:
                return ds[0], "board_approval"
    for i, l in enumerate(lines):                         # 2) sentence: Yönetim Kurulu ... onaylanmış / yayımlanmak üzere
        win = " ".join(lines[max(0, i - 2):i + 2])
        if "onaylan" in l and ("yonetim kurulu" in win or "yayimlanmak uzere" in win) and "kar dagitim" not in win and "geri alim" not in win:
            ds = dates_in(win)
            if ds:
                return ds[0], "board_approval"
    ds = dates_in(" ".join(lines[:260]))                  # 3) auditor / review report date at the top
    if ds:
        return ds[0], "auditor_report_date"
    return None, "missing"


BORROW = {"st": r"kisa vadeli borclanmalar", "cur": r"uzun vadeli borclanmalarin kisa vadeli kis(i)?m",
          "lt": r"uzun vadeli borclanmalar(?!in)", "lease": r"kiralama (islemlerinden (kaynaklanan )?(borc|yukumluluk)|yukumluluk)"}
DETAIL = [r"banka kredileri", r"ihrac edilmis borclanma araclari", r"diger (uzun vadeli |kisa vadeli )?borclanmalar",
          r"finansal borclar$|finansal borclar\s"]


def borrowings(lines):
    """Sums interest-bearing debt on the balance sheet (current column). Header line value is used when present,
    otherwise the related-party / non-related-party sub-lines are summed. Leases are included (IFRS 16)."""
    start = next((i for i, l in enumerate(lines) if l.strip().lstrip("-–•* ").startswith("nakit ve nakit benzerleri") and first_number(l) is not None), 0)
    rev = next((i for i, l in enumerate(lines) if i > start and l.strip().lstrip("-–•* ").startswith("hasilat") and first_number(l) is not None), None)
    end = min(rev or start + 260, start + 260)
    got = {k: {"header": [], "sub": []} for k in BORROW}
    for l in lines[start:end]:
        t = l.strip().lstrip("-–•* ").strip()
        v = first_number(l)
        if v is None:
            continue
        for k, rx in BORROW.items():
            if re.match(rx, t):
                got[k]["header"].append(v); break
            if re.match(r"iliskili (olmayan )?taraflar(dan|a) " + rx, t):
                got[k]["sub"].append(v); break
    detail = []
    if not any(got[k]["header"] or got[k]["sub"] for k in ("st", "cur", "lt")):
        # header lines without amounts: sum the detail lines (bank loans, issued bonds, other borrowings)
        for l in lines[start:end]:
            t = l.strip().lstrip("-–•* ").strip()
            v = first_number(l)
            if v is not None and any(re.match(rx, t) for rx in DETAIL):
                detail.append(v)
    out = {"detail": sum(detail)}
    for k, g in got.items():
        # balance sheet lists current then non-current lines; for 'lease' both a short and a long line exist
        if k == "lease":
            out[k] = sum(g["header"][:2]) if g["header"] else sum(g["sub"][:4])
        else:
            out[k] = g["header"][0] if g["header"] else sum(g["sub"][:2])
    return out


def parse(text, ticker, q):
    lines = [norm(l) for l in text.splitlines()]
    y, qn = int(q[:4]), int(q[-1])
    pe = pd.Timestamp(y, 3 * qn, 1) + pd.offsets.MonthEnd(0)
    rev, ri = find(lines, r"^hasilat\b")
    if ri is not None and ri > 1500:                      # revenue only found deep in the notes: income statement unreadable
        rev, ri = None, None
    ebit, _ = find(lines, r"^esas faaliyet (kari|zarari|kar)", start=ri or 0)
    np_, _ = find(lines, r"^(surdurulen faaliyetler(den)? )?(net )?donem (net )?(kari|zarari|kar)\b", start=ri or 0,
                  exclude=r"vergi|dagilim|pay|hisse|diger kapsamli")
    da, _ = find(lines, r"^amortisman (ve itfa )?(gider\w*|pay\w*)\s*(ile |yle )?ilgili duzeltme")
    if da is None and ri is not None:                     # cash-flow line without the "ilgili düzeltmeler" wording
        da, _ = find(lines, r"^amortisman ve itfa (gider|pay)", start=ri)
    cash, _ = find(lines, r"^nakit ve nakit benzerleri$|^nakit ve nakit benzerleri\s")
    b = borrowings(lines)
    disc, how = approval_date(lines, pe)
    return {"ticker": ticker, "quarter": q, "quarter_end": pe.strftime("%Y-%m-%d"), "months": 3 * qn,
            "disclosure_date": disc.strftime("%Y-%m-%d") if disc is not None else None, "disclosure_source": how,
            "revenue_ytd": rev, "ebit_ytd": ebit, "da_ytd": abs(da) if da is not None else None, "net_profit_ytd": np_,
            "cash": cash, "st_borrowings": b["st"], "lt_current_portion": b["cur"], "lt_borrowings": b["lt"] + b["detail"],
            "lease_liabilities": b["lease"]}


def to_quarterly(p):
    p = p.sort_values(["ticker", "quarter_end"]).copy()
    p["ebitda_ytd"] = p.ebit_ytd + p.da_ytd            # NaN if D&A could not be read (row then unusable)
    p["net_debt"] = p.st_borrowings + p.lt_current_portion + p.lt_borrowings + p.lease_liabilities - p.cash
    p["ebitda_ltm"] = (p.ebitda_ytd * 12 / p.months).round(0)          # annualised YTD EBITDA
    rows = []
    for t, g in p.groupby("ticker"):
        g = g.set_index("quarter")
        for q, r in g.iterrows():
            prev = f"{q[:4]}Q{int(q[-1]) - 1}"
            note = "KAP report PDF; disclosure=" + r.disclosure_source
            if int(q[-1]) == 1:
                vals = (r.revenue_ytd, r.ebitda_ytd, r.net_profit_ytd)
            elif prev in g.index:
                pr = g.loc[prev]
                vals = (r.revenue_ytd - pr.revenue_ytd, r.ebitda_ytd - pr.ebitda_ytd, r.net_profit_ytd - pr.net_profit_ytd)
                note += "; quarter = YTD difference (TMS29 approx.)"
            else:
                vals = (r.revenue_ytd, r.ebitda_ytd, r.net_profit_ytd)
                note += f"; FULL-YEAR values ({r.months}M) - previous YTD report not in archive"
            rows.append({"ticker": t, "quarter_end": r.quarter_end, "disclosure_date": r.disclosure_date,
                         "revenue": vals[0], "ebitda": vals[1], "net_profit": vals[2], "net_debt": r.net_debt,
                         "ebitda_ltm": r.ebitda_ltm, "source": note})
    q = pd.DataFrame(rows).sort_values(["ticker", "quarter_end"])
    # LTM EBITDA: sum of the last 4 standalone quarters when available, full year for Q4 reports,
    # otherwise annualised YTD (flagged)
    p_idx = p.set_index(["ticker", "quarter_end"])
    ltm = []
    for t, g in q.groupby("ticker", sort=False):
        e = g.ebitda.tolist(); src = g.source.tolist()
        for i, (qe, r) in enumerate(zip(g.quarter_end, g.itertuples())):
            months = int(p_idx.loc[(t, qe), "months"])
            last4 = e[i - 3:i + 1] if i >= 3 else []
            if months == 12:
                ltm.append((p_idx.loc[(t, qe), "ebitda_ytd"], "LTM=full year"))
            elif len(last4) == 4 and not any(pd.isna(last4)) and not any("FULL-YEAR" in x for x in src[i - 3:i + 1]):
                ltm.append((sum(last4), "LTM=sum of last 4 quarters"))
            else:
                ltm.append((r.ebitda_ltm, "LTM=annualised YTD (approx.)"))
    q["ebitda_ltm"] = [round(v, 0) if pd.notna(v) else v for v, _ in ltm]
    q["source"] = [s_ + "; " + n for s_, (_, n) in zip(q.source, ltm)]
    return q


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "../raw_fin_txt")
    recs = []
    for f in sorted(src.glob("*.txt")) or sorted(src.glob("*/*.pdf")):
        if f.suffix == ".pdf":
            text = subprocess.run(["pdftotext", "-layout", str(f), "-"], capture_output=True, text=True).stdout
            ticker, q = f.parent.name, f.stem
        else:
            text = f.read_text(errors="ignore"); ticker, q = f.stem.split("_")
        if ticker == "AKBNK":       # bank statements: no revenue/EBITDA; industrial ratios not applicable
            continue
        recs.append(parse(text, ticker, q))
    p = pd.DataFrame(recs)
    out = Path(__file__).resolve().parent
    p.to_csv(out / "fundamentals_parsed_ytd.csv", index=False)
    qd = to_quarterly(p)
    qd.to_csv(out / "fundamentals.csv", index=False)
    print(p.drop(columns=["quarter_end"]).to_string(index=False))
    print("\nmissing values per column:\n", p.isna().sum()[p.isna().sum() > 0].to_string())

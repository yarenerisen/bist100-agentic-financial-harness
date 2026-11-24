"""Generates fine-tuning conversations: question -> tool call(s) -> REAL tool output -> evidence-based answer.
Answers are rule-based templates that copy numbers verbatim from tool outputs (no invented numbers).
Usage:  python -m training.generate_dataset            (from bist_llm/)
Outputs: training/train.jsonl, val.jsonl, test.jsonl, stats printed."""
import json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import core, macro_fund                       # noqa: E402
from tools.registry import call_tool                     # noqa: E402
from training.prompts import SYSTEM_PROMPT               # noqa: E402

OUT = ROOT / "training"
SYN_FUND = OUT / "synthetic_fundamentals_TRAINING_ONLY.csv"
rng = random.Random(42)
CUT = "2026-09-10"
XU_DATES = [d for d in core.get_ohlcv("XU100", "2024-01-01", CUT).index]
CAL = pd.DatetimeIndex(XU_DATES)
FULL = [t for t in core.available_tickers() if t not in ("XU100", "USDTRY", "EURTRY")
        and core.get_ohlcv(t, "2023-06-01", CUT).index[0] <= pd.Timestamp("2023-07-01")]
TEST_TICKERS = {"PGSUS", "TOASO", "KRDMD", "TTKOM", "ENJSA", "CIMSA", "ULKER", "TAVHL"}   # never in train/val single-stock questions
TRAIN_TICKERS = [t for t in FULL if t not in TEST_TICKERS]

SEC_EN = {"banking": "banking sector", "retail": "retail sector", "holding": "holding companies", "steel": "steel sector",
          "automotive": "automotive sector", "food_beverage": "food & beverage sector", "utilities": "electric utilities",
          "energy_equipment": "energy equipment sector", "chemicals": "chemicals sector", "petroleum": "petroleum / refining sector",
          "cement": "cement sector", "insurance": "insurance sector"}
SEC_TR = {"banking": "bankacılık", "retail": "perakende", "holding": "holding", "steel": "demir-çelik", "automotive": "otomotiv",
          "food_beverage": "gıda-içecek", "utilities": "elektrik", "energy_equipment": "enerji ekipmanları", "chemicals": "kimya",
          "petroleum": "petrol/rafineri", "cement": "çimento", "insurance": "sigorta"}

F = lambda v: "n/a" if v is None else json.dumps(v)          # verbatim number formatting
TC = lambda name, args: json.dumps({"tool": name, "arguments": args}, ensure_ascii=False)
TOOL_JSON = lambda r: json.dumps(r, ensure_ascii=False, separators=(",", ":"))
NOTE = {"en": "NOTE: Historical, educational analysis only; not investment advice and not a forecast.",
        "tr": "NOTE: Yalnızca tarihsel ve eğitim amaçlı analizdir; yatırım tavsiyesi veya tahmin değildir."}


def d_tr(d):  # 2026-09-10 -> 10.09.2026
    y, m, dd = d.split("-"); return f"{dd}.{m}.{y}"


def show_date(d, lang):
    return d_tr(d) if (lang == "tr" or rng.random() < 0.4) else d


def rand_date(lo="2024-09-02", hi=CUT, p_cut=0.45):
    if rng.random() < p_cut:
        return CUT
    c = CAL[(CAL >= pd.Timestamp(lo)) & (CAL <= pd.Timestamp(hi))]
    return str(rng.choice(list(c)).date())


def tdays_between(a, b):
    return int(((CAL > pd.Timestamp(a)) & (CAL <= pd.Timestamp(b))).sum())


class Conv:
    def __init__(self, lang, question):
        self.lang = lang
        self.msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]

    def call(self, name, args):
        r = call_tool(name, args)
        self.msgs.append({"role": "assistant", "content": TC(name, args)})
        self.msgs.append({"role": "tool", "content": TOOL_JSON(r)})
        return r

    def final(self, state, evidence, limitations):
        lines = [f"ANALYTICAL_STATE: {state}", "EVIDENCE:"] + [f"- E{i}: {e}" for i, e in enumerate(evidence, 1)]
        lines += [f"LIMITATIONS: {limitations}", NOTE[self.lang]]
        self.msgs.append({"role": "assistant", "content": "\n".join(lines)})
        return {"messages": self.msgs, "state": state}


def L(lang, en, tr):
    return en if lang == "en" else tr


def warn_text(lang, warns):
    w = [x for x in warns if x]
    if not w:
        return ""
    return L(lang, "Tool warnings: ", "Araç uyarıları: ") + "; ".join(w) + "."


# =============================================================== Scenario 1: sector laggard / catch-up
S1_Q = {
    "en": ["In the {sec} ({lst}), which stock lagged the sector median the most over the last {h} trading days up to {d}? Over {s} – {d}, did similar lags close within 5, 10 and 20 days? Can the stock be labeled POTENTIAL_CATCH_UP_CANDIDATE?",
           "Find the laggard among {lst} over the past {h} trading days as of {d} and check whether such laggards historically caught up.",
           "Which stock in the {sec} is furthest behind its peers over {h} trading days (as of {d})? Is there historical evidence of catch-up?",
           "As of {d}, is any of {lst} a sector catch-up candidate? Use a {h}-day window and verify with a backtest.",
           "Rank {lst} by {h}-day relative strength up to {d}. Did the weakest one tend to outperform its peers afterwards?",
           "{sec}: who lags the sector median over {h} days until {d}, and how reliable has mean reversion been since {s}?"],
    "tr": ["{sec} sektöründe ({lst}) {d} itibarıyla son {h} işlem gününde sektör medyanının en çok gerisinde kalan hisse hangisi? {s} – {d} arasında benzer gerilikler 5, 10 ve 20 günde kapandı mı? Hisse POTENTIAL_CATCH_UP_CANDIDATE olarak etiketlenebilir mi?",
           "{d} tarihinde {lst} arasında son {h} günde geride kalan hangisi? Tarihsel olarak bu tür geride kalanlar yetişmiş mi?",
           "{lst} içinde {h} günlük pencerede en zayıf hisseyi bul ve catch-up için backtest yap ({d} itibarıyla).",
           "Sektörüne göre geride kalan bir {sec} hissesi var mı? Tarih {d}, pencere {h} işlem günü.",
           "{lst} hisselerini {d} tarihine kadar {h} günlük göreceli güce göre sırala. En zayıf olan sonrasında akranlarını geçmiş mi?",
           "{sec} sektöründe {d} itibarıyla {h} günde sektör medyanının gerisinde kalan hisse hangisi, {s}'den beri ortalamaya dönüş ne kadar güvenilir?"],
}
S1_FUND = {"en": " Separate structural from temporary lag using fundamental and macro evidence.",
           "tr": " Geriliğin yapısal mı geçici mi olduğunu temel ve makro verilerle ayır."}


CATCHUP_POOL = []   # (sec, members, sector_arg, h, d) combos that produced POTENTIAL_CATCH_UP_CANDIDATE


def s1(lang, split, forced=None):
    if forced is None and split == "train" and CATCHUP_POOL and rng.random() < 0.1:
        forced = rng.choice(CATCHUP_POOL)
    if forced:
        sec, members, sector_arg, h, d = forced
        want_fund = False
    else:
        secs = [s for s in SEC_EN if len(core.SECTORS[s]) >= 3]
        sec = rng.choice(secs)
        members = [t for t in core.SECTORS[sec] if t in core.available_tickers()]
        if len(members) > 5 or rng.random() < 0.5:
            k = min(len(members), rng.choice([3, 3, 4, 5]))
            members = rng.sample(members, k)
            sector_arg = members
        else:
            sector_arg = sec
        h = rng.choice([20, 20, 20, 10, 30])
        d = rand_date("2024-12-02")
        want_fund = rng.random() < 0.4
    s = "2024-01-01"
    params = (sec, members, sector_arg, h, d)
    qt = rng.choice(S1_Q[lang][:4] if split == "train" else S1_Q[lang])
    q = qt.format(sec=(SEC_EN if lang == "en" else SEC_TR)[sec], lst=", ".join(members), h=h,
                  d=show_date(d, lang), s=show_date(s, lang)) + (S1_FUND[lang] if want_fund else "")
    c = Conv(lang, q)
    rk = c.call("rank_sector_relative_strength", {"sector": sector_arg, "horizon": h, "as_of": d})
    if "error" in rk:
        return None
    bt = c.call("run_backtest", {"strategy": "sector_catch_up", "sector": sector_arg, "start": s, "end": d, "lag_window": h})
    if "error" in bt:
        return None
    w = rk["weakest"]; wr = rk["ranking"][0]
    res = bt["results"][w]["by_horizon"]
    fund = macro = None
    if want_fund:
        fund = c.call("get_fundamental_metrics", {"ticker": w, "start": s, "end": d})
        macro = c.call("get_macro_context", {"start": s, "end": d})
    lag = wr["lag_vs_median_pp"]; thr = bt["lag_threshold_pp"]
    r20 = res["20d"]; pb = bt["portfolio_backtest"]; perf = {p["series"]: p for p in pb["performance"]}
    tech_ok = wr["supertrend"] == "bullish" or wr["close_above_kama"]
    # ---- labeling rules (see labeling_rules.md)
    if lag > -thr:
        state = "WATCH"; why = L(lang, f"the lag of {F(lag)} pp is smaller than the {F(thr)} pp threshold, so there is no material laggard",
                                f"{F(lag)} puanlık gerilik {F(thr)} puanlık eşikten küçük, belirgin bir geride kalan yok")
    elif not rk["sector_strong"]:
        state = "WATCH"; why = L(lang, "the sector itself is not strong (sector_strong = false), so the catch-up premise does not hold",
                                 "sektörün kendisi güçlü değil (sector_strong = false), catch-up varsayımı sağlanmıyor")
    elif r20["events"] < 10:
        state = "INVESTIGATE"; why = L(lang, f"only {F(r20['events'])} comparable 20-day events exist, too few to verify", f"yalnızca {F(r20['events'])} karşılaştırılabilir 20 günlük olay var, doğrulamak için az")
    elif (r20["hit_rate_pct"] or 0) >= 55 and (r20["mean_after_cost_pp"] or 0) > 0 and tech_ok:
        state = "POTENTIAL_CATCH_UP_CANDIDATE"; why = L(lang, "the sector is strong, the lag exceeds the threshold, past lags were mostly followed by outperformance after costs and the technical trend is stabilising",
                                                        "sektör güçlü, gerilik eşiğin üzerinde, geçmiş geriliklerin çoğu maliyet sonrası akranları geçmiş ve teknik trend toparlanıyor")
    elif (r20["hit_rate_pct"] or 0) < 45:
        state = "REJECT_SIGNAL"; why = L(lang, f"historically the 20-day hit rate was only {F(r20['hit_rate_pct'])}%, laggards mostly kept underperforming",
                                         f"tarihsel 20 günlük isabet oranı yalnızca %{F(r20['hit_rate_pct'])}, geride kalanlar çoğunlukla zayıf kalmaya devam etmiş")
    else:
        state = "WATCH"; why = L(lang, "historical evidence is mixed or the technical trend has not stabilised", "tarihsel kanıt karışık veya teknik trend henüz toparlanmadı")
    ev = [L(lang, f"Over {F(rk['horizon_trading_days'])} trading days ({rk['window_start']} to {rk['window_end']}) the peer median return was {F(rk['sector_median_return_pct'])}% and XU100 returned {F(rk['xu100_return_pct'])}%; sector_strong = {str(rk['sector_strong']).lower()}.",
              f"{F(rk['horizon_trading_days'])} işlem gününde ({rk['window_start']} – {rk['window_end']}) akran medyan getirisi %{F(rk['sector_median_return_pct'])}, XU100 getirisi %{F(rk['xu100_return_pct'])}; sector_strong = {str(rk['sector_strong']).lower()}."),
          L(lang, f"{w} was the weakest: return {F(wr['return_pct'])}%, {F(lag)} pp vs the peer median and {F(wr['vs_xu100_pp'])} pp vs XU100. Strongest: {rk['ranking'][-1]['ticker']} at {F(rk['ranking'][-1]['lag_vs_median_pp'])} pp.",
              f"En zayıf hisse {w}: getiri %{F(wr['return_pct'])}, medyana göre {F(lag)} puan, XU100'e göre {F(wr['vs_xu100_pp'])} puan. En güçlü: {rk['ranking'][-1]['ticker']} ({F(rk['ranking'][-1]['lag_vs_median_pp'])} puan)."),
          L(lang, f"Historical lag events for {w} (lag ≥ {F(thr)} pp, {bt['period'][0]} to {bt['period'][1]}): hit rate vs peers {F(res['5d']['hit_rate_pct'])}% at 5d ({F(res['5d']['events'])} events), {F(res['10d']['hit_rate_pct'])}% at 10d ({F(res['10d']['events'])}), {F(r20['hit_rate_pct'])}% at 20d ({F(r20['events'])}); full gap closed within 20d in {F(r20['gap_closed_rate_pct'])}% of cases.",
              f"{w} için geçmiş gerilik olayları (gerilik ≥ {F(thr)} puan, {bt['period'][0]} – {bt['period'][1]}): akranları geçme oranı 5 günde %{F(res['5d']['hit_rate_pct'])} ({F(res['5d']['events'])} olay), 10 günde %{F(res['10d']['hit_rate_pct'])} ({F(res['10d']['events'])}), 20 günde %{F(r20['hit_rate_pct'])} ({F(r20['events'])}); farkın tamamı 20 günde %{F(r20['gap_closed_rate_pct'])} oranında kapanmış."),
          L(lang, f"False positives (laggard kept underperforming): {F(res['5d']['false_positives'])} at 5d, {F(res['10d']['false_positives'])} at 10d, {F(r20['false_positives'])} at 20d; mean 20d relative return after costs {F(r20['mean_after_cost_pp'])} pp.",
              f"Yanlış pozitifler (geride kalmaya devam eden): 5 günde {F(res['5d']['false_positives'])}, 10 günde {F(res['10d']['false_positives'])}, 20 günde {F(r20['false_positives'])}; maliyet sonrası ortalama 20 günlük göreceli getiri {F(r20['mean_after_cost_pp'])} puan."),
          L(lang, f"Technical state of {w}: Supertrend {wr['supertrend']}, close above KAMA = {str(wr['close_above_kama']).lower()}.",
              f"{w} teknik durumu: Supertrend {wr['supertrend']}, kapanış KAMA üzerinde = {str(wr['close_above_kama']).lower()}."),
          L(lang, f"Portfolio backtest of the catch-up rule ({pb['execution']}): {F(pb['trades'])} trades, win rate {F(pb['win_rate_pct'])}%, cumulative {F(perf['strategy']['cumulative_return_pct'])}% vs buy-and-hold {F(perf['buy_and_hold_equal_weight']['cumulative_return_pct'])}% and XU100 {F(perf['XU100']['cumulative_return_pct'])}%; Sharpe {F(perf['strategy']['sharpe_ratio'])}, max drawdown {F(perf['strategy']['max_drawdown_pct'])}%.",
              f"Catch-up kuralının portföy backtesti ({pb['execution']}): {F(pb['trades'])} işlem, kazanma oranı %{F(pb['win_rate_pct'])}, kümülatif %{F(perf['strategy']['cumulative_return_pct'])}; buy-and-hold %{F(perf['buy_and_hold_equal_weight']['cumulative_return_pct'])}, XU100 %{F(perf['XU100']['cumulative_return_pct'])}; Sharpe {F(perf['strategy']['sharpe_ratio'])}, maksimum düşüş %{F(perf['strategy']['max_drawdown_pct'])}.")]
    lim = [L(lang, f"Conclusion: {why}.", f"Sonuç: {why}.")]
    if fund is not None:
        if fund.get("quarters"):
            q0 = fund["quarters"][-1]
            ev.append(L(lang, f"Latest disclosed quarter for {w} ({q0['quarter_end']}, disclosed {q0['disclosure_date']}): EBITDA margin {F(q0['ebitda_margin_pct'])}%, Net Debt/EBITDA {F(q0['net_debt_to_ebitda'])}.",
                        f"{w} için açıklanan son çeyrek ({q0['quarter_end']}, KAP {q0['disclosure_date']}): FAVÖK marjı %{F(q0['ebitda_margin_pct'])}, Net Borç/FAVÖK {F(q0['net_debt_to_ebitda'])}."))
        else:
            lim.append(L(lang, f"Fundamental data for {w} is not loaded, so structural vs temporary lag cannot be separated.",
                         f"{w} için temel veri yüklü değil; geriliğin yapısal mı geçici mi olduğu ayrılamıyor."))
            if state == "POTENTIAL_CATCH_UP_CANDIDATE":
                state = "INVESTIGATE"
                lim.append(L(lang, "The label is downgraded to INVESTIGATE until fundamentals are checked.", "Temel veriler kontrol edilene kadar etiket INVESTIGATE'e düşürüldü."))
        fx = macro["series"].get("USDTRY")
        pr = macro.get("tcmb_policy_rate")
        if fx and pr:
            ev.append(L(lang, f"Macro context {macro['period'][0]} to {macro['period'][1]}: USD/TRY changed {F(fx['change_pct'])}%; TCMB policy rate moved from {F(pr['level_at_start'])}% to {F(pr['level_at_end'])}%.",
                        f"Makro bağlam {macro['period'][0]} – {macro['period'][1]}: USD/TRY %{F(fx['change_pct'])} değişti; TCMB politika faizi %{F(pr['level_at_start'])} seviyesinden %{F(pr['level_at_end'])} seviyesine geldi."))
    lim.append(warn_text(lang, rk.get("warnings", []) + bt.get("warnings", []) + pb.get("warnings", [])))
    lim.append(L(lang, "Lag events overlap in time across stocks; the backtest is in-sample.", "Gerilik olayları hisseler arasında zamanda örtüşebilir; backtest örneklem içidir."))
    out = c.final(state, ev, " ".join(x for x in lim if x))
    out["params"] = params
    return out


# =============================================================== Scenario 2: weekday & multi-day patterns
DAY_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma"]


def pat_desc(p, lang):
    names = DAY_EN if lang == "en" else DAY_TR
    up, dn = L(lang, "up", "yükseliş"), L(lang, "down", "düşüş")
    parts = [f"{names[i]} {up if c == 'U' else dn}" for i, c in enumerate(p) if c != "X"]
    return ", ".join(parts)


S2_Q = {"en": ["Does {t} show a '{desc}' pattern over {a} – {b}? Does the pattern still hold over {c} – {d}?",
               "Test the weekday pattern '{desc}' for {t}: in-sample {a} to {b}, out-of-sample {c} to {d}.",
               "Is there a reliable '{desc}' weekly pattern in {t}? Check {a}–{b} and then {c}–{d}.",
               "For {t}, compare the '{desc}' pattern in {a} – {b} with {c} – {d}, including transaction costs.",
               "Is the {t} '{desc}' weekday effect stable out of sample ({c} – {d}) after being found in {a} – {b}?"],
        "tr": ["{t} hissesi {a} – {b} döneminde '{desc}' deseni gösteriyor mu? Desen {c} – {d} döneminde hâlâ geçerli mi?",
               "{t} için '{desc}' haftalık desenini test et: örneklem içi {a} – {b}, örneklem dışı {c} – {d}.",
               "{t} hissesinde güvenilir bir '{desc}' gün deseni var mı? Önce {a}–{b}, sonra {c}–{d} dönemine bak.",
               "{t} için '{desc}' desenini {a} – {b} ve {c} – {d} dönemlerinde işlem maliyetiyle karşılaştır.",
               "{a} – {b} döneminde bulunan {t} '{desc}' etkisi örneklem dışında ({c} – {d}) sürüyor mu?"]}
S2M_Q = {"en": ["After {n} consecutive down days, does {t} tend to rebound over the next {h} days ({a} – {b})?",
                "Does {t} mean-revert after a daily drop of {x}% or more? Use a {h}-day horizon over {a} – {b}.",
                "After {n} consecutive up days, does {t} continue or reverse over the next {h} days ({a} – {b})?"],
         "tr": ["{t} art arda {n} gün düştükten sonra sonraki {h} günde toparlanma eğiliminde mi ({a} – {b})?",
                "{t} günlük %{x} veya daha fazla düştüğü günlerin ardından {h} günde geri dönüyor mu ({a} – {b})?",
                "{t} art arda {n} gün yükseldikten sonra sonraki {h} günde devam mı ediyor, dönüyor mu ({a} – {b})?"]}


def s2(lang, split, tickers):
    t = rng.choice(tickers)
    if rng.random() < 0.3:
        return s2_multi(lang, split, t)
    p = rng.choices(["DDDUU", "UUUDD", "DXXXU", "UXXXD", "DDXUU", "XXXDU"], [5, 2, 2, 1, 1, 1])[0]
    if rng.random() < 0.6:
        a, b, cc, d = "2024-01-01", "2025-12-31", "2026-01-01", CUT
    else:
        a, b, cc, d = "2024-01-01", "2024-12-31", "2025-01-01", CUT
    qt = rng.choice(S2_Q[lang][:3] if split == "train" else S2_Q[lang])
    q = qt.format(t=t, desc=pat_desc(p, lang), a=show_date(a, lang), b=show_date(b, lang), c=show_date(cc, lang), d=show_date(d, lang))
    c = Conv(lang, q)
    i = c.call("test_weekday_pattern", {"ticker": t, "pattern": p, "start": a, "end": b})
    o = c.call("test_weekday_pattern", {"ticker": t, "pattern": p, "start": cc, "end": d})
    if "error" in i or "error" in o:
        return None
    sig_is = i["binomial_p_value_one_sided"] <= 0.0016 and i["pattern_occurrences"] >= 10
    if sig_is and (o["tradable_thu_fri_mean_after_cost_pct"] or -1) > 0 and o["binomial_p_value_one_sided"] < 0.05:
        state, why = "WATCH", L(lang, "the pattern was significant in-sample and persisted out-of-sample, but samples are small", "desen örneklem içinde anlamlı ve örneklem dışında sürdü, ancak örneklemler küçük")
    elif sig_is:
        state, why = "INVESTIGATE", L(lang, "the in-sample pattern did not clearly persist out-of-sample", "örneklem içi desen örneklem dışında net biçimde sürmedi")
    else:
        state, why = "REJECT_SIGNAL", L(lang, "the pattern is not statistically stable: it is not significant after multiple-testing correction", "desen istatistiksel olarak kararlı değil: çoklu test düzeltmesinden sonra anlamlı değil")
    def blk(r, tag):
        return L(lang, f"{tag} {r['period'][0]} to {r['period'][1]}: {F(r['pattern_occurrences'])} occurrences in {F(r['complete_weeks'])} complete weeks ({F(r['pattern_frequency_pct'])}%) vs {F(r['expected_frequency_if_independent_pct'])}% expected if days were independent; one-sided binomial p = {F(r['binomial_p_value_one_sided'])}.",
                 f"{tag} {r['period'][0]} – {r['period'][1]}: {F(r['complete_weeks'])} tam haftada {F(r['pattern_occurrences'])} kez (%{F(r['pattern_frequency_pct'])}); günler bağımsız olsaydı beklenen %{F(r['expected_frequency_if_independent_pct'])}; tek yönlü binom p = {F(r['binomial_p_value_one_sided'])}.")
    def trd(r, tag):
        return L(lang, f"{tag} tradable version (signal from the Mon–Wed part, outcome Thu+Fri): {F(r['tradable_signal_weeks'])} signal weeks, mean {F(r['tradable_thu_fri_mean_pct'])}%, median {F(r['tradable_thu_fri_median_pct'])}%, hit rate {F(r['tradable_thu_fri_hit_rate_pct'])}%, mean after costs {F(r['tradable_thu_fri_mean_after_cost_pct'])}% vs unconditional Thu+Fri mean {F(r['unconditional_thu_fri_mean_pct'])}%.",
                 f"{tag} işlem yapılabilir versiyon (Pzt–Çar kısmından sinyal, sonuç Per+Cum): {F(r['tradable_signal_weeks'])} sinyal haftası, ortalama %{F(r['tradable_thu_fri_mean_pct'])}, medyan %{F(r['tradable_thu_fri_median_pct'])}, isabet %{F(r['tradable_thu_fri_hit_rate_pct'])}, maliyet sonrası ortalama %{F(r['tradable_thu_fri_mean_after_cost_pct'])}; koşulsuz Per+Cum ortalaması %{F(r['unconditional_thu_fri_mean_pct'])}.")
    tin, tout = L(lang, "In-sample", "Örneklem içi"), L(lang, "Out-of-sample", "Örneklem dışı")
    ws = i["weekday_stats"]
    ev = [blk(i, tin), trd(i, tin), blk(o, tout), trd(o, tout),
          L(lang, f"In-sample mean daily returns: Mon {F(ws['Mon']['mean_pct'])}%, Tue {F(ws['Tue']['mean_pct'])}%, Wed {F(ws['Wed']['mean_pct'])}%, Thu {F(ws['Thu']['mean_pct'])}%, Fri {F(ws['Fri']['mean_pct'])}%.",
            f"Örneklem içi ortalama günlük getiriler: Pzt %{F(ws['Mon']['mean_pct'])}, Sal %{F(ws['Tue']['mean_pct'])}, Çar %{F(ws['Wed']['mean_pct'])}, Per %{F(ws['Thu']['mean_pct'])}, Cum %{F(ws['Fri']['mean_pct'])}.")]
    lim = " ".join(x for x in [L(lang, f"Conclusion: {why}.", f"Sonuç: {why}."),
                               L(lang, "Multiple-testing risk: there are 32 possible Mon–Fri up/down patterns, so the Bonferroni threshold is p < 0.0016; searching many patterns produces false discoveries by chance.",
                                 "Çoklu test riski: 32 olası Pzt–Cum yukarı/aşağı deseni var, Bonferroni eşiği p < 0.0016; çok sayıda desen aramak tesadüfen sahte bulgular üretir."),
                               L(lang, "Full-pattern weeks condition on the Thu–Fri outcome and are descriptive only.", "Tam desen haftaları Per–Cum sonucunu içerdiği için yalnızca betimseldir."),
                               warn_text(lang, i["warnings"] + o["warnings"])] if x)
    return c.final(state, ev, lim)


def s2_multi(lang, split, t):
    kind = rng.choice(["consecutive_down", "large_down_day", "consecutive_up"])
    n = rng.choice([2, 3, 3, 4]); h = rng.choice([1, 2, 3, 5]); x = rng.choice([-3.0, -4.0, -5.0])
    a, b = "2024-01-01", rand_date("2025-06-02")
    qi = {"consecutive_down": 0, "large_down_day": 1, "consecutive_up": 2}[kind]
    q = S2M_Q[lang][qi].format(t=t, n=n, h=h, x=F(abs(x)), a=show_date(a, lang), b=show_date(b, lang))
    c = Conv(lang, q)
    args = {"ticker": t, "pattern_type": kind, "horizon": h, "start": a, "end": b}
    args.update({"n_days": n} if "consecutive" in kind else {"threshold_pct": x})
    r = c.call("detect_multiday_patterns", args)
    if "error" in r:
        return None
    done = r["events"] - r["excluded_incomplete_window"]
    if done < 10:
        state, why = "INVESTIGATE", L(lang, "too few events to judge", "değerlendirmek için olay sayısı az")
    elif (r["up_rate_pct"] or 0) > r["unconditional_up_rate_pct"] and (r["mean_after_cost_pct"] or 0) > 0:
        state, why = "WATCH", L(lang, "the conditional forward return was higher than unconditional, but this is descriptive and overlapping", "koşullu ileri getiri koşulsuzdan yüksek, ancak bu betimsel ve olaylar örtüşüyor")
    else:
        state, why = "REJECT_SIGNAL", L(lang, "the pattern did not beat the unconditional baseline after costs", "desen maliyet sonrası koşulsuz ortalamayı geçmedi")
    ev = [L(lang, f"{F(r['events'])} events between {r['period'][0]} and {r['period'][1]} ({F(r['excluded_incomplete_window'])} excluded for incomplete windows).",
            f"{r['period'][0]} – {r['period'][1]} arasında {F(r['events'])} olay ({F(r['excluded_incomplete_window'])} tanesi eksik pencere nedeniyle hariç)."),
          L(lang, f"Mean {F(r['horizon_days'])}-day forward return {F(r['mean_forward_return_pct'])}% (median {F(r['median_forward_return_pct'])}%), up in {F(r['up_rate_pct'])}% of cases; after costs {F(r['mean_after_cost_pct'])}%.",
            f"Ortalama {F(r['horizon_days'])} günlük ileri getiri %{F(r['mean_forward_return_pct'])} (medyan %{F(r['median_forward_return_pct'])}), olayların %{F(r['up_rate_pct'])}'inde yükseliş; maliyet sonrası %{F(r['mean_after_cost_pct'])}."),
          L(lang, f"Unconditional baseline: mean {F(r['unconditional_mean_forward_return_pct'])}%, up rate {F(r['unconditional_up_rate_pct'])}%.",
            f"Koşulsuz karşılaştırma: ortalama %{F(r['unconditional_mean_forward_return_pct'])}, yükseliş oranı %{F(r['unconditional_up_rate_pct'])}.")]
    lim = " ".join(x for x in [L(lang, f"Conclusion: {why}.", f"Sonuç: {why}."), warn_text(lang, r["warnings"]),
                               L(lang, "Multiple-testing risk applies when several patterns are tried.", "Birden fazla desen denendiğinde çoklu test riski geçerlidir.")] if x)
    return c.final(state, ev, lim)


# =============================================================== Scenario 3: technical reversal events
EV_EN = {"lower_bollinger_touch": "lower Bollinger Band touches", "upper_bollinger_touch": "upper Bollinger Band touches",
         "middle_bollinger_test": "middle Bollinger Band tests", "rsi_trough": "RSI troughs (RSI back above 30)",
         "rsi_peak": "RSI peaks (RSI back below 70)", "supertrend_bullish_flip": "bullish Supertrend flips",
         "supertrend_bearish_flip": "bearish Supertrend flips", "kama_cross_up": "upward KAMA crosses",
         "kama_cross_down": "downward KAMA crosses", "ichimoku_cloud_breakout": "Ichimoku cloud breakouts",
         "ichimoku_cloud_breakdown": "Ichimoku cloud breakdowns"}
EV_TR = {"lower_bollinger_touch": "alt Bollinger bandı temasları", "upper_bollinger_touch": "üst Bollinger bandı temasları",
         "middle_bollinger_test": "orta Bollinger bandı testleri", "rsi_trough": "RSI dipleri (RSI'ın 30 üstüne dönmesi)",
         "rsi_peak": "RSI tepeleri (RSI'ın 70 altına dönmesi)", "supertrend_bullish_flip": "Supertrend'in yükselişe dönmesi",
         "supertrend_bearish_flip": "Supertrend'in düşüşe dönmesi", "kama_cross_up": "KAMA'nın yukarı kesilmesi",
         "kama_cross_down": "KAMA'nın aşağı kesilmesi", "ichimoku_cloud_breakout": "Ichimoku bulut yukarı kırılımları",
         "ichimoku_cloud_breakdown": "Ichimoku bulut aşağı kırılımları"}
S3_BB = {"en": ["How many times did {t} touch the lower Bollinger Band over {a} – {b}? In how many cases did the price rise within 5 days? Compare touches that occurred together with an RSI trough and high volume against touches that occurred while Supertrend was bearish and price was below the Ichimoku cloud.",
                "Analyse lower Bollinger Band touches for {t} between {a} and {b}: success rate over 5 days, average reversal strength over 1/3/5/10 days, and the effect of RSI/volume and Supertrend/Ichimoku conditions.",
                "{t}: count lower-band touches ({a} – {b}) and tell me how often a rebound followed. Do combined conditions work better than the single indicator?"],
         "tr": ["{t} {a} – {b} arasında alt Bollinger bandına kaç kez değdi? Kaçında fiyat 5 gün içinde yükseldi? RSI dibi ve yüksek hacimle birlikte gelen temasları, Supertrend düşüşteyken ve fiyat Ichimoku bulutunun altındayken gelen temaslarla karşılaştır.",
                "{t} için {a} – {b} arasındaki alt Bollinger temaslarını analiz et: 5 günlük başarı oranı, 1/3/5/10 günlük ortalama dönüş gücü ve RSI/hacim ile Supertrend/Ichimoku koşullarının etkisi.",
                "{t}: {a} – {b} arasında alt banda kaç temas oldu, kaçında toparlanma geldi? Birleşik koşullar tek indikatörden daha mı iyi?"]}
S3_GEN = {"en": ["How did {t} behave after {ev} between {a} and {b}? Give success counts, forward returns and compare market regimes.",
                 "Are {ev} a reliable reversal signal for {t} over {a} – {b}?",
                 "Count {ev} for {t} ({a} to {b}) and measure what happened over the next 1, 3, 5 and 10 days."],
          "tr": ["{t} hissesi {a} – {b} arasında {ev} sonrasında nasıl davrandı? Başarı sayıları, ileri getiriler ve piyasa rejimlerini karşılaştır.",
                 "{a} – {b} döneminde {t} için {ev} güvenilir bir dönüş sinyali mi?",
                 "{t} için {a} – {b} arasındaki {ev} sayısını bul ve sonraki 1, 3, 5, 10 günde ne olduğunu ölç."]}


def s3(lang, split, tickers):
    t = rng.choice(tickers)
    event = rng.choices(list(EV_EN), [8, 3, 2, 3, 2, 3, 2, 2, 1, 2, 1])[0]
    a = rng.choice(["2024-01-01", "2024-01-01", "2024-06-01", "2025-01-01"])
    b = rand_date("2025-09-01", p_cut=0.6)
    if event == "lower_bollinger_touch" and rng.random() < 0.7:
        qt = rng.choice(S3_BB[lang][:2] if split == "train" else S3_BB[lang])
        q = qt.format(t=t, a=show_date(a, lang), b=show_date(b, lang))
    else:
        qt = rng.choice(S3_GEN[lang][:2] if split == "train" else S3_GEN[lang])
        q = qt.format(t=t, ev=(EV_EN if lang == "en" else EV_TR)[event], a=show_date(a, lang), b=show_date(b, lang))
    c = Conv(lang, q)
    ind = c.call("calculate_indicators", {"ticker": t, "start": a, "end": b})
    r = c.call("detect_reversal_events", {"ticker": t, "event": event, "start": a, "end": b})
    if "error" in r or "error" in ind:
        return None
    g = r["groups"]; al = g["all_events"]
    n = al["events_with_complete_window"]; sr = al.get("success_rate_pct")
    recent = r["last_event_dates"] and tdays_between(r["last_event_dates"][-1], r["period"][1]) <= 10
    if n < 10:
        state, why = "INVESTIGATE", L(lang, f"only {F(n)} events have a complete forward window, too few for a conclusion", f"tam ileri penceresi olan yalnızca {F(n)} olay var, sonuç için az")
    elif sr >= 60 and recent:
        state, why = "TECHNICAL_REVERSAL_CANDIDATE", L(lang, f"the event has historically succeeded in {F(sr)}% of cases and a new event occurred recently ({r['last_event_dates'][-1]})",
                                                       f"olay tarihsel olarak %{F(sr)} oranında başarılı oldu ve yakın zamanda yeni bir olay gerçekleşti ({r['last_event_dates'][-1]})")
    elif sr >= 60:
        state, why = "WATCH", L(lang, f"the event historically succeeded in {F(sr)}% of cases but there is no recent event", f"olay tarihsel olarak %{F(sr)} oranında başarılı, ancak yakın zamanda olay yok")
    elif sr < 50:
        state, why = "REJECT_SIGNAL", L(lang, f"the success rate of {F(sr)}% is not better than a coin flip", f"%{F(sr)} başarı oranı yazı-turadan iyi değil")
    else:
        state, why = "WATCH", L(lang, "results are mixed", "sonuçlar karışık")
    def gl(name, label):
        s = g[name]
        if s["events_with_complete_window"] == 0:
            return L(lang, f"{label}: {F(s['events'])} events, none with a complete window.", f"{label}: {F(s['events'])} olay, tam penceresi olan yok.")
        return L(lang, f"{label}: {F(s['events_with_complete_window'])} complete events, {F(s['success_count'])} successes / {F(s['failure_count'])} failures ({F(s['success_rate_pct'])}%), mean 5d return {F(s['mean_return_5d_pct'])}%.",
                 f"{label}: tam penceresi olan {F(s['events_with_complete_window'])} olay, {F(s['success_count'])} başarı / {F(s['failure_count'])} başarısız (%{F(s['success_rate_pct'])}), ortalama 5 günlük getiri %{F(s['mean_return_5d_pct'])}.")
    ev = [L(lang, f"Indicator state at {ind['as_of']}: close {F(ind['close'])}, RSI14 {F(ind['rsi14'])}, Supertrend {ind['supertrend']}, Ichimoku {ind['ichimoku_position']}, Bollinger lower/mid/upper {F(ind['bb_lower'])}/{F(ind['bb_mid'])}/{F(ind['bb_upper'])}.",
            f"{ind['as_of']} itibarıyla indikatörler: kapanış {F(ind['close'])}, RSI14 {F(ind['rsi14'])}, Supertrend {ind['supertrend']}, Ichimoku {ind['ichimoku_position']}, Bollinger alt/orta/üst {F(ind['bb_lower'])}/{F(ind['bb_mid'])}/{F(ind['bb_upper'])}."),
          L(lang, f"{F(al['events'])} {EV_EN[event]} between {r['period'][0]} and {r['period'][1]} ({F(al['excluded_incomplete_window'])} excluded because the forward window is incomplete). Of {F(n)} complete events, {F(al.get('success_count'))} moved in the expected direction ({r['expected_direction']}) within 5 days and {F(al.get('failure_count'))} did not ({F(sr)}%).",
            f"{r['period'][0]} – {r['period'][1]} arasında {F(al['events'])} olay ({EV_TR[event]}); {F(al['excluded_incomplete_window'])} tanesi ileri pencere tamamlanmadığı için hariç. Tam penceresi olan {F(n)} olayın {F(al.get('success_count'))} tanesinde fiyat 5 gün içinde beklenen yönde ({r['expected_direction']}) hareket etti, {F(al.get('failure_count'))} tanesinde etmedi (%{F(sr)})."),
          L(lang, f"Average reversal strength (raw return): 1d {F(al['mean_return_1d_pct'])}%, 3d {F(al['mean_return_3d_pct'])}%, 5d {F(al['mean_return_5d_pct'])}%, 10d {F(al['mean_return_10d_pct'])}%; median 5d {F(al['median_return_5d_pct'])}%.",
            f"Ortalama dönüş gücü (ham getiri): 1 gün %{F(al['mean_return_1d_pct'])}, 3 gün %{F(al['mean_return_3d_pct'])}, 5 gün %{F(al['mean_return_5d_pct'])}, 10 gün %{F(al['mean_return_10d_pct'])}; 5 günlük medyan %{F(al['median_return_5d_pct'])}."),
          gl("with_rsi_trough_and_high_volume", L(lang, "With RSI trough + high volume", "RSI dibi + yüksek hacimle")),
          gl("with_bearish_supertrend_and_below_cloud", L(lang, "With bearish Supertrend + below Ichimoku cloud", "Supertrend düşüşte + bulutun altında")),
          gl("regime_xu100_rising", L(lang, "When XU100 was rising", "XU100 yükselişteyken")),
          gl("regime_xu100_falling", L(lang, "When XU100 was falling", "XU100 düşüşteyken"))]
    small = [k for k, v in g.items() if v.get("warning") and v["events"] > 0]
    lim = " ".join(x for x in [L(lang, f"Conclusion: {why}.", f"Sonuç: {why}."),
                               L(lang, "Combined-condition groups are small; " if small else "", "Birleşik koşul grupları küçük; " if small else ""),
                               warn_text(lang, r["warnings"]),
                               L(lang, "Events whose forward window is not complete at the analysis date are excluded from the statistics.",
                                 "Analiz tarihinde ileri penceresi tamamlanmamış olaylar istatistiklerden hariç tutuldu.")] if x)
    return c.final(state, ev, lim)


# =============================================================== Scenario 4: quarterly fundamentals & price reaction
REAL_FUND = pd.read_csv(ROOT / "data" / "fundamentals.csv")
REAL_FUND_TICKERS = sorted(REAL_FUND.dropna(subset=["revenue", "ebitda"]).ticker.unique())


def make_synthetic_fundamentals(tickers):
    """Training file = REAL parsed KAP data for the 13 covered tickers + synthetic rows for all other tickers."""
    tickers = [t for t in tickers if t not in REAL_FUND_TICKERS]
    rows = []
    disc = {"03-31": "05-{:02d}", "06-30": "08-{:02d}", "09-30": "11-{:02d}", "12-31": "03-{:02d}"}
    for t in tickers:
        rev = rng.uniform(5e3, 2e5); marg = rng.uniform(0.06, 0.25); nd = rng.uniform(-0.5, 3.0)
        for qe in pd.date_range("2023-09-30", "2026-06-30", freq="QE"):
            rev *= rng.uniform(0.85, 1.2); marg = min(max(marg + rng.uniform(-0.03, 0.03), 0.01), 0.4)
            e = rev * marg; nd = max(nd + rng.uniform(-0.4, 0.4), -1)
            key = qe.strftime("%m-%d"); yr = qe.year + (1 if key == "12-31" else 0)
            dd = f"{yr}-" + disc[key].format(rng.randint(2, 28) if key != "12-31" else rng.randint(1, 10))
            rows.append({"ticker": t, "quarter_end": qe.strftime("%Y-%m-%d"), "disclosure_date": dd, "revenue": round(rev, 1),
                         "ebitda": round(e, 1), "net_profit": round(e * rng.uniform(0.2, 0.7), 1), "net_debt": round(nd * e * 4, 1),
                         "ebitda_ltm": round(e * 4 * rng.uniform(0.9, 1.1), 1), "source": "SYNTHETIC_TRAINING_ONLY"})
    pd.concat([REAL_FUND, pd.DataFrame(rows)]).to_csv(SYN_FUND, index=False)


S4_Q = {"en": ["For all {t} quarterly financial statements publicly disclosed over {a} – {b}, how did EBITDA margin and Net Debt/EBITDA change? What were the 1-, 5- and 20-day returns after each disclosure date, relative to XU100 and the sector? Did improving quarters receive a better price reaction?",
               "Analyse {t}'s quarterly fundamentals disclosed between {a} and {b} and the abnormal price reaction after each KAP disclosure.",
               "Did improving quarters of {t} ({a} – {b}) get a better post-disclosure reaction than weaker quarters?"],
        "tr": ["{t} için {a} – {b} arasında kamuya açıklanan tüm çeyreklik finansal tablolarda FAVÖK marjı ve Net Borç/FAVÖK nasıl değişti? Her açıklama tarihinden sonra XU100'e ve sektöre göre 1, 5 ve 20 günlük getiriler ne oldu? İyileşen çeyrekler daha iyi fiyat tepkisi aldı mı?",
               "{t} şirketinin {a} – {b} arasında açıklanan çeyreklik temel verilerini ve her KAP açıklamasından sonraki anormal fiyat tepkisini analiz et.",
               "{t} için {a} – {b} döneminde iyileşen çeyrekler, zayıf çeyreklere göre açıklama sonrası daha iyi tepki aldı mı?"]}


def s4(lang, split, tickers, synthetic=True):
    real = [x for x in tickers if x in REAL_FUND_TICKERS]
    t = rng.choice(real) if real and rng.random() < 0.6 else rng.choice(tickers)
    a, b = "2024-01-01", CUT if rng.random() < 0.7 else rand_date("2025-06-02")
    qt = rng.choice(S4_Q[lang][:2] if split == "train" else S4_Q[lang])
    q = qt.format(t=t, a=show_date(a, lang), b=show_date(b, lang))
    real_csv = macro_fund.FUND_CSV
    if synthetic:
        macro_fund.FUND_CSV = SYN_FUND
    try:
        c = Conv(lang, q)
        m = c.call("get_fundamental_metrics", {"ticker": t, "start": a, "end": b})
        ch = c.call("calculate_fundamental_change", {"ticker": t, "start": a, "end": b})
    finally:
        macro_fund.FUND_CSV = real_csv
    if "error" in m or "error" in ch:
        return None
    if not ch["quarters"]:
        return c.final("INVESTIGATE", [L(lang, f"No disclosed quarterly statements for {t} are available in the fundamentals data for {a} to {b}.",
                                         f"{a} – {b} için {t} şirketinin açıklanmış çeyreklik tablosu temel veri dosyasında bulunmuyor.")],
                       L(lang, "Fundamental data is not loaded, so margin changes and post-disclosure reactions cannot be analysed. " + warn_text(lang, m["warnings"] + ch["warnings"]),
                         "Temel veri yüklü olmadığı için marj değişimleri ve açıklama sonrası tepkiler analiz edilemiyor. " + warn_text(lang, m["warnings"] + ch["warnings"])))
    qs = ch["quarters"]; grp = ch["improving_vs_not_20d"]
    complete20 = sum(1 for x in qs if x["post_disclosure"]["20d"]["complete"])
    if complete20 < 4:
        state = "INVESTIGATE"
    elif qs[-1]["improving"]:
        state = "FUNDAMENTAL_IMPROVEMENT_CANDIDATE"
    else:
        state = "WATCH"
    ev = []
    for x in qs:
        pdx = x["post_disclosure"]
        def ab(key):
            vals = []
            for k in ("1d", "5d", "20d"):
                v = pdx[k]
                vals.append(F(v.get(key)) if v["complete"] else L(lang, "incomplete", "eksik"))
            return "/".join(vals)
        sec_part = L(lang, f"; vs sector {ab('abnormal_vs_sector_pp')} pp", f"; sektöre göre {ab('abnormal_vs_sector_pp')} puan") \
            if any("abnormal_vs_sector_pp" in pdx[k] for k in ("1d", "5d", "20d")) else ""
        ev.append(L(lang, f"{x['quarter_end']} (KAP {x['disclosure_date']}): EBITDA margin {F(x['ebitda_margin_pct'])}% (QoQ {F(x['ebitda_margin_change_qoq_pp'])} pp), Net Debt/EBITDA {F(x['net_debt_to_ebitda'])} (QoQ {F(x['net_debt_to_ebitda_change_qoq'])}), improving={str(x['improving']).lower()}; abnormal 1d/5d/20d vs XU100 {ab('abnormal_vs_xu100_pp')} pp{sec_part}.",
                    f"{x['quarter_end']} (KAP {x['disclosure_date']}): FAVÖK marjı %{F(x['ebitda_margin_pct'])} (çeyreklik {F(x['ebitda_margin_change_qoq_pp'])} puan), Net Borç/FAVÖK {F(x['net_debt_to_ebitda'])} (çeyreklik {F(x['net_debt_to_ebitda_change_qoq'])}), iyileşme={str(x['improving']).lower()}; 1/5/20 gün anormal getiri XU100'e göre {ab('abnormal_vs_xu100_pp')} puan{sec_part}."))
    ev.append(L(lang, f"Mean 20d abnormal return vs XU100: improving quarters {F(grp['improving']['mean_abnormal_20d_vs_xu100_pp'])} pp ({F(grp['improving']['quarters'])} quarters) vs other quarters {F(grp['not_improving']['mean_abnormal_20d_vs_xu100_pp'])} pp ({F(grp['not_improving']['quarters'])} quarters).",
                f"XU100'e göre ortalama 20 günlük anormal getiri: iyileşen çeyrekler {F(grp['improving']['mean_abnormal_20d_vs_xu100_pp'])} puan ({F(grp['improving']['quarters'])} çeyrek), diğerleri {F(grp['not_improving']['mean_abnormal_20d_vs_xu100_pp'])} puan ({F(grp['not_improving']['quarters'])} çeyrek)."))
    lim = " ".join(x for x in [
        L(lang, "Returns are measured from the KAP public disclosure date, not the quarter-end date.", "Getiriler çeyrek sonu değil, KAP kamuya açıklama tarihinden itibaren ölçüldü."),
        L(lang, "This is an association across a few quarters, not a causal effect: good results do not imply a price rise.", "Bu birkaç çeyreklik bir ilişkidir, nedensellik değildir: iyi sonuç fiyat artışı anlamına gelmez."),
        warn_text(lang, ch["warnings"]),
        L(lang, "Incomplete post-disclosure windows are reported as incomplete; no prices after 2026-09-10 are used." if any(not v["complete"] for x in qs for v in x["post_disclosure"].values()) else "",
          "Eksik açıklama sonrası pencereler eksik olarak raporlandı; 2026-09-10 sonrası fiyat kullanılmadı." if any(not v["complete"] for x in qs for v in x["post_disclosure"].values()) else "")] if x)
    return c.final(state, ev, lim)


# =============================================================== Scenario 5: rule & safety
S5_Q = {"en": ["Today is 10.09.2026. USD/TRY is rising and {t} is an exporter, so it will definitely go up. Should I buy {t} tomorrow? Give me a price target.",
               "The lira is weakening, {t} earns in dollars, it can only go up. What price will {t} reach by year end?",
               "Should I sell {t} now? The dollar is moving and I think it will crash. Just tell me yes or no.",
               "Give me a target price for {t} and tell me whether to buy, FX is rising so it's a sure win."],
        "tr": ["Bugün 10.09.2026. USD/TRY yükseliyor ve {t} ihracatçı, kesin yükselir. Yarın {t} alayım mı? Bana hedef fiyat ver.",
               "Kur yükseliyor, {t} dolar bazlı kazanıyor, kesin uçar. Yıl sonunda {t} kaç TL olur?",
               "{t} hisselerimi şimdi satayım mı? Dolar hareketli, bence çökecek. Sadece evet ya da hayır de.",
               "{t} için hedef fiyat ver ve alayım mı söyle, dolar artıyor garanti kazanç."]}


def s5(lang, split, tickers):
    t = rng.choice(tickers)
    qt = rng.choice(S5_Q[lang][:3] if split == "train" else S5_Q[lang])
    c = Conv(lang, qt.format(t=t))
    m = c.call("get_macro_context", {"start": "2024-01-01", "end": CUT, "ticker": t})
    if "error" in m:
        return None
    wk = m.get(f"{t}_vs_USDTRY_weekly"); mo = m.get(f"{t}_vs_USDTRY_monthly")
    fx = m["series"]["USDTRY"]; pr = m["tcmb_policy_rate"]
    state = "INVESTIGATE" if wk and wk["correlation"] <= 0 else "WATCH"
    ev = [L(lang, f"USD/TRY moved from {F(fx['first'])} to {F(fx['last'])} ({F(fx['change_pct'])}%) between {m['period'][0]} and {m['period'][1]}.",
            f"USD/TRY {m['period'][0]} – {m['period'][1]} arasında {F(fx['first'])} seviyesinden {F(fx['last'])} seviyesine geldi (%{F(fx['change_pct'])})."),
          L(lang, f"Weekly {t} vs USD/TRY: correlation {F(wk['correlation'])} over {F(wk['observations'])} weeks, same direction in {F(wk['same_direction_pct'])}% of weeks; mean {t} weekly return {F(wk['mean_stock_return_when_usdtry_up_pct'])}% when USD/TRY rose vs {F(wk['mean_stock_return_when_usdtry_down_pct'])}% when it fell.",
            f"Haftalık {t} – USD/TRY: {F(wk['observations'])} haftada korelasyon {F(wk['correlation'])}, haftaların %{F(wk['same_direction_pct'])}'inde aynı yön; USD/TRY yükseldiğinde {t} ortalama haftalık getirisi %{F(wk['mean_stock_return_when_usdtry_up_pct'])}, düştüğünde %{F(wk['mean_stock_return_when_usdtry_down_pct'])}."),
          L(lang, f"Monthly {t} vs USD/TRY: correlation {F(mo['correlation'])} over {F(mo['observations'])} months.", f"Aylık {t} – USD/TRY: {F(mo['observations'])} ayda korelasyon {F(mo['correlation'])}."),
          L(lang, f"TCMB policy rate moved from {F(pr['level_at_start'])}% to {F(pr['level_at_end'])}% over the period.", f"TCMB politika faizi dönem içinde %{F(pr['level_at_start'])} seviyesinden %{F(pr['level_at_end'])} seviyesine geldi.")]
    claim = L(lang, f"The claim that {t} 'will definitely' move is not verified: the historical relationship with USD/TRY is {'negative' if wk['correlation'] <= 0 else 'positive but weak and not predictive'}.",
              f"{t} hissesinin 'kesin' hareket edeceği iddiası doğrulanmamıştır: USD/TRY ile tarihsel ilişki {'negatif' if wk['correlation'] <= 0 else 'pozitif ama zayıf ve öngörü gücü yok'}.")
    lim = " ".join([L(lang, "I cannot give buy/sell recommendations, price targets or predictions; this system only provides historical analytical states.",
                      "Al/sat önerisi, hedef fiyat veya tahmin veremem; bu sistem yalnızca tarihsel analitik durum sunar."), claim,
                    L(lang, "Correlation is not causation and does not forecast tomorrow's price.", "Korelasyon nedensellik değildir ve yarının fiyatını öngörmez."),
                    warn_text(lang, m["warnings"])])
    return c.final(state, ev, lim)


# =============================================================== Edge cases
EDGE_FUTURE = {"en": ["What will {t} close at on {fd}?", "Rank the {sec} as of {fd} and find the laggard.",
                      "Use prices up to {fd} to test lower Bollinger touches for {t}.", "Predict {t}'s return for next month."],
               "tr": ["{t} {fd} tarihinde kaçtan kapanır?", "{fd} itibarıyla {sec} sektörünü sırala ve geride kalanı bul.",
                      "{t} için {fd} tarihine kadar olan fiyatlarla alt Bollinger temaslarını test et.", "{t} gelecek ay ne kadar getiri sağlar, tahmin et."]}


def edge_future(lang, split, tickers):
    t = rng.choice(tickers); sec = rng.choice(list(SEC_EN))
    fd = rng.choice(["2026-10-15", "2026-11-20", "2026-12-31", "2027-03-01", "2026-09-30"])
    q = rng.choice(EDGE_FUTURE[lang]).format(t=t, sec=(SEC_EN if lang == "en" else SEC_TR)[sec], fd=show_date(fd, lang))
    c = Conv(lang, q)
    return c.final("ANALYSIS_UNSAFE",
                   [L(lang, "The request needs information after the analysis date 2026-09-10, which is not available to this system.",
                      "İstek, analiz tarihi 2026-09-10 sonrasına ait bilgi gerektiriyor; bu bilgi sistemde yok.")],
                   L(lang, "Using data after 2026-09-10 would be look-ahead bias, and forecasting prices is outside the allowed scope. A historical analysis up to 2026-09-10 can be run instead.",
                     "2026-09-10 sonrası veri kullanmak look-ahead bias olur ve fiyat tahmini izin verilen kapsamın dışındadır. Bunun yerine 2026-09-10'a kadar tarihsel analiz yapılabilir."))


def edge_unknown(lang, split, tickers):
    bad = rng.choice(["XYZAB", "APPLE", "GOOGL", "BTCTR", "ABCDE", "TSLA"])
    q = L(lang, f"How many times did {bad} touch the lower Bollinger Band in 2025?", f"{bad} 2025'te alt Bollinger bandına kaç kez değdi?")
    c = Conv(lang, q)
    r = c.call("detect_reversal_events", {"ticker": bad, "event": "lower_bollinger_touch", "start": "2025-01-01", "end": "2025-12-31"})
    return c.final("ANALYSIS_UNSAFE", [L(lang, f"The tool returned {r.get('error')}: {r.get('message')}.", f"Araç {r.get('error')} döndürdü: {r.get('message')}.")],
                   L(lang, "No price data exists for this ticker in the study universe, so no statistics can be reported.",
                     "Bu hisse için çalışma evreninde fiyat verisi yok, bu nedenle istatistik raporlanamaz."))


def edge_history(lang, split, tickers):
    t = rng.choice(tickers)
    a, b = rng.choice([("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31"), ("2026-01-01", CUT), ("2024-01-01", CUT)])
    q = L(lang, rng.choice([f"What was {t}'s return and maximum drawdown between {show_date(a, lang)} and {show_date(b, lang)}?",
                            f"Summarise {t}'s price history from {show_date(a, lang)} to {show_date(b, lang)}."]),
          rng.choice([f"{t} hissesinin {show_date(a, lang)} – {show_date(b, lang)} arası getirisi ve maksimum düşüşü ne oldu?",
                      f"{t} fiyat geçmişini {show_date(a, lang)} – {show_date(b, lang)} için özetle."]))
    c = Conv(lang, q)
    r = c.call("get_stock_history", {"ticker": t, "start": a, "end": b})
    if "error" in r:
        return None
    ev = [L(lang, f"{t} closed at {F(r['first_close'])} on the first day and {F(r['last_close'])} on the last day of {r['period'][0]} to {r['period'][1]}: total return {F(r['total_return_pct'])}%.",
            f"{t} {r['period'][0]} – {r['period'][1]} döneminin ilk günü {F(r['first_close'])}, son günü {F(r['last_close'])} kapanış yaptı: toplam getiri %{F(r['total_return_pct'])}."),
          L(lang, f"Highest close {F(r['max_close'])} ({r['max_close_date']}), lowest close {F(r['min_close'])} ({r['min_close_date']}); maximum drawdown {F(r['max_drawdown_pct'])}%, annualised volatility {F(r['annualised_volatility_pct'])}%.",
            f"En yüksek kapanış {F(r['max_close'])} ({r['max_close_date']}), en düşük {F(r['min_close'])} ({r['min_close_date']}); maksimum düşüş %{F(r['max_drawdown_pct'])}, yıllık oynaklık %{F(r['annualised_volatility_pct'])}.")]
    return c.final("WATCH", ev, L(lang, "Descriptive statistics only; prices are adjusted for splits and dividends.", "Yalnızca betimsel istatistiktir; fiyatlar bölünme ve temettüye göre düzeltilmiştir."))


# =============================================================== main
PLAN = {"train": [(s1, 260), (s2, 230), (s3, 260), (s4, 170), (None, 30), (s5, 120), (edge_future, 50), (edge_unknown, 25), (edge_history, 50)],
        "val": [(s1, 20), (s2, 20), (s3, 20), (s4, 15), (s5, 10), (edge_future, 5), (edge_history, 5)],
        "test": [(s1, 25), (s2, 25), (s3, 25), (s4, 20), (s5, 15), (edge_future, 5), (edge_unknown, 5)]}


ACCEPT = {"WATCH": 0.35, "REJECT_SIGNAL": 0.45, "INVESTIGATE": 0.7}


def scan_catchup(n=1500):
    """Find sector/date combos where the catch-up rule fires, so the rare label is represented."""
    seen = set()
    for _ in range(n):
        r = s1("en", "train", forced=None) if not CATCHUP_POOL or rng.random() < 0.9 else None
        if r and r["state"] == "POTENTIAL_CATCH_UP_CANDIDATE":
            key = json.dumps(r["params"][2:], default=str)
            if key not in seen:
                seen.add(key); CATCHUP_POOL.append(r["params"])
    print("catch-up candidate combos found:", len(CATCHUP_POOL))


def run():
    make_synthetic_fundamentals(TRAIN_TICKERS + sorted(TEST_TICKERS))
    scan_catchup()
    stats = {}
    for split, plan in PLAN.items():
        tickers = TRAIN_TICKERS if split != "test" else sorted(TEST_TICKERS)
        rows = []
        for fn, n in plan:
            made, tries = 0, 0
            while made < n and tries < n * 30:
                tries += 1
                lang = "tr" if rng.random() < 0.5 else "en"
                if fn is None:      # S4 with real (empty) fundamentals -> INVESTIGATE
                    r = s4(lang, split, tickers, synthetic=False)
                elif fn is s1:
                    r = s1(lang, split)
                else:
                    r = fn(lang, split, tickers)
                if r is None:
                    continue
                # balance labels: keep frequent states less often so rare candidate states are learned
                if rng.random() > ACCEPT.get(r["state"], 1.0):
                    continue
                r["scenario"] = "s4_empty" if fn is None else fn.__name__
                rows.append(r); made += 1
        rng.shuffle(rows)
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"messages": r["messages"], "meta": {"scenario": r["scenario"], "state": r["state"]}}, ensure_ascii=False) + "\n")
        lens = [sum(len(m["content"]) for m in r["messages"]) for r in rows]
        stats[split] = {"n": len(rows), "chars_mean": int(np.mean(lens)), "chars_max": int(np.max(lens)),
                        "states": pd.Series([r["state"] for r in rows]).value_counts().to_dict(),
                        "scenarios": pd.Series([r["scenario"] for r in rows]).value_counts().to_dict()}
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    run()

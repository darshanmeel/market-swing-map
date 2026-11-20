"""Swing map per market: turning points, distance from highs, past falls this deep, and a saved guess each day.

Usage: python market_highs.py [TICKER ...]   (Yahoo symbols, default: every market in NAMES)
Writes one offline page per market (highs_<ticker>.html, from swing_template.html), today's numbers to
market_highs.csv, and the day's guess to predictions/<ticker>.json (older guesses kept and scored).
"""
import json
import math
import sys
from html import escape
from pathlib import Path

import pandas as pd
import yfinance as yf
from plotly.offline import get_plotlyjs

NAMES = {"^IXIC": "Nasdaq Composite", "^GSPC": "S&P 500", "^DJI": "Dow Jones", "^GDAXI": "DAX 40",
         "^FTSE": "FTSE 100", "^FCHI": "CAC 40", "^N225": "Nikkei 225", "^KS11": "KOSPI", "^HSI": "Hang Seng",
         "^BSESN": "Sensex", "GC=F": "Gold (futures)", "SI=F": "Silver (futures)", "CL=F": "WTI crude oil (futures)"}
SHOW_YEARS = 10
WARMUP_YEARS = 1  # turning points start a year early so the first swing in view is a real one
DAILY_ROWS = 30
MIN_DIP = 0.02  # a fall from the all-time high smaller than this is noise
BIG_FALL = 10  # falls this deep (%) get a label and a row in the falls table
WINDOWS = {"1-year": "365D", "6-month": "182D"}
FWD = 252  # trading days (about a year) followed after a past fall reached today's depth
OUT = Path(__file__).parent
PRED = OUT / "predictions"  # one file per market: every day's prediction, kept
BACKFILL = 20  # on a market's first run, also make the last 20 days' guesses from the data each day had
PICK = 8  # scenarios per guess: the past falls that reached today's depth at the most similar speed
HORIZON = 252  # trading days (about a year) each guess looks ahead
SHOW_GUESSES = 60  # newest guesses carried into the page; the file keeps all


def load(tickers):
    # Full history so the all-time high is real, not just the max of the shown window.
    close = yf.download(tickers, period="max", auto_adjust=False, progress=False)["Close"]
    data = {t: close[t].dropna().loc[lambda x: x > 0] for t in tickers if t in close}  # WTI went negative in 2020
    for t in tickers:
        if t not in data or data[t].empty:
            print(f"No data for {t}: check the Yahoo symbol.", file=sys.stderr)
            data.pop(t, None)
    if not data:
        sys.exit("No data downloaded; nothing written.")
    return data


def highs(s):
    """Each close against the all-time high set BEFORE that day, so a new record shows as a gain over the old one."""
    out = pd.DataFrame({"close": s})
    out["high_ATH"] = s.shift().cummax()
    out["pct_ATH"] = s / out.high_ATH - 1
    out["prev_ath_date"] = pd.Series(s.index, index=s.index).where(s >= s.cummax()).ffill().shift()
    return out


def span(days):
    if days < 60:
        return f"{days} days"
    return f"{round(days / 30.44)} months" if days < 730 else f"{days / 365.25:.1f} years"


def episodes(d, start):
    """One row per fall from an all-time high: peak, low, recovery, and the rise after the low."""
    pct = d.pct_ATH
    below = pct < 0
    rows = []
    for _, p in pct[below].groupby((~below).cumsum()[below]):
        if p.min() > -MIN_DIP:
            continue
        low, after, first = p.idxmin(), pct.index[pct.index > p.index[-1]], p.index[0]
        peak = d.prev_ath_date[first]
        rows.append({
            "Peak on": peak.date(), "Peak close": round(d.high_ATH[first], 2),
            "Fall": f"{p.min() * 100:.1f}% in {span((low - peak).days)}{'' if len(after) else ' so far'}",
            "Low on": low.date(), "Depth %": round(p.min() * 100, 2), "Close at low": round(d.close[low], 2),
            "Recovery": span((after[0] - low).days) if len(after) else "not yet",
            "Back at high": after[0].date() if len(after) else "not yet",
            "Days to recover": (after[0] - low).days if len(after) else None,
            "Status": "reversed" if len(after) else "still falling",
            "_end": after[0] if len(after) else d.index[-1],
        })
    df = pd.DataFrame(rows)
    df["Days to recover"] = df["Days to recover"].astype("Int64")
    # Rise after the low: to the highest close before the next big fall began (or until today).
    big = df[df["Depth %"] <= -BIG_FALL]
    rise, up, up_days = {}, {}, {}
    for i, nxt in zip(big.index, list(big.index[1:]) + [None]):
        if df.Status[i] != "reversed":
            continue
        low = pd.Timestamp(df["Low on"][i])
        end = pd.Timestamp(df["Peak on"][nxt]) if nxt is not None else d.index[-1]
        seg = d.close[(d.index > low) & (d.index <= end)]
        up[i], up_days[i] = float(seg.max() / df['Close at low'][i] - 1), (seg.idxmax() - low).days
        rise[i] = (f"+{up[i] * 100:.1f}% in {span(up_days[i])}"
                   f"{'' if nxt is not None else ' so far'}")
    df["Rise after low"] = pd.Series(rise)
    df["_up"], df["_upDays"] = pd.Series(up), pd.Series(up_days)
    return df[df._end >= start].drop(columns="_end").reset_index(drop=True)


def similar(d):
    """Past falls from an all-time high that went at least as deep as today's: how fast, how much further, what came next."""
    pct, now = d.pct_ATH, d.pct_ATH.iloc[-1]
    if now > -MIN_DIP:
        return None
    peak_now = d.prev_ath_date.iloc[-1]
    days_now = (d.index[-1] - peak_now).days
    every = episodes(d, d.index[0])
    every = every[every.Status == "reversed"]
    eps = every[every["Depth %"] <= now * 100]
    rows = []
    for _, e in eps.iterrows():
        peak, low = (pd.Timestamp(e[k]) for k in ("Peak on", "Low on"))
        after = pct[pct.index > peak]
        hit = after[after <= now].index[0]
        rows.append({
            "peak": str(peak.date()), "hitDays": (hit - peak).days,
            "first": float(after[after.index <= peak + pd.Timedelta(days=days_now)].min()),
            "more": float(d.close[low] / d.close[hit] - 1), "moreDays": (low - hit).days,
            "low": str(low.date()), "depth": float(e["Depth %"]) / 100,
            "recDays": int(e["Days to recover"]), "recover": e["Recovery"],
            "rise": e["Rise after low"] if isinstance(e["Rise after low"], str) else "",
            "up": None if pd.isna(e["_up"]) else e["_up"], "upDays": None if pd.isna(e["_upDays"]) else int(e["_upDays"])})
    return {"now": float(now), "days": days_now, "peak": str(peak_now.date()), "since": str(d.index[0].year), "rows": rows,
            "scen": [{"extra": x, "cases": next_year(d, every, (1 + now) * (1 - x) - 1)} for x in (0, 0.1, 0.2, 0.3)]}


def next_year(d, eps, depth):
    """For each past fall that went at least `depth` below its high: the year after the day it first got there, as returns."""
    out = []
    for _, e in eps[eps["Depth %"] <= depth * 100].iterrows():
        after = d.pct_ATH[d.index > pd.Timestamp(e["Peak on"])]
        hit = after[after <= depth].index[0]
        fwd = d.close[d.index >= hit].iloc[:FWD + 1]
        out.append({"peak": str(e["Peak on"]), "hit": str(hit.date()), "fwd": [round(float(v / fwd.iloc[0] - 1), 4) for v in fwd]})
    return out


def snapshot(d):
    """The day's guess: the past falls that got this deep at the most similar speed, each replayed from today's close."""
    s = similar(d)
    if not s or len(s["rows"]) < 3:
        return None
    last = float(d.close.iloc[-1])
    pairs = list(zip(s["rows"], s["scen"][0]["cases"]))
    assert all(r["peak"] == c["peak"] for r, c in pairs)
    pairs.sort(key=lambda rc: abs(math.log(max(rc[0]["hitDays"], 1) / max(s["days"], 1))))
    return {"date": str(d.index[-1].date()), "close": round(last, 2), "depth": round(s["now"], 4), "days": s["days"], "n": len(pairs),
            "cases": [{"peak": r["peak"], "hitDays": r["hitDays"], "path": [round(last * (1 + x), 2) for x in c["fwd"][:HORIZON + 1]]}
                      for r, c in pairs[:PICK]]}


def score(h, d):
    """A guess against the closes since: inside the scenarios or not, and which scenario it follows most closely."""
    real = [round(float(x), 2) for x in d.close[d.index > pd.Timestamp(h["date"])].iloc[:HORIZON]]
    out, k = {**h, "real": real}, len(real)
    paths = [c["path"] for c in h["cases"] if len(c["path"]) > k]
    if k == 0 or len(paths) < 2:
        return out
    now, at = real[-1], [p[k] for p in paths]
    up = sum(x > h["close"] for x in at)
    near = min(h["cases"], key=lambda c: sum(abs(r / c["path"][j + 1] - 1) for j, r in enumerate(real) if j + 1 < len(c["path"])))
    if now < min(at):
        res, why = "below", f"{(now / min(at) - 1) * 100:.2f}% under the lowest scenario: weaker than every one of them"
    elif now > max(at):
        res, why = "above", f"{(now / max(at) - 1) * 100:+.2f}% over the highest scenario: stronger than every one of them"
    else:
        res, why = "inside", f"{up} of {len(at)} scenarios were higher by now; it is {'higher' if now > h['close'] else 'lower'}"
        if up * 2 != len(at) and (up * 2 > len(at)) != (now > h["close"]):
            why += ", against most of them"
    return {**out, "lo": min(at), "hi": max(at), "res": res, "why": why, "near": near["peak"]}


def track(name, d):
    """Keep one guess per day (a rerun on the same day replaces it) and score each against the closes since."""
    f = PRED / f"{name}.json"
    day = str(d.index[-1].date())
    hist = json.loads(f.read_text()) if f.exists() else [snapshot(d[d.index <= x]) for x in d.index[-BACKFILL - 1:-1]]
    hist = [h for h in hist if h and h["date"] != day] + [snapshot(d)]
    hist = [h for h in hist if h]
    PRED.mkdir(exist_ok=True)
    f.write_text(json.dumps(hist), encoding="utf-8")
    return [score(h, d) for h in hist[-SHOW_GUESSES:]]


def highs_lows(d):
    day, before = d.index[-1], d.close.iloc[:-1]
    last = d.close.iloc[-1]
    rows = [("All-time high", before)]
    for name, win in WINDOWS.items():
        rows.append((f"{name} high", before[before.index > day - pd.Timedelta(win)]))
    out = [{"label": k, "value": float(v.max()), "date": str(v.idxmax().date()), "pct": float(last / v.max() - 1)}
           for k, v in rows]
    for name, win in WINDOWS.items():
        v = before[before.index > day - pd.Timedelta(win)]
        out.append({"label": f"{name} low", "value": float(v.min()), "date": str(v.idxmin().date()),
                    "pct": float(last / v.min() - 1)})
    return out


def facts(d, eps):
    pct, ath_on, day = d.pct_ATH.iloc[-1], d.prev_ath_date.iloc[-1], d.index[-1]
    out = [("Days since the all-time high", f"{(day - ath_on).days} (high on {ath_on.date()})")]
    if pct < 0:
        before = d.pct_ATH[(d.index < ath_on) & (d.pct_ATH <= pct)]
        out.append(("Last time this far below the high",
                    f"{before.index[-1].date()} ({before.iloc[-1] * 100:+.1f}%)" if len(before) else "never"))
        deep = eps[(eps.Status == "reversed") & (eps["Depth %"] <= pct * 100)]
        if len(deep):
            out.append((f"Past falls this deep ({SHOW_YEARS}y)",
                        f"{len(deep)}, ended at {deep['Depth %'].max():.1f}% to {deep['Depth %'].min():.1f}%; median "
                        f"{span(int(deep['Days to recover'].median()))} from the low back to the high"))
    done = eps[(eps.Status == "reversed") & (eps["Depth %"] <= -BIG_FALL)]
    if len(done):
        e = done.iloc[-1]
        out.append((f"Last finished {BIG_FALL}%+ fall", f"{e['Fall']} to {e['Low on']}, then {e['Rise after low']}"))
    return [(k, escape(v)) for k, v in out]


def daily_table(d):
    t = d.tail(DAILY_ROWS + 1)
    out = pd.DataFrame({"Date": t.index.date, "Close": t.close.round(2), "Day %": (t.close.pct_change() * 100).round(2),
                        "vs ATH %": (t.pct_ATH * 100).round(2)})
    for name, win in WINDOWS.items():
        prev = d.close.rolling(win, closed="left")
        out[f"vs {name} high %"] = ((t.close / prev.max().loc[t.index] - 1) * 100).round(2)
        out[f"vs {name} low %"] = ((t.close / prev.min().loc[t.index] - 1) * 100).round(2)
    return out.iloc[1:].iloc[::-1].to_html(index=False, border=0, na_rep="-")


def page_data(t, d, eps, pages, name):
    start = d.index[-1] - pd.DateOffset(years=SHOW_YEARS + WARMUP_YEARS)
    w = d[d.index >= start]
    big = eps[(eps["Depth %"] <= -BIG_FALL) & (eps.Status == "reversed")]
    cols = ["Peak on", "Peak close", "Fall", "Low on", "Close at low", "Recovery", "Back at high", "Rise after low"]
    return {
        "name": NAMES.get(t, t), "ticker": t, "pages": pages,
        "dates": [str(x.date()) for x in w.index], "close": [round(float(x), 2) for x in w.close],
        "pctAth": [round(float(x), 5) for x in w.pct_ATH.fillna(0)],
        "athBefore": float(d.close[d.index < start].max()) if (d.index < start).any() else 0.0,
        "athValue": float(d.high_ATH.iloc[-1]), "athDate": str(d.prev_ath_date.iloc[-1].date()),
        "falls": [{"low": str(e["Low on"]), "depth": float(e["Depth %"]), "text": e["Fall"]} for _, e in big.iterrows()],
        "fallsTable": eps[eps["Depth %"] <= -BIG_FALL][cols].iloc[::-1].to_html(index=False, border=0, na_rep="-"),
        "rank": [{"peak": str(e["Peak on"]), "low": str(e["Low on"]), "depth": float(e["Depth %"]), "now": e.Status != "reversed"}
                 for _, e in episodes(d, d.index[0]).iterrows() if e["Depth %"] <= -BIG_FALL or e.Status != "reversed"],
        "dailyTable": daily_table(d), "highsLows": highs_lows(d), "facts": facts(d, eps), "similar": similar(d), "track": track(name, d),
    }


def main():
    tickers = sys.argv[1:] or list(NAMES)
    data = {t: highs(s) for t, s in load(tickers).items()}
    slug = {t: t.strip('^').replace('=', '_').lower() for t in data}
    files = {t: f"highs_{slug[t]}.html" for t in data}
    template = (OUT / "swing_template.html").read_text(encoding="utf-8")
    (OUT / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")

    summary = []
    for t, d in data.items():
        start = d.index[-1] - pd.DateOffset(years=SHOW_YEARS)
        pages = [{"name": NAMES.get(o, o), "href": files[o], "here": o == t} for o in data]
        payload = json.dumps(page_data(t, d, episodes(d, start), pages, slug[t])).replace("</", "<\\/")
        html = template.replace("__TITLE__", escape(f"Swing map: {NAMES.get(t, t)}")).replace("__DATA__", payload)
        (OUT / files[t]).write_text(html, encoding="utf-8")
        summary.append({"index": NAMES.get(t, t), "date": d.index[-1].date(), "close": round(d.close.iloc[-1], 2),
                        "vs ATH %": round(d.pct_ATH.iloc[-1] * 100, 2), "ATH set on": d.prev_ath_date.iloc[-1].date()})
    table = pd.DataFrame(summary)
    table.to_csv(OUT / "market_highs.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

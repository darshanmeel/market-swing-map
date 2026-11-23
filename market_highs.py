"""Below the Record per market: turning points, distance from highs, past falls this deep, and a saved guess each day.

Usage: python market_highs.py [TICKER ...]   (Yahoo symbols, default: every market in markets.csv)
       python market_highs.py --add SYMBOL "Name" [--group GROUP]   (adds it to markets.csv, group "Other US-listed"
       unless given, then builds it; a symbol Yahoo has no data for is not added)
       python market_highs.py --summary-only   (only the home page numbers, from the stored closes: no download)
       python market_highs.py --serve [PORT]   (serves this folder on http://127.0.0.1:PORT, default 8765, so app.html
       can build a market with its "Build now" button)
Markets build in parallel, one process per CPU core; a failed market is printed and makes the exit code non-zero.
Writes each market's data to data/<slug>.js, shown offline by the one page app.html?m=<slug>,
and the day's guess to predictions/<slug>.json
(older guesses kept and scored). data/index.js lists every market for the page's menu, with the last close of each built one;
data/summary.js holds the numbers the home page index.html shows.
Closes are kept in data/closes.csv, so a run only downloads the latest days.
"""
import csv
import json
import math
import os
import re
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import yfinance as yf
from plotly.offline import get_plotlyjs

LIST = Path(__file__).parent / "markets.csv"  # every market: symbol, name, menu group


def read_list():
    with open(LIST, newline="", encoding="utf-8-sig") as f:  # -sig: Excel saves a BOM
        rows = list(csv.DictReader(f))
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], {})[r["symbol"]] = r["name"]
    # Indices, commodities and crypto keep the list's order; company groups are sorted by name
    groups = {g: m if g in ("Indices", "Commodities and crypto") else dict(sorted(m.items(), key=lambda kv: kv[1]))
              for g, m in groups.items()}
    return {r["symbol"]: r["name"] for r in rows}, groups


NAMES, GROUPS = read_list()
SHOW_YEARS = 10
WARMUP_YEARS = 1  # turning points start a year early so the first swing in view is a real one
DAILY_ROWS = 30
MIN_DIP = 0.02  # a fall from the all-time high smaller than this is noise
BIG_FALL = 10  # falls this deep (%) get a label and a row in the falls table
WINDOWS = {"1-year": "365D", "6-month": "182D"}
FWD = 252  # trading days (about a year) followed after a past fall reached today's depth
OUT = Path(__file__).parent
STORE = OUT / "data" / "closes.csv"
INDEX = OUT / "data" / "index.js"  # the menu: every market in markets.csv and whether it is built
SUMMARY = OUT / "data" / "summary.js"  # the home page numbers per market; only index.html loads it
PRED = OUT / "predictions"  # one file per market: every day's prediction, kept
BACKFILL = 20  # on a market's first run, also make the last 20 days' guesses from the data each day had
PICK = 8  # scenarios per guess: the past falls that reached today's depth at the most similar speed
HORIZON = 252  # trading days (about a year) each guess looks ahead
SHOW_GUESSES = 60  # newest guesses carried into the page; the file keeps all
BT_STEP = 5  # back-test paths kept every 5th trading day, to keep the page small
TREND_SIZES = (8, 10, 15)  # reversal sizes (%) the home page reads the trend at: indices 8, the others 10 and 15


def fetch(tickers, **when):
    return yf.download(tickers, auto_adjust=False, progress=False, **when)["Close"]


def load(tickers):
    # Closes are kept in data/closes.csv; a run only fetches the last few days, and a market's full history
    # (so the all-time high is real) only when it is new or its old closes changed (a split rescales them).
    old = pd.read_csv(STORE, index_col=0, parse_dates=True) if STORE.exists() else pd.DataFrame()
    have = [t for t in tickers if t in old and old[t].notna().any()]
    recent = fetch(have, start=old.index[-1] - pd.Timedelta("10D")) if have else pd.DataFrame()
    both = recent.index.intersection(old.index[:-1])  # the last stored close may be a mid-day price
    split = [t for t in have if t not in recent or (recent.loc[both, t] / old.loc[both, t] - 1).abs().max() > .01]
    full = [t for t in tickers if t not in have] + split
    close = recent.drop(columns=split, errors="ignore").combine_first(old)
    if full:  # in chunks: hundreds of new symbols in one call get rate-limited into empty columns
        new = pd.concat([fetch(full[i:i + 50], period="max") for i in range(0, len(full), 50)], axis=1)
        close = close.drop(columns=full, errors="ignore").combine_first(new)
    STORE.parent.mkdir(exist_ok=True)
    close.to_csv(STORE)
    data = {t: close[t].dropna().loc[lambda x: x > 0] for t in tickers if t in close}  # WTI went negative in 2020
    for t in tickers:
        if t not in data or data[t].empty:
            print(f"No data for {t}: check the Yahoo symbol.", file=sys.stderr)
            data.pop(t, None)
        elif data[t].index[-1] < close.index[-1] - pd.Timedelta("10D"):  # delisted or taken over
            print(f"No close for {t} since {data.pop(t).index[-1].date()}: remove it from markets.csv.", file=sys.stderr)
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
    df = pd.DataFrame(rows, columns=["Peak on", "Peak close", "Fall", "Low on", "Depth %", "Close at low", "Recovery",
                                     "Back at high", "Days to recover", "Status", "_end"])
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


def similar(d, every=None):
    """Past falls from an all-time high that went at least as deep as today's: how fast, how much further, what came next."""
    pct, now = d.pct_ATH, d.pct_ATH.iloc[-1]
    if now > -MIN_DIP:
        return None
    peak_now = d.prev_ath_date.iloc[-1]
    days_now = (d.index[-1] - peak_now).days
    every = episodes(d, d.index[0]) if every is None else every
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
        hit = after[after <= max(depth, after.min())].index[0]  # "Depth %" is rounded, so the low may miss depth by a hair
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
    turned = episodes(d, d.index[0])
    turned = turned[turned.Status == "reversed"]
    return [{**score(h, d), "alt": other_ways(d[d.index <= h["date"]], turned)} for h in hist[-SHOW_GUESSES:]]


def other_ways(d, turned):
    """Four other guesses from the same close, as returns: middle path and 10-90% band (trend: one line)."""
    c, k = d.close.to_numpy(), np.arange(HORIZON + 1)
    band = lambda m: {"n": len(m), **{key: [round(float(x), 3) for x in np.nanquantile(m, q, axis=0)]
                                      for key, q in (("mid", .5), ("lo", .1), ("hi", .9))}}
    out = {}
    # Shallower falls that turned: falls that stopped between half of today's depth and today's depth, replayed from their low
    now = d.pct_ATH.iloc[-1] * 100
    lows = turned[(turned["Depth %"] > now) & (turned["Depth %"] <= now / 2) & (pd.to_datetime(turned["Back at high"]) <= d.index[-1])]["Low on"]
    m = np.full((len(lows), HORIZON + 1), np.nan)
    for r, low in enumerate(lows):
        i = d.index.get_loc(pd.Timestamp(low)); seg = c[i:i + HORIZON + 1]
        m[r, :len(seg)] = seg / seg[0] - 1
    if len(lows) >= 3:
        out["shallow"] = band(m)
    # Random walk: this market's own daily moves drawn at random, 1000 paths; seeded so a rebuild gives the same guess
    r = np.diff(np.log(c))
    out["random"] = band(np.exp(np.cumsum(np.c_[np.zeros(1000), np.random.default_rng(len(c)).choice(r, (1000, HORIZON))], axis=1)) - 1)
    # Any year: every 1-year stretch in the history (start every 5th day), from any starting point
    st = np.arange(0, len(c) - HORIZON, 5)
    out["year"] = band(c[st[:, None] + k] / c[st][:, None] - 1)
    # Trend continues: the last 60 trading days' daily pace kept up
    out["trend"] = {"mid": [round(float(x), 3) for x in np.exp(k * (np.log(c[-1] / c[-61]) / 60)) - 1]}
    return out


def backtest(d):
    """Each past fall from a record that got as deep as today: the five guesses made on its first close that deep,
    from the closes up to that day only, against the real year after (returns, every 5th day)."""
    now, c = d.pct_ATH.iloc[-1], d.close.to_numpy()
    if now > -MIN_DIP:
        return None
    eps = episodes(d, d.index[0])
    r3 = lambda a: [round(float(x), 3) for x in a]
    rows = []
    for _, e in eps[(eps["Depth %"] <= now * 100) & (eps.Status == "reversed")].iterrows():
        after = d.pct_ATH[d.index > pd.Timestamp(e["Peak on"])]
        hit = after[after <= now].index[0]
        i = d.index.get_loc(hit)
        if i < HORIZON:  # under a year of earlier closes: too little to guess from
            continue
        past = d[d.index <= hit]
        ways, s = {}, snapshot(past)
        if s:  # past deeper falls: middle (as on the page), lowest and highest case
            P = [np.array(x["path"]) / s["close"] - 1 for x in s["cases"]]
            v = [sorted(q[j] for q in P if len(q) > j) for j in range(max(map(len, P)))]
            ways["deeper"] = {"n": len(P), "mid": [x[len(x) // 2] for x in v], "lo": [x[0] for x in v], "hi": [x[-1] for x in v]}
        turned = episodes(past, past.index[0])
        ways |= other_ways(past, turned[turned.Status == "reversed"])
        rows.append({"peak": str(e["Peak on"]), "hit": str(hit.date()), "depth": round(float(d.pct_ATH.iloc[i]), 3),
                     "low": str(e["Low on"]), "bottom": round(float(e["Depth %"]) / 100, 3),
                     "real": r3(c[i:i + HORIZON + 1:BT_STEP] / c[i] - 1),
                     "ways": {k: {"n": w.get("n"), **{q: r3(w[q][::BT_STEP]) for q in ("mid", "lo", "hi") if q in w}} for k, w in ways.items()}})
    return {"now": round(float(now), 3), "step": BT_STEP, "rows": rows}


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


def swing_falls(c, th=BIG_FALL / 100):
    """Falls of th or more from a swing high (a high that held until a th drop), each ended by a th rise; the last may be ongoing."""
    out, down, hi, hid, lo, lod = [], False, c.iloc[0], c.index[0], None, None
    for t, v in c.items():
        if not down and v > hi:
            hi, hid = v, t
        elif not down and v <= hi * (1 - th):
            down, lo, lod = True, v, t
        elif down and v < lo:
            lo, lod = v, t
        elif down and v >= lo * (1 + th):
            out.append({"peak": str(hid.date()), "low": str(lod.date()), "depth": round((lo / hi - 1) * 100, 2), "now": False})
            down, hi, hid = False, v, t
    last = c.iloc[-1]
    if last < hi:
        out.append({"peak": str(hid.date()), "low": str((lod if down else c.index[-1]).date()), "depth": round((last / hi - 1) * 100, 2), "now": True})
    return out


def page_data(t, d, eps, pages, name, every, sim):
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
        "rank": [{"peak": str(e["Peak on"]), "low": str(e["Low on"]), "depth": float(e["Depth %"]), "now": e.Status != "reversed",
                  "back": None if e.Status != "reversed" else str(e["Back at high"])}
                 for _, e in every.iterrows() if e["Depth %"] <= -BIG_FALL or e.Status != "reversed"],
        "rankSwing": swing_falls(d.close),
        "dailyTable": daily_table(d), "highsLows": highs_lows(d), "facts": facts(d, eps), "similar": sim, "track": track(name, d), "backtest": backtest(d),
    }


def trend(c, prev_max, t):
    """The Trend shape tile of app.html at reversal size t, with the same rules: the running extreme becomes a turning
    point once the price reverses by t; the last swing high and low are tagged against the one before."""
    up, hi, lo, piv = True, 0, 0, []
    for i in range(1, len(c)):
        if up:
            if c[i] > c[hi]:
                hi = i
            elif c[i] <= c[hi] * (1 - t):
                piv.append((hi, "H"))
                up, lo = False, i
        elif c[i] < c[lo]:
            lo = i
        elif c[i] >= c[lo] * (1 + t):
            piv.append((lo, "L"))
            up, hi = True, i
    if not piv:
        return None
    last, tags = {"H": None, "L": None}, {"H": [], "L": []}
    for i, k in piv:
        p = last[k]
        if k == "H":
            tag = "ATH" if c[i] > prev_max[i] else "" if p is None else "HH" if c[i] > c[p] else "LH"
        else:
            tag = "" if p is None else "HL" if c[i] > c[p] else "LL"
        last[k] = i
        if tag:
            tags[k].append(tag)
    h, l = (tags[k][-1] if tags[k] else "" for k in "HL")
    h_up, l_up = h in ("HH", "ATH"), l == "HL"
    i, k = piv[-1]
    return {"trend": "Uptrend" if h_up and l_up else "Downtrend" if not (h_up or l_up) else "Mixed", "high": h, "low": l,
            "belowLow": last["L"] is not None and c[-1] < c[last["L"]],
            "aboveHigh": last["H"] is not None and c[-1] > c[last["H"]], "rising": up,
            "move": round(c[-1] / c[i] - 1, 4), "from": i, "fromType": k,
            # The last two finished swings, oldest first: start, end and move
            "legs": [(a, b, round(c[b] / c[a] - 1, 4)) for (a, _), (b, _) in zip(piv[-3:-1], piv[-2:])]}


def overview(d, every=None, s=None):
    """A few numbers per market for the home page, from the same closes the market page uses. `every` and `s` are
    episodes() over the whole history and similar(), when the caller has them already."""
    day = d.index[-1]
    start = day - pd.DateOffset(years=SHOW_YEARS + WARMUP_YEARS)
    w = d[d.index >= start]
    c = [round(float(x), 2) for x in w.close]  # as in the page's data file, so the turns match the page
    prev_max, m = [], float(d.close[d.index < start].max()) if (d.index < start).any() else 0.0
    for x in c:
        prev_max.append(m)
        m = max(m, x)
    rec = day if d.pct_ATH.iloc[-1] >= 0 else d.prev_ath_date.iloc[-1]  # prev_ath_date is the record before today's
    moves = np.log(d.close).diff()[d.index > day - pd.DateOffset(years=1)].dropna()  # crypto trades every day: ~365 moves
    out = {"date": str(day.date()), "close": c[-1], "day": round(c[-1] / c[-2] - 1, 4) if len(c) > 1 else None,
           "ath": round(float(d.pct_ATH.iloc[-1]), 4), "athDate": str(rec.date()), "athDays": (day - rec).days,
           "vol": round(float(moves.std() * math.sqrt(len(moves))), 4) if len(moves) > 1 else None}
    for y in (1, 5, 10):
        then = day - pd.DateOffset(years=y)
        past = d.close[d.index <= then]
        out[f"r{y}"] = round(float((d.close.iloc[-1] / past.iloc[-1]) ** (1 / y) - 1), 4) if len(past) else None  # 5y and 10y: per year
        if y > 1:
            out[f"dd{y}"] = round(min(float(d.pct_ATH[d.index > then].min()), 0), 4)  # shorter history: since listing
            out[f"hi{y}"] = round(float(d.close.iloc[-1] / d.close[d.index > then].max() - 1), 4)  # below the 5y / 10y high
    out["tr"] = {}
    for t in TREND_SIZES:
        z = trend(c, prev_max, t / 100)
        if z:
            out["tr"][t] = {**z, "from": str(w.index[z["from"]].date()), "since": (day - w.index[z["from"]]).days,
                            "legs": [{"move": m, "from": str(w.index[a].date()), "to": str(w.index[b].date()), "days": (w.index[b] - w.index[a]).days}
                                     for a, b, m in z["legs"]]}
    # As the page's "Below the record" tile: past 10%+ falls from a record in 10 years, and how many went deeper than today
    y10 = str((day - pd.DateOffset(years=10)).date())
    eps = episodes(d, d.index[0]) if every is None else every
    past = eps[(eps.Status == "reversed") & (eps["Depth %"] <= -BIG_FALL) & (eps["Peak on"].astype(str) >= y10)]
    out["falls"], out["deeper"] = len(past), int((past["Depth %"] < round(float(d.pct_ATH.iloc[-1]), 5) * 100).sum())
    # As the page's "Higher a year later" tile: past falls that got this deep, and how many were higher a year on
    s = similar(d, eps) if s is None else s
    yrs = [x["fwd"][FWD] for x in s["scen"][0]["cases"] if len(x["fwd"]) > FWD] if s and s["rows"] else []
    out["yrUp"], out["yrN"] = sum(v > 0 for v in yrs), len(yrs)
    return out


def build(t, d, slug):
    """One market's data file and guess; runs in a worker process. Returns its home page numbers."""
    start = d.index[-1] - pd.DateOffset(years=SHOW_YEARS)
    # Every market in the menu, also those not rebuilt in this run
    pages = [{"name": n, "href": f"app.html?m={slug[o]}", "here": o == t, "group": g} for g, m in GROUPS.items() for o, n in m.items()]
    every = episodes(d, d.index[0])
    sim = similar(d, every)
    payload = json.dumps(page_data(t, d, episodes(d, start), pages, slug[t], every, sim)).replace("</", "<\\/")
    (OUT / "data" / f"{slug[t]}.js").write_text(f"window.D = {payload};\n", encoding="utf-8")
    return overview(d, every, sim)


def file_date(f):
    """The last close a data file holds: the last entry of its "dates" list, read without parsing the whole payload."""
    s = f.read_text(encoding="utf-8")
    i = s.find('"dates": [')
    m = re.search(r'"([0-9-]{10})"\]', s[i:s.find("]", i) + 1]) if i >= 0 else None
    return m and m[1]


def write_index(slug, done, sums):
    """data/index.js for the menu: every market in markets.csv; a built one (its data file exists) has its last close
    and build time. Markets not rebuilt in this run keep their entry while their file is unchanged; a file written since
    (another run, another folder) gets its date from the file itself. The home page numbers go to data/summary.js, so the
    market page does not load them; a market without new ones keeps its old ones."""
    try:
        old = json.loads(INDEX.read_text(encoding="utf-8").removeprefix("window.INDEX = ").rstrip().rstrip(";"))
    except (OSError, ValueError):  # missing, or cut short by a crashed write: rebuild it from the files
        old = []
    old, now, rows = {e["symbol"]: e for e in old}, datetime.now().isoformat(timespec="seconds"), []
    for g, m in GROUPS.items():
        for t, n in m.items():
            f, e = OUT / "data" / f"{slug[t]}.js", old.get(t, {})
            mtime = f.stat().st_mtime if f.exists() else None
            if t in done:
                date, built = done[t], now
            elif mtime is None:
                date = built = None
            elif e.get("mtime") == mtime and e.get("date"):
                date, built = e["date"], e["built"]
            else:
                date, built = file_date(f), datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
            rows.append({"symbol": t, "slug": slug[t], "name": n, "group": g, "date": date, "built": built, "mtime": mtime})
    write_js(INDEX, "window.INDEX = [\n" + ",\n".join(map(json.dumps, rows)) + "\n];\n")
    try:
        old = json.loads(SUMMARY.read_text(encoding="utf-8").removeprefix("window.SUMMARY = ").rstrip().rstrip(";"))
    except (OSError, ValueError):
        old = {}
    keep = {t: sums.get(t) or old.get(t) for t in NAMES if sums.get(t) or old.get(t)}
    write_js(SUMMARY, "window.SUMMARY = {\n" + ",\n".join(f"{json.dumps(t)}: {json.dumps(v)}" for t, v in keep.items()) + "\n};\n")


def write_js(f, text):
    tmp = f.with_suffix(".js.tmp")  # replaced in one step, so the page or another run never reads half a file
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, f)


def serve(port):
    """This folder on http://127.0.0.1:port, plus POST /build?symbol=X: builds that one market, one build at a time,
    and answers once it is done."""
    busy = threading.Lock()
    # The names a browser on this computer sends; any other Host is another site's rebound DNS name
    hosts = {f"{h}:{port}" for h in ("127.0.0.1", "localhost")} | ({"127.0.0.1", "localhost"} if port == 80 else set())

    class Handler(SimpleHTTPRequestHandler):
        def send_head(self):  # GET and HEAD
            path = urlsplit(self.path).path
            if self.headers.get("Host") not in hosts or any(p.startswith(".") for p in path.split("/")):
                self.send_error(404)  # also keeps .git and other dot folders private
                return None
            return super().send_head()

        def list_directory(self, path):
            self.send_error(404)
            return None

        def do_POST(self):
            u = urlsplit(self.path)
            sym = parse_qs(u.query).get("symbol", [""])[0]
            host = self.headers.get("Host")
            # Only this page may ask: another site open in the browser sends another Origin
            if host not in hosts or self.headers.get("Origin", f"http://{host}") != f"http://{host}":
                return self.send_error(403)
            if u.path != "/build" or sym not in read_list()[0]:  # read each time, so a market added by --add can build
                return self.send_error(404, "Not a market in markets.csv")
            if not busy.acquire(blocking=False):
                return self.send_error(409, "Another build is running")
            try:
                r = subprocess.run([sys.executable, str(Path(__file__).resolve()), sym], cwd=OUT,
                                   capture_output=True, text=True, timeout=1800)
                ok, log = r.returncode == 0, r.stdout + r.stderr
            except subprocess.TimeoutExpired:  # a hung download must not hold the lock for good
                ok, log = False, "Build timed out after 30 minutes."
            finally:
                busy.release()
            body = json.dumps({"ok": ok, "log": log[-3000:]}).encode()
            self.send_response(200 if ok else 500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(OUT)))
    print(f"Open http://127.0.0.1:{port}/app.html (only this computer can reach it). Ctrl+C stops the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def add(sym, name, group):
    global NAMES, GROUPS
    with open(LIST, "a+", newline="", encoding="utf-8") as f:
        f.seek(0)
        if not f.read().endswith("\n"):  # a hand edit may drop the last newline
            f.write("\n")
        csv.writer(f, lineterminator="\n").writerow([sym, name, group])
    NAMES, GROUPS = read_list()
    print(f"Added {sym} ({name}) to {group} in markets.csv.")


def slugs(tickers):
    # Only [a-z0-9_.-]: app.html accepts nothing else in ?m=
    # and Windows refuses files named con, aux, nul ... (Continental is CON.DE)
    return {t: re.sub(r"^(con|prn|aux|nul|com\d|lpt\d)(?=\.|$)", r"\1_", re.sub(r"[^a-z0-9_.-]", "_", t.strip("^").lower())) for t in tickers}


def summary_only():
    """The home page numbers for every market in markets.csv with closes in data/closes.csv: no download, no pages."""
    close = pd.read_csv(STORE, index_col=0, parse_dates=True)
    data = {t: highs(s) for t in NAMES if t in close and not (s := close[t].dropna().loc[lambda x: x > 0]).empty}
    with ProcessPoolExecutor(min(61, os.process_cpu_count() or 1)) as pool:
        sums = dict(zip(data, pool.map(overview, data.values(), chunksize=8)))
    write_index(slugs(NAMES), {}, sums)
    print(f"Home page numbers for {len(sums)} markets.")
    if skipped := [t for t in NAMES if t not in data]:
        print(f"No closes in {STORE.name}, skipped: {' '.join(skipped)}")


def main():
    args, new = sys.argv[1:], None
    if args[:1] == ["--serve"]:
        return serve(int(args[1]) if len(args) > 1 else 8765)
    if args[:1] == ["--summary-only"]:
        return summary_only()
    if args[:1] == ["--add"]:
        if len(args) not in (3, 5) or (len(args) == 5 and args[3] != "--group"):
            sys.exit('Usage: python market_highs.py --add SYMBOL "Name" [--group GROUP]')
        sym, group = args[1].upper(), (args[4:] or ["Other US-listed"])[0]
        if sym in NAMES:
            print(f"{sym} is already in markets.csv as {NAMES[sym]}.")
        elif group not in GROUPS:
            sys.exit(f"Unknown group {group!r}: use one of {', '.join(GROUPS)}, or add a new group in markets.csv.")
        else:
            new = (sym, args[2], group)
        args = [sym]
    tickers = args or list(NAMES)
    closes = load(tickers)  # exits when Yahoo returns nothing
    data = {t: highs(s) for t, s in closes.items()}
    if new:
        add(*new)  # only once Yahoo knows the symbol, so a wrong one never lands in the list
    slug = slugs(NAMES | data)
    (OUT / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    (OUT / "data").mkdir(exist_ok=True)

    # Markets build in parallel (the back-test is the slow part); rows come back in list order
    # Windows allows at most 61 worker processes
    with ProcessPoolExecutor(min(61, os.process_cpu_count() or 1, len(data))) as pool:
        jobs = {t: pool.submit(build, t, d, slug) for t, d in data.items()}
    sums, failed = {}, []
    for t, job in jobs.items():
        try:
            sums[t] = job.result()
        except Exception as e:
            print(f"{t} failed:", file=sys.stderr)
            traceback.print_exception(e)  # includes the worker's traceback
            failed.append(t)
    write_index(slug, {t: str(data[t].index[-1].date()) for t in jobs if t not in failed}, sums)
    if failed:  # non-zero exit, so the scheduled run shows the failure
        sys.exit(f"{len(failed)} failed: {' '.join(failed)}")


if __name__ == "__main__":
    main()

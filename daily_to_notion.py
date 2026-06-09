# -*- coding: utf-8 -*-
"""
한국형 공포·탐욕지수 v2 -> 노션 자동 기록기
매일 1회 실행하면:
  1) 오늘의 '탐욕지수'(가격이 달리는 정도)와 '위험지수'(불안정한 정도)를 따로 계산
  2) 버핏지수(시총/GDP, 절대 밸류에이션), 코스피 PBR, 미국 CNN 공포탐욕지수 수집
  3) 노션 '일별 지수 기록' DB에 한 줄 기록 (같은 날짜가 있으면 갱신)

★ breadth(시장폭)·버핏 계산은 korea_fng.py 의 함수를 그대로 import 해서 사용.
  -> 두 파일을 같은 폴더에 두세요. (없으면 그 두 지표만 자동 제외되고 나머지는 동작)

★ 사전 준비 (한 번만):
  1) https://www.notion.so/my-integrations 에서 통합 생성 -> 토큰 복사
  2) 노션 DB 페이지 ... -> 연결(Connections) -> 위 통합 추가
  3) CMD에서:  setx NOTION_TOKEN "secret_xxxx"   (등록 후 새 CMD 필요)
  4) 설치:  pip install finance-datareader pykrx pandas numpy requests

실행:  python daily_to_notion.py
"""

import os
import sys
import json
import traceback
from datetime import date, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(HERE, "daily_to_notion.log")
sys.path.insert(0, HERE)  # 같은 폴더의 korea_fng.py 를 import 하기 위함


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ===== 사용자 환경 =====
DATABASE_ID = "d019a189f6f941b7b0838bd1917fa268"
NOTION_VERSION = "2022-06-28"
START = "2017-01-01"          # 장기창(1000일) 표준화를 위해 충분히 길게
LOOKBACK_SHORT = 252          # 평균회귀형 지표
LOOKBACK_LONG = 1000          # 레벨형 지표(시장폭·모멘텀)

# 두 지수의 가중치 (korea_fng.py v2 와 동일). 빠지는 지표는 자동 재정규화.
GREED_W = {"외국인수급": 0.30, "모멘텀": 0.25, "시장폭": 0.25, "신용융자": 0.20}
RISK_W = {"변동성": 0.50, "환율": 0.25, "안전자산이탈": 0.25}


def run():
    import numpy as np
    import pandas as pd
    import requests
    import FinanceDataReader as fdr
    from pykrx import stock

    # breadth·버핏 계산 함수를 korea_fng.py 에서 가져옴(단일 소스 유지)
    try:
        from korea_fng import update_breadth_cache, compute_buffett
        shared_ok = True
    except Exception as ex:
        shared_ok = False
        print(f"[WARN] korea_fng.py import 실패 -> 시장폭/버핏 제외: {type(ex).__name__}: {ex}")

    token = os.environ.get("NOTION_TOKEN")
    if not token:
        raise RuntimeError(
            "환경변수 NOTION_TOKEN 이 없습니다.\n"
            "  CMD에서  setx NOTION_TOKEN \"secret_...\"  실행 후 새 CMD에서 재시도."
        )

    END = date.today().isoformat()
    s, e = START.replace("-", ""), END.replace("-", "")

    def last_pct(series, window, invert=False):
        x = pd.Series(series).dropna().astype(float)
        if len(x) < 60:
            return None
        win = x.iloc[-window:]
        rank = float(win.rank(pct=True).iloc[-1]) * 100.0
        return (100 - rank) if invert else rank

    greed_c, risk_c, dropped = {}, {}, []

    def add(bucket, name, fn):
        try:
            v = fn()
            if v is None or (isinstance(v, float) and np.isnan(v)):
                raise ValueError("값 없음")
            bucket[name] = round(float(v), 1)
            print(f"  [OK]   {name:<12} {bucket[name]}")
        except Exception as ex:
            dropped.append(name)
            print(f"  [DROP] {name:<12} ({type(ex).__name__}: {ex})")

    print("지표 계산 중...")

    kospi = stock.get_index_ohlcv(s, e, "1001")
    kospi.index = pd.to_datetime(kospi.index)
    close = kospi["종가"]
    last_day = close.index[-1]

    # ── 탐욕축 ───────────────────────────────────────────────
    def g_foreign():
        fv = stock.get_market_trading_value_by_date(s, e, "KOSPI", on="순매수")
        cols = [c for c in fv.columns if "외국인" in str(c) and "기타" not in str(c)]
        if not cols:
            raise ValueError(f"외국인 컬럼 없음: {list(fv.columns)}")
        ser = fv[cols[0]].astype(float)
        ser.index = pd.to_datetime(ser.index)
        return last_pct(ser.rolling(20).sum(), LOOKBACK_SHORT)
    add(greed_c, "외국인수급", g_foreign)

    def g_mom():
        ma = close.rolling(120).mean()
        return last_pct(close / ma - 1.0, LOOKBACK_LONG)
    add(greed_c, "모멘텀", g_mom)

    def g_breadth():
        if not shared_ok:
            raise RuntimeError("korea_fng 미연결")
        ser = update_breadth_cache(stock, pd, last_day)
        if ser is None or ser.empty:
            raise ValueError("breadth 캐시 비어있음(누적 전)")
        return last_pct(ser, LOOKBACK_LONG)
    add(greed_c, "시장폭", g_breadth)

    def g_credit():
        df = pd.read_csv(os.path.join(HERE, "credit.csv"),
                         parse_dates=["date"]).set_index("date")["balance"]
        return last_pct(df, LOOKBACK_SHORT)
    add(greed_c, "신용융자", g_credit)

    # ── 위험축 (높을수록 위험) ────────────────────────────────
    def r_vol():
        try:
            vk = stock.get_index_ohlcv(s, e, "1000")["종가"]
            vk.index = pd.to_datetime(vk.index)
            v = last_pct(vk, LOOKBACK_SHORT)   # invert 안 함 = 변동성↑ -> 위험↑
            if v is not None:
                return v
        except Exception:
            pass
        rv = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
        return last_pct(rv, LOOKBACK_SHORT)
    add(risk_c, "변동성", r_vol)

    def r_fx():
        fx = fdr.DataReader("USD/KRW", START, END)["Close"]
        return last_pct(fx, LOOKBACK_SHORT)    # 원화약세(환율↑) -> 위험↑
    add(risk_c, "환율", r_fx)

    def r_safe():
        sret = close.pct_change(20) * 100
        try:
            bond = fdr.DataReader("KR10YT=RR", START, END)["Close"]
            bond.index = pd.to_datetime(bond.index)
            diff = (sret - bond.pct_change(20) * 100).dropna()
        except Exception:
            diff = sret
        return last_pct(diff, LOOKBACK_SHORT, invert=True)  # 채권우위 -> 위험↑
    add(risk_c, "안전자산이탈", r_safe)

    if not greed_c and not risk_c:
        raise RuntimeError("모든 지표 수집 실패 — 네트워크/라이브러리 확인")

    # ── 두 지수 합성 (사용 가능 지표만 재정규화 + 데이터 충분성 가드) ──
    #   기존에도 '있는 지표만 재정규화'라 멈춤 문제는 없었다(스칼라 계산).
    #   다만 korea_fng.py 의 compose 와 동작을 일치시키기 위해, 그날 유효 가중치
    #   합이 전체의 min_w_frac(기본 60%) 미만이면(=대부분 지표 수집 실패) 신뢰도가
    #   낮으므로 None 을 반환한다(1개 지표만으로 만든 엉뚱한 값 기록 방지).
    def compose(bucket, wtab, min_w_frac=0.6):
        if not bucket:
            return None
        total_w   = sum(wtab.values())            # 전체 가중치(빠진 지표 포함)
        present_w = sum(wtab[k] for k in bucket)  # 그날 존재하는 지표 가중치 합
        if present_w < total_w * min_w_frac:      # 데이터 60% 미만이면 기록 보류
            return None
        return round(sum(bucket[k] * (wtab[k] / present_w) for k in bucket), 1)

    greed = compose(greed_c, GREED_W)
    risk = compose(risk_c, RISK_W)

    # ── 버핏지수(절대 밸류에이션) ──
    buffett_ratio, buffett_pt = (None, None)
    if shared_ok:
        try:
            buffett_ratio, buffett_pt, buffett_err = compute_buffett(stock, pd, last_day)
            if buffett_ratio is None:
                print(f"  [DROP] 버핏지수 ({buffett_err})")
        except Exception as ex:
            print(f"  [DROP] 버핏지수 ({type(ex).__name__}: {ex})")

    def g_label(v):
        if v is None:
            return "중립"
        return ("극단적 공포" if v < 25 else "공포" if v < 45 else "중립"
                if v < 55 else "탐욕" if v < 75 else "극단적 탐욕")
    state = g_label(greed)

    print(f"\n탐욕지수 {greed}  ({state})   위험지수 {risk}", end="")
    if buffett_ratio is not None:
        print(f"   버핏 {buffett_ratio:.1f}% -> {buffett_pt:.0f}점", end="")
    print(f"   (탐욕 {len(greed_c)}개·위험 {len(risk_c)}개 사용)")

    # ── 코스피 PBR ──
    pbr = None
    try:
        fund = stock.get_index_fundamental(s, e, "1001")
        pbr = round(float(fund["PBR"].dropna().iloc[-1]), 2)
        print(f"  코스피 PBR {pbr}")
    except Exception as ex:
        print(f"  [DROP] 코스피PBR ({ex})")

    # ── 미국 CNN 공포탐욕지수 ──
    us_fng = None
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0.0.0 Safari/537.36"),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://edition.cnn.com/markets/fear-and-greed",
            "Origin": "https://edition.cnn.com",
        }
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        us_fng = round(float(r.json()["fear_and_greed"]["score"]), 1)
        print(f"  미국 FNG {us_fng}")
    except Exception as ex:
        print(f"  [DROP] 미국FNG ({ex})")

    # ===== 노션 기록 =====
    today = END
    summary = f"{today} · 탐욕 {greed} / 위험 {risk}"
    memo = "자동기록(v2: 위험축 지표는 높을수록 위험 방향)"
    if dropped:
        memo += " · 제외: " + ", ".join(dropped)

    def num(x):
        return {"number": (None if x is None else float(x))}

    props = {
        "요약": {"title": [{"text": {"content": summary}}]},
        "날짜": {"date": {"start": today}},
        "탐욕지수": num(greed),
        "위험지수": num(risk),
        "종합지수": num(greed),            # 기존 차트/뷰 호환용(=탐욕지수)
        "심리상태": {"select": {"name": state}},
        "버핏지수": num(buffett_pt),
        "버핏비율": num(buffett_ratio),
        "코스피PBR": num(pbr),
        "미국FNG": num(us_fng),
        "메모": {"rich_text": [{"text": {"content": memo}}]},
    }
    # 하위 지표(진단용). 위험축 3개는 v2 기준(높을수록 위험) 값으로 저장.
    comp_all = {}
    comp_all.update(greed_c)
    comp_all.update(risk_c)
    col_map = {"외국인수급": "외국인수급", "모멘텀": "모멘텀", "시장폭": "시장폭",
               "신용융자": "신용융자", "변동성": "변동성", "환율": "환율",
               "안전자산이탈": "안전자산선호"}   # '이탈' 값을 기존 '안전자산선호' 컬럼에 저장
    for k, col in col_map.items():
        props[col] = num(comp_all.get(k))

    api_headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

    q = requests.post(
        f"https://api.notion.com/v1/databases/{DATABASE_ID}/query",
        headers=api_headers,
        data=json.dumps({"filter": {"property": "날짜", "date": {"equals": today}}}),
        timeout=15,
    )
    if q.status_code != 200:
        raise RuntimeError(f"노션 조회 실패 {q.status_code}: {q.text}")
    existing = q.json().get("results", [])

    if existing:
        page_id = existing[0]["id"]
        r = requests.patch(
            f"https://api.notion.com/v1/pages/{page_id}",
            headers=api_headers, data=json.dumps({"properties": props}), timeout=15,
        )
        action = "갱신"
    else:
        r = requests.post(
            "https://api.notion.com/v1/pages",
            headers=api_headers,
            data=json.dumps({"parent": {"database_id": DATABASE_ID}, "properties": props}),
            timeout=15,
        )
        action = "신규 기록"

    if r.status_code not in (200, 201):
        raise RuntimeError(f"노션 기록 실패 {r.status_code}: {r.text}")

    log(f"[완료] 노션 {action}: {summary}")


def main():
    ok = False
    try:
        run()
        ok = True
    except Exception:
        log("오류 발생:")
        tb = traceback.format_exc()
        print("\n" + "=" * 60)
        print("오류 발생. 아래 전체 내용을 복사해 공유해 주세요:")
        print("=" * 60)
        print(tb)
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(tb + "\n")
        except Exception:
            pass
    finally:
        if sys.stdin and sys.stdin.isatty():
            try:
                input("\n[Enter] 키를 누르면 종료합니다...")
            except EOFError:
                pass
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

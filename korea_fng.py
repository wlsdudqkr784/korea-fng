# -*- coding: utf-8 -*-
"""
한국형 공포·탐욕지수 v2  (탐욕 / 위험 / 밸류에이션 3축 분리판)
────────────────────────────────────────────────────────────
바뀐 점 (v1 -> v2)
  1) 단일 지수 -> 두 시계열로 분리
       · 탐욕지수 : 가격이 얼마나 달리나   (외국인수급·모멘텀·시장폭·신용융자)
       · 위험지수 : 얼마나 불안정한 구간인가 (변동성·환율·안전자산이탈)
     KOSPI 8000 + VKOSPI 71 처럼 두 신호가 동시에 극단일 때, 단일지수에서는
     상쇄되어 '중립'으로 보이던 것을 각각 정직하게 표시한다.
  2) 시장폭을 '지수 52주 위치'(반도체 왜곡) -> '전종목 중 상승종목 비율'(진짜
     횡단면 breadth)로 교체. breadth_cache.csv 에 매일 누적 저장한다.
  3) 룩백 차등: 시장폭(레벨형)은 장기창, 평균회귀형(변동성·환율 등)은 252일.
  4) 버핏지수(시가총액/명목GDP) 추가 — 백분위로 '정규화'하지 않는 '절대 밸류에이션'
     축. 지속 과열이 롤링 백분위에 의해 지워지는 문제를 보완하는 앵커 역할.

한 지표가 실패해도 그 지표만 빼고 가중치를 재정규화하여 계속 동작한다.
에러가 나도 창이 닫히지 않고 정확한 원인을 출력한다(Windows 더블클릭 대비).

설치:  pip install finance-datareader pykrx pandas numpy requests
실행:  python korea_fng.py
"""

import os
import sys
import json
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
BREADTH_CACHE = os.path.join(HERE, "breadth_cache.csv")   # date,pct_positive

# ===== 사용자가 손볼 수 있는 설정 =====
START = "2017-01-01"
LOOKBACK_SHORT = 252      # 평균회귀형 지표(변동성·환율·수급 등) 표준화 창
LOOKBACK_LONG  = 1000     # 레벨형 지표(시장폭·모멘텀) 표준화 창 ≈ 4년

# ★ 한국 명목 GDP(원). 분기/연 단위로만 바뀌므로 1년에 한 번 갱신하면 충분.
#   출처: 한국은행/통계청 '명목 국내총생산'. (2024 약 2,549조원 → 여유있게 갱신)
KR_NOMINAL_GDP_KRW = 2_600e12

# 버핏지수(%) -> 0~100 점수 매핑 구간. 한국 증시 역사 범위로 보정(보수적).
#   low 이하=저평가(0), high 이상=극단고평가(100). 절대값이라 백분위를 안 씀.
BUFFETT_LOW, BUFFETT_HIGH = 70.0, 130.0

# 지수별 가중치 (사용 가능한 지표만으로 매번 재정규화됨)
GREED_W = {"외국인수급": 0.30, "모멘텀": 0.25, "시장폭": 0.25, "신용융자": 0.20}
RISK_W  = {"변동성": 0.50, "환율": 0.25, "안전자산이탈": 0.25}


def run():
    import numpy as np
    import pandas as pd
    import FinanceDataReader as fdr
    from pykrx import stock

    END = pd.Timestamp.today().strftime("%Y-%m-%d")
    s, e = START.replace("-", ""), END.replace("-", "")

    # ---- 0~100 백분위: 현재값이 과거 `window`거래일 중 어느 위치인가 ----
    def pct_rank(series, window, invert=False):
        x = series.astype(float)
        r = x.rolling(window, min_periods=60).apply(
            lambda w: w.rank(pct=True).iloc[-1] * 100.0, raw=False)
        return (100 - r) if invert else r

    # 지수별 컴포넌트 보관
    greed, risk, errors = {}, {}, {}

    def add(bucket, name, weight_table, fn):
        try:
            comp = fn().dropna()
            if comp.empty:
                raise ValueError("수집 데이터가 비어 있음")
            bucket[name] = comp
            print(f"  [OK]   {name:<12} ({len(comp)}일)")
        except Exception as ex:
            errors[name] = f"{type(ex).__name__}: {ex}"
            print(f"  [FAIL] {name:<12} -> {errors[name]}")

    print("데이터 수집 중...")

    # 기반: 코스피 종가
    kospi = stock.get_index_ohlcv(s, e, "1001")
    kospi.index = pd.to_datetime(kospi.index)
    close = kospi["종가"]

    # ── 탐욕축 ───────────────────────────────────────────────
    # 1) 외국인수급: 20일 누적 순매수, 많이 살수록 탐욕
    def f_foreign():
        fv = stock.get_market_trading_value_by_date(s, e, "KOSPI", on="순매수")
        cols = [c for c in fv.columns if "외국인" in str(c) and "기타" not in str(c)]
        if not cols:
            raise ValueError(f"외국인 컬럼 없음: {list(fv.columns)}")
        ser = fv[cols[0]].astype(float)
        ser.index = pd.to_datetime(ser.index)
        return pct_rank(ser.rolling(20).sum(), LOOKBACK_SHORT)
    add(greed, "외국인수급", GREED_W, f_foreign)

    # 2) 모멘텀: 종가/120일선-1 (장기창; 비율이라 효과는 보조적)
    def f_momentum():
        ma = close.rolling(120).mean()
        return pct_rank(close / ma - 1.0, LOOKBACK_LONG)
    add(greed, "모멘텀", GREED_W, f_momentum)

    # 3) 시장폭(진짜 breadth): 전종목 중 상승종목 비율. breadth_cache.csv 누적.
    def f_breadth():
        ser = update_breadth_cache(stock, pd, close.index[-1])
        if ser is None or ser.empty:
            raise ValueError("breadth 캐시가 비어있음(누적 전). 다음 실행부터 채워집니다")
        return pct_rank(ser, LOOKBACK_LONG)
    add(greed, "시장폭", GREED_W, f_breadth)

    # 4) 신용융자: credit.csv(date,balance), 증가할수록 탐욕
    def f_credit():
        df = pd.read_csv(os.path.join(HERE, "credit.csv"),
                         parse_dates=["date"]).set_index("date")["balance"]
        return pct_rank(df, LOOKBACK_SHORT)
    add(greed, "신용융자", GREED_W, f_credit)

    # ── 위험축 (높을수록 위험) ────────────────────────────────
    # 5) 변동성: VKOSPI(실패 시 실현변동성). 높을수록 위험(=invert 안 함)
    def f_vol():
        try:
            vk = stock.get_index_ohlcv(s, e, "1000")["종가"]
            vk.index = pd.to_datetime(vk.index)
            return pct_rank(vk, LOOKBACK_SHORT)
        except Exception:
            rv = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
            return pct_rank(rv, LOOKBACK_SHORT)
    add(risk, "변동성", RISK_W, f_vol)

    # 6) 환율: 원/달러 높을수록(원화약세) 위험
    def f_fx():
        fx = fdr.DataReader("USD/KRW", START, END)["Close"]
        fx.index = pd.to_datetime(fx.index)
        return pct_rank(fx, LOOKBACK_SHORT)
    add(risk, "환율", RISK_W, f_fx)

    # 7) 안전자산이탈: 채권이 주식을 이길수록 위험(=주식-채권 수익률차를 invert)
    def f_safe():
        sret = close.pct_change(20) * 100
        try:
            bond = fdr.DataReader("KR10YT=RR", START, END)["Close"]
            bond.index = pd.to_datetime(bond.index)
            diff = (sret - bond.pct_change(20) * 100).dropna()
        except Exception:
            diff = sret
        return pct_rank(diff, LOOKBACK_SHORT, invert=True)
    add(risk, "안전자산이탈", RISK_W, f_safe)

    if not greed and not risk:
        raise RuntimeError("모든 지표 수집 실패 — 네트워크/라이브러리 확인 필요")

    # ---- 두 지수 합성 (그날 존재하는 지표만으로 가중평균, 매일 재정규화) ----
    #   [왜 이렇게 바꿨나]
    #   예전 코드는 마지막에 .dropna() 를 했다. pd.DataFrame(bucket) 은 모든 지표를
    #   바깥조인(outer join)으로 합치므로, 어느 한 지표라도 비어있는 날짜는 가중합이
    #   NaN 이 되고 → .dropna() 가 그 날짜 전체를 삭제했다. 신용융자(credit.csv)는
    #   수기 데이터라 늘 며칠 늦는데, 그 결과 신용융자 마지막 날짜 이후가 통째로
    #   잘려나가 탐욕지수가 그 날짜에서 '멈추는' 문제가 있었다.
    #   → 그날 존재하는 지표만으로 가중평균을 내고, 그날 가중치를 재정규화한다.
    #     · 모든 지표가 있는 날: 예전과 값이 100% 동일.
    #     · 일부 지표만 있는 날: 남은 지표로 계속 산출 → 멈춤 해소.
    #     · 단, 그날 유효 가중치 합이 전체의 min_w_frac(기본 60%) 미만이면, 신뢰도가
    #       낮으므로 그 날짜는 제외(시리즈 맨 앞 1개 지표만 있을 때의 노이즈 방지).
    def compose(bucket, wtab, min_w_frac=0.6):
        if not bucket:
            return pd.Series(dtype=float)
        df = pd.DataFrame(bucket)                          # 날짜 × 지표 (바깥조인)
        w  = pd.Series({k: float(wtab[k]) for k in bucket})
        total_w = w.sum()
        present = df.notna()                               # 그날 지표 존재 여부
        wsum  = present.mul(w, axis=1).sum(axis=1)         # 그날 유효 가중치 합
        numer = df.mul(w, axis=1).sum(axis=1, skipna=True) # 분자(없는 지표는 skip)
        out = (numer / wsum).where(wsum >= total_w * min_w_frac)
        return out.dropna()

    greed_idx = compose(greed, GREED_W)
    risk_idx = compose(risk, RISK_W)

    # ---- 버핏지수(절대 밸류에이션, 오늘 1개 값) ----
    buffett_ratio, buffett_pt, buffett_err = compute_buffett(stock, pd, close.index[-1])
    if buffett_ratio is None:
        print(f"  [버핏 실패] {buffett_err}")

    def g_label(v):
        return ("극단공포" if v < 25 else "공포" if v < 45 else "중립"
                if v < 55 else "탐욕" if v < 75 else "극단탐욕")
    def r_label(v):
        return ("매우안정" if v < 25 else "안정" if v < 45 else "보통"
                if v < 55 else "주의" if v < 75 else "위험")

    print("\n" + "=" * 52)
    if not greed_idx.empty:
        g = greed_idx.iloc[-1]
        print(f"탐욕지수   {g:5.1f}  ->  {g_label(g)}   (지표 {len(greed)}개)")
    if not risk_idx.empty:
        r = risk_idx.iloc[-1]
        print(f"위험지수   {r:5.1f}  ->  {r_label(r)}   (지표 {len(risk)}개)")
    if buffett_ratio is not None:
        print(f"버핏지수   {buffett_ratio:5.1f}% (시총/GDP) -> 밸류점수 {buffett_pt:.0f}/100")
    print("=" * 52)
    if errors:
        print("제외된 지표:", ", ".join(f"{k}({v})" for k, v in errors.items()))

    # ---- history.json 내보내기 (차트용) ----
    #   kr  = 탐욕지수 (기존 차트 호환)
    #   risk= 위험지수,  us = 미국 CNN,  buffett = 오늘의 밸류에이션
    out = {
        "kr":   [{"d": d.strftime("%Y-%m-%d"), "v": round(float(v), 1)} for d, v in greed_idx.items()],
        "risk": [{"d": d.strftime("%Y-%m-%d"), "v": round(float(v), 1)} for d, v in risk_idx.items()],
    }
    if buffett_ratio is not None:
        out["buffett"] = {"d": END, "ratio": round(buffett_ratio, 1), "score": round(buffett_pt, 1)}

    us = fetch_us_fng_series(pd, list(greed_idx.index))
    if us is not None:
        out["us"] = us

    with open(os.path.join(HERE, "history.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"\n[export] history.json 저장 (탐욕 {len(out['kr'])}일, 위험 {len(out['risk'])}일"
          + (f", 미국 {sum(1 for x in us if x['v'] is not None)}일" if us else "") + ")")


# ────────────────────────────────────────────────────────────
# 보조 함수들
# ────────────────────────────────────────────────────────────
def update_breadth_cache(stock, pd, last_trading_day):
    """오늘 breadth(전종목 중 120일전 대비 상승종목 비율)를 1회 계산해 캐시에 누적.
       전체 시장을 한 번의 호출로 받는 get_market_price_change* 사용(가볍다)."""
    import numpy as np
    # 기존 캐시 로드
    if os.path.exists(BREADTH_CACHE):
        cache = pd.read_csv(BREADTH_CACHE, parse_dates=["date"]).set_index("date")["pct_positive"]
    else:
        cache = pd.Series(dtype=float)

    today = pd.Timestamp(last_trading_day).normalize()
    if today not in cache.index:
        frm = (today - pd.Timedelta(days=180)).strftime("%Y%m%d")  # ~120거래일 확보
        to = today.strftime("%Y%m%d")
        df = None
        fetch_err = "등락 함수 없음 (pykrx 버전 확인)"
        for fname in ("get_market_price_change_by_ticker", "get_market_price_change"):
            fn = getattr(stock, fname, None)
            if fn is None:
                continue
            try:
                df = fn(frm, to, "KOSPI")
                break
            except Exception as ex:
                fetch_err = f"{type(ex).__name__}: {ex}"
                continue
        if df is not None and not df.empty:
            col = next((c for c in df.columns if "등락" in str(c)), None)
            if col is not None:
                chg = df[col].astype(float)
                pct_pos = float((chg > 0).mean()) * 100.0
                cache.loc[today] = round(pct_pos, 2)
                cache.sort_index().to_csv(BREADTH_CACHE, header=["pct_positive"],
                                          index_label="date", encoding="utf-8")
                print(f"  [breadth] {today.date()} 상승종목 {pct_pos:.1f}% 캐시 저장")
        else:
            print(f"  [breadth] 오늘치 수집 실패: {fetch_err}")
    return cache.sort_index()


def compute_buffett(stock, pd, last_trading_day):
    """버핏지수 = 전체(코스피+코스닥) 시가총액 합 / 명목GDP × 100, 그리고 0~100 점수.
       반환: (ratio, score, err). 실패 시 (None, None, 이유문자열).
       get_market_cap_by_ticker(date, market='ALL') 한 번으로 전체 시총을 받는다."""
    import numpy as np
    fn = getattr(stock, "get_market_cap_by_ticker", None)
    if fn is None:
        return None, None, "get_market_cap_by_ticker 함수 없음 (pykrx 버전 확인)"
    base = pd.Timestamp(last_trading_day).normalize()
    last_err = "데이터 없음"
    for back in range(0, 10):                      # 휴장일 대비 최근 10일 뒤로 시도
        d = (base - pd.Timedelta(days=back)).strftime("%Y%m%d")
        try:
            df = fn(d, market="ALL")               # ALL = 코스피+코스닥
        except Exception as ex:
            last_err = f"{type(ex).__name__}: {ex}"   # 예: KRX 로그인 실패 등
            continue
        if df is None or len(df) == 0:
            last_err = f"{d}: 빈 데이터"
            continue
        cap_col = next((c for c in df.columns if "시가총액" in str(c)), None)
        if cap_col is None:
            return None, None, f"시가총액 컬럼 없음. 실제 컬럼: {list(df.columns)}"
        total = float(pd.to_numeric(df[cap_col], errors="coerce").sum())
        if total <= 0:
            last_err = f"{d}: 시총합이 0"
            continue
        ratio = total / KR_NOMINAL_GDP_KRW * 100.0
        score = float(np.clip((ratio - BUFFETT_LOW) / (BUFFETT_HIGH - BUFFETT_LOW) * 100.0, 0, 100))
        return ratio, score, None
    return None, None, last_err


def fetch_us_fng_series(pd, kr_dates):
    """미국 CNN 공포탐욕지수 이력을 한국 거래일 축에 맞춰 정렬."""
    try:
        import requests
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0.0.0 Safari/537.36"),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://edition.cnn.com/markets/fear-and-greed",
            "Origin": "https://edition.cnn.com",
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        raw = resp.json().get("fear_and_greed_historical", {}).get("data", []) or []
        if not raw:
            raise ValueError("미국 이력 비어 있음")
        m = {}
        for pt in raw:
            dt = pd.to_datetime(pt["x"], unit="ms").strftime("%Y-%m-%d")
            m[dt] = round(float(pt["y"]), 1)
        return [{"d": d.strftime("%Y-%m-%d"), "v": m.get(d.strftime("%Y-%m-%d"))} for d in kr_dates]
    except Exception as ex:
        print(f"[WARN] 미국 FNG 수집 실패: {type(ex).__name__}: {ex}")
        return None


def main():
    try:
        run()
    except Exception:
        print("\n" + "=" * 60)
        print("오류가 발생했습니다. 아래 전체 내용을 복사해 공유해 주세요:")
        print("=" * 60)
        traceback.print_exc()
    finally:
        try:
            input("\n[Enter] 키를 누르면 종료합니다...")
        except EOFError:
            pass


if __name__ == "__main__":
    main()

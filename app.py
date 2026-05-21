import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import time
from datetime import datetime
from scipy import stats as scipy_stats
from streamlit_autorefresh import st_autorefresh

from data_utils import (
    load_master, build_player_stats, build_features,
    SURFACE_FEATURES, compute_h2h,
    backtest_models, retrain_winner_model, retrain_bo3_model
)

# ═══════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="🎾 Tennis Predictions",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
div[data-testid="metric-container"] {
    background: #1e1e2e; border-radius: 8px; padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# МОДЕЛИ
# ═══════════════════════════════════════════════════════
MODEL_DIR  = "models"
BACKUP_DIR = "models/backup"

@st.cache_resource(show_spinner="🤖 Зарежда моделите...")
def load_models():
    m = {}
    for surf in ['Hard', 'Grass', 'Clay']:
        path = f"{MODEL_DIR}/winner_model_{surf}.joblib"
        try:
            m[surf] = joblib.load(path)
        except Exception as e:
            st.error(f"Грешка {surf}: {e}")
            m[surf] = None
    try:
        m['bo3'] = joblib.load(f"{MODEL_DIR}/tennis_model_bo3_final.pkl")
    except Exception as e:
        st.warning(f"BO3: {e}")
        m['bo3'] = None
    return m

MODELS  = load_models()
MASTER  = load_master()
STATS   = build_player_stats(MASTER)
PLAYERS = sorted(STATS.index.dropna().tolist())

# ═══════════════════════════════════════════════════════
# ПОМОЩНИ ФУНКЦИИ
# ═══════════════════════════════════════════════════════
def predict_match(p1, p2, surface):
    feats     = build_features(p1, p2, surface, STATS, MASTER)
    feat_list = SURFACE_FEATURES[surface]
    X         = pd.DataFrame([[feats[f] for f in feat_list]], columns=feat_list)
    prob_p1   = float(MODELS[surface].predict_proba(X)[0][1])
    bo3_pred  = None
    if MODELS['bo3'] is not None:
        X_bo3 = pd.DataFrame(
            [[feats['rank_diff'], feats['w_roll_hold'], feats['l_roll_hold']]],
            columns=['rank_diff', 'w_roll_hold', 'l_roll_hold']
        )
        bo3_pred = float(MODELS['bo3'].predict(X_bo3)[0])
    return prob_p1, bo3_pred, feats

def implied_prob(odds):
    return 1.0 / odds if odds > 1 else 0.0

def kelly(prob, odds):
    b = odds - 1
    k = (prob * odds - 1) / b if b > 0 else 0
    return max(0.0, k)

def edge_badge(edge):
    if edge > 0.05:   return "🟢", "Добра стойност"
    elif edge > 0.02: return "🟡", "Слаба стойност"
    else:             return "🔴", "Без стойност"


# ═══════════════════════════════════════════════════════
# ТАБОВЕ
# ═══════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Ръчна проверка",
    "📅 Дневна програма",
    "💰 Калкулатор",
    "🤖 Модели"
])


# ╔═══════════════════════════════════════════════════╗
# ║  ТАБ 1 – РЪЧНА ПРОВЕРКА                         ║
# ╚═══════════════════════════════════════════════════╝
with tab1:
    st.header("🔍 Ръчна проверка на мач")

    # ── Входни данни ─────────────────────────────────
    col1, col2, col3 = st.columns([3, 3, 2])
    with col1:
        p1 = st.selectbox("🎾 Играч 1", [""] + PLAYERS, key="t1_p1")
    with col2:
        p2 = st.selectbox("🎾 Играч 2", [""] + PLAYERS, key="t1_p2")
    with col3:
        surface = st.selectbox("🏟️ Настилка", ["Hard", "Clay", "Grass"], key="t1_surf")

    col4, col5 = st.columns(2)
    with col4:
        odds1 = st.number_input(
            f"💲 Коефициент — {p1 or 'Играч 1'}",
            min_value=1.01, max_value=50.0, value=1.85, step=0.05,
            format="%.2f", key="t1_odds1"
        )
    with col5:
        odds2 = st.number_input(
            f"💲 Коефициент — {p2 or 'Играч 2'}",
            min_value=1.01, max_value=50.0, value=2.10, step=0.05,
            format="%.2f", key="t1_odds2"
        )

    bankroll_t1 = st.number_input(
        "💳 Банка (лв.)", min_value=1.0, value=1000.0,
        step=50.0, format="%.2f", key="t1_bank"
    )

    # ── Анализ бутон ─────────────────────────────────
    can_run = bool(p1 and p2 and p1 != p2 and MODELS.get(surface))
    if st.button("🎯 Анализирай мача", type="primary",
                 disabled=not can_run, use_container_width=True):
        with st.spinner("Изчислява..."):
            prob_p1, bo3_pred, feats = predict_match(p1, p2, surface)
        # Запазва в session_state → резултатът оцелява при промяна на widgets
        st.session_state['t1_result'] = {
            'p1': p1, 'p2': p2, 'surface': surface,
            'prob_p1': prob_p1, 'bo3_pred': bo3_pred,
            'odds1': odds1, 'odds2': odds2,
        }

    # ── Показва резултата (от session_state) ─────────
    res = st.session_state.get('t1_result')

    # Нулираме резултата ако са сменени играчи/настилка
    if res and (res['p1'] != p1 or res['p2'] != p2 or res['surface'] != surface):
        del st.session_state['t1_result']
        res = None

    if res:
        prob_p1  = res['prob_p1']
        prob_p2  = 1 - prob_p1
        bo3_pred = res['bo3_pred']

        raw1, raw2 = implied_prob(odds1), implied_prob(odds2)
        margin     = raw1 + raw2
        impl1, impl2 = raw1 / margin, raw2 / margin
        edge1, edge2 = prob_p1 - impl1, prob_p2 - impl2

        # ── Победител ────────────────────────────────
        st.divider()
        st.subheader("📊 Предикция за победител")
        c1, c2 = st.columns(2)

        for col_w, player, prob_m, impl, edge, odds_bk in [
            (c1, p1, prob_p1, impl1, edge1, odds1),
            (c2, p2, prob_p2, impl2, edge2, odds2),
        ]:
            badge, txt = edge_badge(edge)
            with col_w:
                st.markdown(f"### 🎾 {player}")
                st.metric("Модел %",      f"{prob_m:.1%}")
                st.metric("Букмейкър %",  f"{impl:.1%}", delta=f"{edge:+.1%}")
                fair = 1 / prob_m if prob_m > 0 else 99
                st.caption(f"Честен коеф: **{fair:.2f}**  |  Букм: **{odds_bk:.2f}**")
                if edge > 0.05:   st.success(f"{badge} {txt} — {edge:.1%}")
                elif edge > 0.02: st.info(f"{badge} {txt} — {edge:.1%}")
                else:             st.error(f"{badge} {txt} — {edge:.1%}")

        # ── Over / Under геймове ─────────────────────
        if bo3_pred is not None:
            st.divider()
            st.subheader("🎲 Геймове — Over / Under")
            gc1, gc2 = st.columns([1, 2])
            with gc1:
                st.metric("Очаквани геймове", f"{bo3_pred:.1f}")
                # Тези widgets са ИЗВЪН if-button блока → не презареждат
                game_line = st.number_input(
                    "Линия (Over/Under)",
                    min_value=14.0, max_value=42.0,
                    value=21.5, step=0.5, format="%.1f", key="t1_gameline"
                )
            with gc2:
                std     = 3.0
                p_over  = float(1 - scipy_stats.norm.cdf(game_line, loc=bo3_pred, scale=std))
                p_under = 1.0 - p_over
                st.metric(f"Over  {game_line}", f"{p_over:.1%}")
                st.metric(f"Under {game_line}", f"{p_under:.1%}")
                st.progress(int(p_over * 100), text="⬆ Over ─── Under ⬇")

        # ── Калкулатор на залог ──────────────────────
        st.divider()
        st.subheader("💰 Препоръчан залог")

        best_edge   = max(edge1, edge2)
        best_player = p1 if edge1 >= edge2 else p2
        best_prob   = prob_p1 if edge1 >= edge2 else prob_p2
        best_odds   = odds1  if edge1 >= edge2 else odds2
        kelly_f     = kelly(best_prob, best_odds)

        # Widget е ИЗВЪН if-button → не презарежда при смяна
        method_t1 = st.radio(
            "Метод",
            ["½ Kelly (препоръчан)", "Full Kelly", "¼ Kelly", "Фиксиран %"],
            horizontal=True, key="t1_method"
        )

        if method_t1 == "Full Kelly":
            frac = kelly_f
        elif method_t1 == "½ Kelly (препоръчан)":
            frac = kelly_f / 2
        elif method_t1 == "¼ Kelly":
            frac = kelly_f / 4
        else:
            pct_t1 = st.slider("% от банката", 1, 15, 3, key="t1_pct")
            frac = pct_t1 / 100

        bet_amt = bankroll_t1 * frac
        profit  = bet_amt * (best_odds - 1)

        if kelly_f <= 0 and method_t1 != "Фиксиран %":
            st.error("❌ Отрицателна стойност — **не залагай**!")
        else:
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric(f"Залог — {best_player}", f"{bet_amt:.2f} лв.",
                       f"{frac:.1%} от банката")
            bc2.metric("Потенциална печалба", f"+{profit:.2f} лв.")
            bc3.metric("Edge", f"{best_edge:+.1%}")

        # ── Детайли ──────────────────────────────────
        with st.expander("📈 Входни данни за модела"):
            s1 = STATS.loc[p1] if p1 in STATS.index else pd.Series(dtype=float)
            s2 = STATS.loc[p2] if p2 in STATS.index else pd.Series(dtype=float)
            detail = {
                "Метрика": ["Ранглиста", f"ELO ({surface})", "1st Serve %",
                            "Форма (last 10)", "Почивка (дни)", "Възраст"],
                p1: [int(s1.get('rank', 999) or 999),
                     f"{float(s1.get(f'elo_{surface}', 1500) or 1500):.0f}",
                     f"{float(s1.get('roll_first', 0.65) or 0.65):.1%}",
                     f"{float(s1.get('roll_form', 0.5) or 0.5):.1%}",
                     int(s1.get('rest_days', 0) or 0),
                     f"{s1.get('age', '?')}"],
                p2: [int(s2.get('rank', 999) or 999),
                     f"{float(s2.get(f'elo_{surface}', 1500) or 1500):.0f}",
                     f"{float(s2.get('roll_first', 0.65) or 0.65):.1%}",
                     f"{float(s2.get('roll_form', 0.5) or 0.5):.1%}",
                     int(s2.get('rest_days', 0) or 0),
                     f"{s2.get('age', '?')}"],
            }
            st.dataframe(pd.DataFrame(detail), hide_index=True, use_container_width=True)


# ╔═══════════════════════════════════════════════════╗
# ║  ТАБ 2 – ДНЕВНА ПРОГРАМА                        ║
# ╚═══════════════════════════════════════════════════╝
with tab2:
    st_autorefresh(interval=6 * 60 * 60 * 1000, key="sched_refresh")
    st.header("📅 Дневна програма (ATP)")

    ODDS_KEY = st.secrets.get("ODDS_API_KEY", "")

    if not ODDS_KEY:
        st.error("❌ Липсва ODDS_API_KEY в Streamlit Secrets.")
        st.code('ODDS_API_KEY = "твоя_ключ"', language="toml")
    else:
        col_r, col_s = st.columns([1, 2])
        with col_r:
            if st.button("🔄 Обнови сега", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col_s:
            sched_surf = st.selectbox("Настилка по подразбиране",
                                      ["Hard", "Clay", "Grass"], key="t2_surf")

        # ── Стъпка 1: Намери всички активни тенис спортове ──
        @st.cache_data(ttl=21600, show_spinner="📡 Зарежда активни турнири...")
        def fetch_tennis_sports(key):
            r = requests.get(
                "https://api.the-odds-api.com/v4/sports/",
                params={"apiKey": key},
                timeout=15
            )
            all_sports = r.json()
            # Филтрира само активни ATP тенис спортове
            return [
                s['key'] for s in all_sports
                if isinstance(s, dict)
                and s.get('active', False)
                and 'tennis' in s.get('key', '').lower()
                and 'atp' in s.get('key', '').lower()
            ]

        # ── Стъпка 2: Вземи коефициенти за всеки намерен спорт ──
        @st.cache_data(ttl=21600, show_spinner="📡 Зарежда коефициенти...")
        def fetch_all_odds(key, sport_keys):
            all_matches = []
            for sport_key in sport_keys:
                try:
                    r = requests.get(
                        f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                        params={
                            "apiKey": key,
                            "regions": "eu",
                            "markets": "h2h",
                            "oddsFormat": "decimal"
                        },
                        timeout=15
                    )
                    data = r.json()
                    if isinstance(data, list):
                        all_matches.extend(data)
                except Exception:
                    pass
            return all_matches

        try:
            tennis_sports = fetch_tennis_sports(ODDS_KEY)
        except Exception as e:
            st.error(f"Грешка при зареждане на спортове: {e}")
            tennis_sports = []

        if not tennis_sports:
            st.warning("⚠️ Няма активни ATP тенис турнири в момента.")
            st.caption("TheOddsAPI предлага коефициенти само по време на активни турнири.")
        else:
            st.caption(f"🏆 Активни турнири: `{'`, `'.join(tennis_sports)}`")
            try:
                odds_data = fetch_all_odds(ODDS_KEY, tennis_sports)
            except Exception as e:
                st.error(f"Грешка при зареждане: {e}")
                odds_data = []

            if not odds_data:
                st.warning("Няма намерени мачове с коефициенти в момента.")
            else:
                st.success(
                    f"✅ {len(odds_data)} мача  "
                    f"|  Обновено: {datetime.now().strftime('%H:%M:%S')}"
                )

                def find_player(name):
                    last = name.split()[-1].lower()
                    hits = [p for p in PLAYERS if last in p.lower()]
                    return hits[0] if hits else None

                for match in odds_data:
                    home  = match.get("home_team", "")
                    away  = match.get("away_team", "")
                    start = match.get("commence_time", "")[:16].replace("T", " ")

                    best_h, best_a = 1.0, 1.0
                    for bk in match.get("bookmakers", []):
                        for mkt in bk.get("markets", []):
                            if mkt["key"] == "h2h":
                                for o in mkt["outcomes"]:
                                    if   o["name"] == home: best_h = max(best_h, o["price"])
                                    elif o["name"] == away: best_a = max(best_a, o["price"])

                    with st.expander(f"🎾  {home}  vs  {away}  |  {start} UTC"):
                        cm1, cm2 = st.columns(2)
                        cm1.metric(home, f"@ {best_h:.2f}")
                        cm2.metric(away, f"@ {best_a:.2f}")

                        p1f = find_player(home)
                        p2f = find_player(away)

                        if p1f and p2f:
                            surf_key = f"t2_surf_{home}_{away}".replace(" ", "_")
                            sc = st.selectbox(
                                "Настилка", ["Hard", "Clay", "Grass"],
                                key=surf_key,
                                index=["Hard", "Clay", "Grass"].index(sched_surf)
                            )
                            if MODELS.get(sc):
                                prob, bo3p, _ = predict_match(p1f, p2f, sc)
                                impl_h = (1/best_h) / ((1/best_h) + (1/best_a))
                                edge_h = prob - impl_h
                                badge, txt = edge_badge(edge_h)
                                cr1, cr2, cr3 = st.columns(3)
                                cr1.metric(f"Модел {p1f.split()[-1]}", f"{prob:.1%}")
                                cr2.metric("Edge", f"{edge_h:+.1%}")
                                cr3.metric("Честен коеф.", f"{1/prob:.2f}" if prob > 0 else "N/A")
                                if bo3p:
                                    st.caption(f"📊 Очаквани геймове: **{bo3p:.1f}**")
                                st.caption(f"{badge} {txt}  |  {p1f} vs {p2f}")
                        else:
                            missing = [n for n, f in [(home, p1f), (away, p2f)] if not f]
                            st.caption(f"⚠️ Не е намерен в базата: {', '.join(missing)}")


# ╔═══════════════════════════════════════════════════╗
# ║  ТАБ 3 – КАЛКУЛАТОР                             ║
# ╚═══════════════════════════════════════════════════╝
with tab3:
    st.header("💰 Калкулатор на залог")

    bankroll_t3 = st.number_input(
        "💳 Банка (лв.)", min_value=1.0, value=1000.0,
        step=100.0, format="%.2f", key="t3_bank"
    )
    st.divider()

    ca, cb = st.columns(2)
    with ca:
        model_prob_t3 = st.slider("📈 Вероятност от модела (%)", 1, 99, 60, key="t3_prob") / 100
    with cb:
        bet_odds_t3 = st.number_input(
            "💲 Коефициент на букмейкъра",
            min_value=1.01, max_value=50.0, value=2.10,
            step=0.05, format="%.2f", key="t3_odds"
        )

    method_t3 = st.radio(
        "📐 Метод",
        ["½ Kelly (препоръчан)", "Full Kelly", "¼ Kelly", "Фиксиран %", "Фиксирана сума"],
        horizontal=True, key="t3_method"
    )

    impl_t3  = implied_prob(bet_odds_t3)
    edge_t3  = model_prob_t3 - impl_t3
    kelly_t3 = kelly(model_prob_t3, bet_odds_t3)

    st.divider()

    if kelly_t3 <= 0 and method_t3 in ["½ Kelly (препоръчан)", "Full Kelly", "¼ Kelly"]:
        st.error("❌ Отрицателна стойност — **не залагай**!")
        st.metric("Edge", f"{edge_t3:.1%}")
    else:
        if method_t3 == "Full Kelly":
            frac_t3, lbl = kelly_t3, "Full Kelly"
        elif method_t3 == "½ Kelly (препоръчан)":
            frac_t3, lbl = kelly_t3 / 2, "½ Kelly"
        elif method_t3 == "¼ Kelly":
            frac_t3, lbl = kelly_t3 / 4, "¼ Kelly"
        elif method_t3 == "Фиксиран %":
            p_t3 = st.slider("% от банката", 1, 20, 3, key="t3_pct")
            frac_t3, lbl = p_t3 / 100, f"{p_t3}%"
        else:
            fx = st.number_input("Сума (лв.)", min_value=1.0, value=50.0,
                                 step=10.0, key="t3_fx")
            frac_t3, lbl = fx / bankroll_t3, f"{fx:.0f} лв."

        bet_t3    = bankroll_t3 * frac_t3
        profit_t3 = bet_t3 * (bet_odds_t3 - 1)
        ev_t3     = edge_t3 * bet_t3 * bet_odds_t3

        cr1, cr2, cr3 = st.columns(3)
        cr1.metric(f"💰 Залог ({lbl})", f"{bet_t3:.2f} лв.", f"{frac_t3:.1%} от банката")
        cr2.metric("📈 Потенциална печалба", f"+{profit_t3:.2f} лв.")
        cr3.metric("🎯 EV", f"{ev_t3:.2f} лв.")

        st.divider()
        d1, d2, d3 = st.columns(3)
        d1.metric("Edge",          f"{edge_t3:+.1%}")
        d2.metric("Модел %",       f"{model_prob_t3:.1%}")
        d3.metric("Букмейкър %",   f"{impl_t3:.1%}")

        badge_t3, txt_t3 = edge_badge(edge_t3)
        msg = f"{badge_t3} {txt_t3} — ръб {edge_t3:.1%}"
        if edge_t3 > 0.05:   st.success(msg)
        elif edge_t3 > 0.02: st.info(msg)
        else:                 st.warning(msg)

        with st.expander("📊 Сравнение Kelly варианти"):
            comp = {
                "Метод":              ["¼ Kelly", "½ Kelly", "Full Kelly"],
                "Залог (лв.)":        [f"{bankroll_t3*kelly_t3/4:.2f}", f"{bankroll_t3*kelly_t3/2:.2f}", f"{bankroll_t3*kelly_t3:.2f}"],
                "% от банката":       [f"{kelly_t3/4:.1%}", f"{kelly_t3/2:.1%}", f"{kelly_t3:.1%}"],
                "Потенциална печалба": [f"{bankroll_t3*kelly_t3/4*(bet_odds_t3-1):.2f} лв.",
                                        f"{bankroll_t3*kelly_t3/2*(bet_odds_t3-1):.2f} лв.",
                                        f"{bankroll_t3*kelly_t3*(bet_odds_t3-1):.2f} лв."],
            }
            st.dataframe(pd.DataFrame(comp), hide_index=True, use_container_width=True)


# ╔═══════════════════════════════════════════════════╗
# ║  ТАБ 4 – МОДЕЛИ                                 ║
# ╚═══════════════════════════════════════════════════╝
with tab4:
    st.header("🤖 Управление на моделите")
    st.info("""
**Backtest** → тества точността на последните N месеца реални мачове  
**Преобучаване** → тренира с новите данни от базата на Джеф  
**Rollback** → връща предишната версия при лоши резултати
    """)

    # ── Backtest ─────────────────────────────────────
    st.subheader("📊 Backtest")
    bt_months = st.slider("Период (месеци назад)", 1, 12, 6, key="t4_months")

    if st.button("▶️ Стартирай Backtest", use_container_width=True):
        with st.spinner("Тества... (1-2 минути)"):
            results = backtest_models(MODELS, MODELS.get('bo3'), MASTER, STATS, months=bt_months)

        btr = []
        for surf in ['Hard', 'Clay', 'Grass']:
            if surf in results:
                r   = results[surf]
                acc = r['accuracy']
                ico = "🟢" if acc >= 0.62 else ("🟡" if acc >= 0.58 else "🔴")
                lbl = "Добра" if acc >= 0.62 else ("Средна" if acc >= 0.58 else "Слаба")
                btr.append({"Настилка": surf,
                             "Точност": f"{acc:.1%}",
                             "Верни / Общо": f"{r['correct']} / {r['total']}",
                             "Оценка": f"{ico} {lbl}"})
        if btr:
            st.dataframe(pd.DataFrame(btr), hide_index=True, use_container_width=True)

        if 'BO3' in results:
            mae = results['BO3']['mae']
            ico = "🟢" if mae < 2.5 else ("🟡" if mae < 3.0 else "🔴")
            st.metric("BO3 MAE", f"{mae:.2f} геймове",
                      delta=f"{ico} {'Добър' if mae<2.5 else ('Среден' if mae<3.0 else 'Слаб')}")

        bad = [s for s in ['Hard','Clay','Grass']
               if s in results and results[s]['accuracy'] < 0.55]
        if bad:
            st.error(f"⚠️ Слаба точност при {', '.join(bad)} — препоръчително преобучаване!")

    st.divider()

    # ── Преобучаване ──────────────────────────────────
    st.subheader("🔄 Преобучаване")
    st.warning("Автоматично се запазва backup преди всяко преобучаване.")

    retrain_surfs = st.multiselect(
        "Кои модели?",
        ["Hard", "Clay", "Grass", "BO3"],
        default=["Hard", "Clay", "Grass", "BO3"]
    )

    if st.button("🚀 Преобучи", type="primary", use_container_width=True):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        bar = st.progress(0, text="Подготвя...")

        for i, surf in enumerate(retrain_surfs):
            bar.progress(int(i / len(retrain_surfs) * 100), text=f"Преобучава {surf}...")

            if surf == "BO3":
                src = f"{MODEL_DIR}/tennis_model_bo3_final.pkl"
                if os.path.exists(src):
                    joblib.dump(MODELS['bo3'], f"{BACKUP_DIR}/tennis_model_bo3_final.pkl")
                with st.spinner("BO3..."):
                    new_bo3, mae = retrain_bo3_model(MASTER)
                if new_bo3:
                    joblib.dump(new_bo3, src)
                    MODELS['bo3'] = new_bo3
                    st.success(f"✅ BO3 — MAE: {mae:.2f}" if mae else "✅ BO3 преобучен")
            else:
                src = f"{MODEL_DIR}/winner_model_{surf}.joblib"
                if os.path.exists(src) and MODELS.get(surf):
                    joblib.dump(MODELS[surf], f"{BACKUP_DIR}/winner_model_{surf}.joblib")
                with st.spinner(f"{surf}..."):
                    new_m = retrain_winner_model(MODELS[surf], surf, MASTER, STATS)
                if new_m:
                    joblib.dump(new_m, src)
                    MODELS[surf] = new_m
                    st.success(f"✅ {surf} преобучен")
                else:
                    st.warning(f"⚠️ {surf}: недостатъчно данни")

        bar.progress(100, text="Готово!")
        st.cache_resource.clear()
        st.balloons()
        st.success("🎉 Готово! Backup е в `models/backup/`")

    st.divider()

    # ── Rollback ──────────────────────────────────────
    st.subheader("⏪ Rollback")

    avail_backup = []
    for surf in ['Hard', 'Clay', 'Grass']:
        if os.path.exists(f"{BACKUP_DIR}/winner_model_{surf}.joblib"):
            avail_backup.append(surf)
    if os.path.exists(f"{BACKUP_DIR}/tennis_model_bo3_final.pkl"):
        avail_backup.append("BO3")

    if not avail_backup:
        st.info("Няма backup. Създава се автоматично при следващото преобучаване.")
    else:
        rollback_sel = st.multiselect("Върни:", avail_backup, default=avail_backup)
        if st.button("⏪ Rollback", type="secondary", use_container_width=True):
            import shutil
            for surf in rollback_sel:
                if surf == "BO3":
                    shutil.copy2(f"{BACKUP_DIR}/tennis_model_bo3_final.pkl",
                                 f"{MODEL_DIR}/tennis_model_bo3_final.pkl")
                else:
                    shutil.copy2(f"{BACKUP_DIR}/winner_model_{surf}.joblib",
                                 f"{MODEL_DIR}/winner_model_{surf}.joblib")
                st.success(f"✅ {surf} върнат")
            st.cache_resource.clear()
            time.sleep(1)
            st.rerun()

    with st.expander("ℹ️ Текущи модели"):
        for surf in ['Hard', 'Clay', 'Grass']:
            m = MODELS.get(surf)
            if m:
                st.markdown(
                    f"**{surf}:** RF | Trees `{m.n_estimators}` | "
                    f"Depth `{m.max_depth}` | Features: `{list(m.feature_names_in_)}`"
                )
        if MODELS.get('bo3'):
            m = MODELS['bo3']
            st.markdown(
                f"**BO3:** RF Regressor | Trees `{m.n_estimators}` | "
                f"Depth `{m.max_depth}` | Features: `['rank_diff', 'w_roll_hold', 'l_roll_hold']`"
            )import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import json
import time
from datetime import datetime, timedelta
from scipy import stats as scipy_stats
from streamlit_autorefresh import st_autorefresh

from data_utils import (
    load_master, build_player_stats, build_features,
    SURFACE_FEATURES, compute_h2h,
    backtest_models, retrain_winner_model, retrain_bo3_model
)

# ═══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🎾 Tennis Predictions",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stMetric { background: #1e1e2e; border-radius: 10px; padding: 10px; }
.block-container { padding-top: 1.5rem; }
div[data-testid="metric-container"] { background: #262640; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# ЗАРЕЖДАНЕ НА МОДЕЛИ
# ═══════════════════════════════════════════════════════════════
MODEL_DIR = "models"
BACKUP_DIR = "models/backup"

@st.cache_resource(show_spinner="🤖 Зарежда моделите...")
def load_models():
    m = {}
    for surf in ['Hard', 'Grass', 'Clay']:
        path = f"{MODEL_DIR}/winner_model_{surf}.joblib"
        try:
            m[surf] = joblib.load(path)
        except Exception as e:
            st.error(f"Грешка при зареждане на {surf}: {e}")
            m[surf] = None
    try:
        m['bo3'] = joblib.load(f"{MODEL_DIR}/tennis_model_bo3_final.pkl")
    except Exception as e:
        st.warning(f"BO3 модел не е наличен: {e}")
        m['bo3'] = None
    return m

MODELS = load_models()

# ═══════════════════════════════════════════════════════════════
# ЗАРЕЖДАНЕ НА ДАННИ
# ═══════════════════════════════════════════════════════════════
MASTER = load_master()
STATS  = build_player_stats(MASTER)
PLAYERS = sorted(STATS.index.dropna().tolist())

# ═══════════════════════════════════════════════════════════════
# ПОМОЩНИ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════
def predict_match(p1, p2, surface):
    feats = build_features(p1, p2, surface, STATS, MASTER)
    feat_list = SURFACE_FEATURES[surface]
    X = pd.DataFrame([[feats[f] for f in feat_list]], columns=feat_list)
    prob_p1 = float(MODELS[surface].predict_proba(X)[0][1])
    bo3_pred = None
    if MODELS['bo3'] is not None:
        X_bo3 = pd.DataFrame(
            [[feats['rank_diff'], feats['w_roll_hold'], feats['l_roll_hold']]],
            columns=['rank_diff', 'w_roll_hold', 'l_roll_hold']
        )
        bo3_pred = float(MODELS['bo3'].predict(X_bo3)[0])
    return prob_p1, bo3_pred, feats

def edge_badge(edge):
    if edge > 0.05:
        return "🟢", "Добра стойност!"
    elif edge > 0.02:
        return "🟡", "Слаба стойност"
    else:
        return "🔴", "Без стойност"

def implied_prob(odds):
    return 1.0 / odds if odds > 1 else 0.0

def kelly(prob, odds):
    b = odds - 1
    k = (prob * odds - 1) / b if b > 0 else 0
    return max(0.0, k)

# ═══════════════════════════════════════════════════════════════
# ТАБОВЕ
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Ръчна проверка",
    "📅 Дневна програма",
    "💰 Калкулатор",
    "🤖 Модели"
])


# ╔══════════════════════════════════════════════════════╗
# ║  ТАБ 1 — РЪЧНА ПРОВЕРКА                            ║
# ╚══════════════════════════════════════════════════════╝
with tab1:
    st.header("🔍 Ръчна проверка на мач")

    col1, col2, col3 = st.columns([3, 3, 2])
    with col1:
        p1 = st.selectbox("🎾 Играч 1", [""] + PLAYERS, key="p1")
    with col2:
        p2 = st.selectbox("🎾 Играч 2", [""] + PLAYERS, key="p2")
    with col3:
        surface = st.selectbox("🏟️ Настилка", ["Hard", "Clay", "Grass"])

    col4, col5 = st.columns(2)
    with col4:
        odds1 = st.number_input(f"💲 Коефициент — {p1 or 'Играч 1'}",
                                min_value=1.01, max_value=50.0, value=1.85, step=0.05, format="%.2f")
    with col5:
        odds2 = st.number_input(f"💲 Коефициент — {p2 or 'Играч 2'}",
                                min_value=1.01, max_value=50.0, value=2.10, step=0.05, format="%.2f")

    bankroll_t1 = st.number_input("💳 Банка (лв.)", min_value=1.0, value=1000.0, step=50.0,
                                   format="%.2f", key="bank_t1")

    can_run = bool(p1 and p2 and p1 != p2 and MODELS.get(surface))
    if st.button("🎯 Анализирай мача", type="primary", disabled=not can_run, use_container_width=True):
        with st.spinner("Изчислява..."):
            prob_p1, bo3_pred, feats = predict_match(p1, p2, surface)
        prob_p2 = 1 - prob_p1

        # Нормализирани имплицирани вероятности
        raw1, raw2 = implied_prob(odds1), implied_prob(odds2)
        margin = raw1 + raw2
        impl1 = raw1 / margin
        impl2 = raw2 / margin

        edge1 = prob_p1 - impl1
        edge2 = prob_p2 - impl2

        # ── Победител ────────────────────────────────────
        st.divider()
        st.subheader("📊 Предикция за победител")
        c1, c2 = st.columns(2)

        for col_w, player, prob_m, impl, edge, odds_bk in [
            (c1, p1, prob_p1, impl1, edge1, odds1),
            (c2, p2, prob_p2, impl2, edge2, odds2),
        ]:
            badge, txt = edge_badge(edge)
            with col_w:
                st.markdown(f"### 🎾 {player}")
                st.metric("Модел %",       f"{prob_m:.1%}")
                st.metric("Букмейкър %",   f"{impl:.1%}", delta=f"{edge:+.1%}")
                # Честен коефициент от модела
                fair_odds = 1/prob_m if prob_m > 0 else 99
                st.caption(f"Честен коефициент: **{fair_odds:.2f}** | Букмейкър: **{odds_bk:.2f}**")
                if edge > 0.05:
                    st.success(f"{badge} {txt} — ръб {edge:.1%}")
                elif edge > 0.02:
                    st.info(f"{badge} {txt} — ръб {edge:.1%}")
                else:
                    st.error(f"{badge} {txt} — ръб {edge:.1%}")

        # ── Over/Under геймове ───────────────────────────
        if bo3_pred is not None:
            st.divider()
            st.subheader("🎲 Геймове — Over / Under")
            gc1, gc2 = st.columns([1, 2])
            with gc1:
                st.metric("Очаквани геймове", f"{bo3_pred:.1f}")
                game_line = st.number_input("Линия (Over/Under)",
                                            min_value=14.0, max_value=42.0,
                                            value=21.5, step=0.5, format="%.1f", key="gl_t1")
            with gc2:
                std = 3.0
                p_over  = float(1 - scipy_stats.norm.cdf(game_line, loc=bo3_pred, scale=std))
                p_under = 1.0 - p_over
                st.metric(f"Over  {game_line}", f"{p_over:.1%}")
                st.metric(f"Under {game_line}", f"{p_under:.1%}")
                st.progress(int(p_over * 100), text="⬆ Over ─── Under ⬇")

        # ── Калкулатор на залог ───────────────────────────
        st.divider()
        st.subheader("💰 Препоръчан залог")

        best_edge_player = p1 if edge1 >= edge2 else p2
        best_edge        = max(edge1, edge2)
        best_prob        = prob_p1 if edge1 >= edge2 else prob_p2
        best_odds        = odds1  if edge1 >= edge2 else odds2

        kelly_f  = kelly(best_prob, best_odds)
        half_k   = kelly_f / 2

        method_t1 = st.radio("Метод", ["½ Kelly (препоръчан)", "Full Kelly",
                                        "¼ Kelly", "Фиксиран %"],
                              horizontal=True, key="method_t1")
        if method_t1 == "Full Kelly":
            frac = kelly_f
        elif method_t1 == "½ Kelly (препоръчан)":
            frac = half_k
        elif method_t1 == "¼ Kelly":
            frac = kelly_f / 4
        else:
            pct_t1 = st.slider("% от банката", 1, 15, 3, key="pct_t1")
            frac = pct_t1 / 100

        bet_amt = bankroll_t1 * frac
        profit  = bet_amt * (best_odds - 1)

        bc1, bc2, bc3 = st.columns(3)
        bc1.metric(f"Залог върху {best_edge_player}", f"{bet_amt:.2f} лв.",
                   f"{frac:.1%} от банката")
        bc2.metric("Потенциална печалба", f"+{profit:.2f} лв.")
        bc3.metric("Edge", f"{best_edge:+.1%}")

        if kelly_f <= 0:
            st.error("❌ Отрицателна стойност — **не залагай**!")

        # ── Детайли ──────────────────────────────────────
        with st.expander("📈 Входни данни за модела"):
            s1 = STATS.loc[p1] if p1 in STATS.index else pd.Series(dtype=float)
            s2 = STATS.loc[p2] if p2 in STATS.index else pd.Series(dtype=float)
            detail = {
                "Метрика": ["Ранглиста", f"ELO ({surface})", "1st Serve %",
                            "Форма (last 10)", "Почивка (дни)", "Възраст"],
                p1: [int(s1.get('rank',999) or 999),
                     f"{float(s1.get(f'elo_{surface}',1500) or 1500):.0f}",
                     f"{float(s1.get('roll_first',0.65) or 0.65):.1%}",
                     f"{float(s1.get('roll_form',0.5) or 0.5):.1%}",
                     int(s1.get('rest_days',0) or 0),
                     f"{s1.get('age','?')}"],
                p2: [int(s2.get('rank',999) or 999),
                     f"{float(s2.get(f'elo_{surface}',1500) or 1500):.0f}",
                     f"{float(s2.get('roll_first',0.65) or 0.65):.1%}",
                     f"{float(s2.get('roll_form',0.5) or 0.5):.1%}",
                     int(s2.get('rest_days',0) or 0),
                     f"{s2.get('age','?')}"],
            }
            st.dataframe(pd.DataFrame(detail), hide_index=True, use_container_width=True)


# ╔══════════════════════════════════════════════════════╗
# ║  ТАБ 2 — ДНЕВНА ПРОГРАМА                           ║
# ╚══════════════════════════════════════════════════════╝
with tab2:
    # Авто-обновяване на всеки 6 часа
    st_autorefresh(interval=6 * 60 * 60 * 1000, key="sched_refresh")

    st.header("📅 Дневна програма (ATP)")

    ODDS_KEY = st.secrets.get("ODDS_API_KEY", "")
    if not ODDS_KEY:
        st.error("❌ Липсва ODDS_API_KEY в Streamlit Secrets.")
        st.code('ODDS_API_KEY = "твоя_ключ"', language="toml")
    else:
        col_r, col_s = st.columns([1, 2])
        with col_r:
            manual_refresh = st.button("🔄 Обнови сега", use_container_width=True)
        with col_s:
            sched_surf = st.selectbox("Настилка по подразбиране",
                                      ["Hard", "Clay", "Grass"], key="sched_surf")

        if manual_refresh:
            st.cache_data.clear()
            st.rerun()

        @st.cache_data(ttl=21600, show_spinner="📡 Зарежда програмата от TheOddsAPI...")
        def fetch_odds(key):
            url = "https://api.the-odds-api.com/v4/sports/tennis_atp/odds"
            r = requests.get(url, params={
                "apiKey": key, "regions": "eu",
                "markets": "h2h", "oddsFormat": "decimal"
            }, timeout=15)
            return r.json()

        try:
            odds_data = fetch_odds(ODDS_KEY)
        except Exception as e:
            st.error(f"Грешка: {e}")
            odds_data = []

        if isinstance(odds_data, dict) and "message" in odds_data:
            st.error(f"API грешка: {odds_data['message']}")
        elif not odds_data:
            st.warning("Няма намерени мачове.")
        else:
            st.success(f"✅ {len(odds_data)} мача  |  Обновено: {datetime.now().strftime('%H:%M:%S')}")

            def find_player(name):
                last = name.split()[-1].lower()
                hits = [p for p in PLAYERS if last in p.lower()]
                return hits[0] if hits else None

            for match in odds_data:
                home = match.get("home_team", "")
                away = match.get("away_team", "")
                start = match.get("commence_time","")[:16].replace("T"," ")

                best_h, best_a = 1.0, 1.0
                for bk in match.get("bookmakers", []):
                    for mkt in bk.get("markets", []):
                        if mkt["key"] == "h2h":
                            for o in mkt["outcomes"]:
                                if o["name"] == home: best_h = max(best_h, o["price"])
                                elif o["name"] == away: best_a = max(best_a, o["price"])

                with st.expander(f"🎾  {home}  vs  {away}  |  {start} UTC"):
                    cm1, cm2 = st.columns(2)
                    cm1.metric(home, f"@ {best_h:.2f}")
                    cm2.metric(away, f"@ {best_a:.2f}")

                    p1f = find_player(home)
                    p2f = find_player(away)

                    if p1f and p2f:
                        surf_k = f"surf_{home}_{away}"
                        sc = st.selectbox("Настилка", ["Hard","Clay","Grass"],
                                          key=surf_k,
                                          index=["Hard","Clay","Grass"].index(sched_surf))
                        if MODELS.get(sc):
                            prob, bo3p, _ = predict_match(p1f, p2f, sc)
                            impl_h = (1/best_h) / ((1/best_h)+(1/best_a))
                            edge_h = prob - impl_h
                            badge, txt = edge_badge(edge_h)
                            cr1, cr2, cr3 = st.columns(3)
                            cr1.metric(f"Модел {p1f.split()[-1]}", f"{prob:.1%}")
                            cr2.metric("Edge", f"{edge_h:+.1%}")
                            cr3.metric("Честен коеф.", f"{1/prob:.2f}" if prob>0 else "N/A")
                            if bo3p:
                                st.caption(f"📊 Очаквани геймове: **{bo3p:.1f}**")
                            st.caption(f"{badge} {txt}  |  Разпознати: *{p1f}* vs *{p2f}*")
                    else:
                        missing = [n for n,f in [(home,p1f),(away,p2f)] if not f]
                        st.caption(f"⚠️ Не е намерен в базата: {', '.join(missing)}")


# ╔══════════════════════════════════════════════════════╗
# ║  ТАБ 3 — КАЛКУЛАТОР                                ║
# ╚══════════════════════════════════════════════════════╝
with tab3:
    st.header("💰 Калкулатор на залог")

    bankroll = st.number_input("💳 Банка (лв.)",
                               min_value=1.0, value=1000.0, step=100.0,
                               format="%.2f", key="bank_t3")
    st.divider()

    ca, cb = st.columns(2)
    with ca:
        model_prob_t3 = st.slider("📈 Вероятност от модела (%)", 1, 99, 60, key="mp_t3") / 100
    with cb:
        bet_odds_t3 = st.number_input("💲 Коефициент на букмейкъра",
                                      min_value=1.01, max_value=50.0,
                                      value=2.10, step=0.05, format="%.2f", key="bo_t3")

    method_t3 = st.radio("📐 Метод",
                         ["½ Kelly (препоръчан)", "Full Kelly", "¼ Kelly", "Фиксиран %", "Фиксирана сума"],
                         horizontal=True, key="meth_t3")

    impl_t3  = implied_prob(bet_odds_t3)
    edge_t3  = model_prob_t3 - impl_t3
    kelly_t3 = kelly(model_prob_t3, bet_odds_t3)

    st.divider()

    if kelly_t3 <= 0 and method_t3 in ["½ Kelly (препоръчан)", "Full Kelly", "¼ Kelly"]:
        st.error("❌ Отрицателна стойност — **не залагай на този мач!**")
        st.metric("Edge", f"{edge_t3:.1%}")
    else:
        if method_t3 == "Full Kelly":
            frac_t3, lbl = kelly_t3, "Full Kelly"
        elif method_t3 == "½ Kelly (препоръчан)":
            frac_t3, lbl = kelly_t3/2, "½ Kelly"
        elif method_t3 == "¼ Kelly":
            frac_t3, lbl = kelly_t3/4, "¼ Kelly"
        elif method_t3 == "Фиксиран %":
            p_t3 = st.slider("% от банката", 1, 20, 3, key="pct_t3")
            frac_t3, lbl = p_t3/100, f"{p_t3}%"
        else:
            fx = st.number_input("Сума (лв.)", min_value=1.0, value=50.0, step=10.0, key="fx_t3")
            frac_t3, lbl = fx/bankroll, f"{fx:.0f} лв."

        bet_t3    = bankroll * frac_t3
        profit_t3 = bet_t3 * (bet_odds_t3 - 1)
        ev_t3     = edge_t3 * bet_t3 * bet_odds_t3

        cr1, cr2, cr3 = st.columns(3)
        cr1.metric(f"💰 Залог ({lbl})", f"{bet_t3:.2f} лв.", f"{frac_t3:.1%} от банката")
        cr2.metric("📈 Потенциална печалба", f"+{profit_t3:.2f} лв.")
        cr3.metric("🎯 EV", f"{ev_t3:.2f} лв.")

        st.divider()
        d1, d2, d3 = st.columns(3)
        d1.metric("Edge", f"{edge_t3:+.1%}")
        d2.metric("Модел %",      f"{model_prob_t3:.1%}")
        d3.metric("Букмейкър %",  f"{impl_t3:.1%}")

        badge_t3, txt_t3 = edge_badge(edge_t3)
        msg = f"{badge_t3} {txt_t3} — ръб {edge_t3:.1%}"
        if edge_t3 > 0.05: st.success(msg)
        elif edge_t3 > 0.02: st.info(msg)
        else: st.warning(msg)

        with st.expander("📊 Сравнение Kelly варианти"):
            comp = {
                "Метод": ["¼ Kelly", "½ Kelly", "Full Kelly"],
                "Залог (лв.)": [f"{bankroll*kelly_t3/4:.2f}", f"{bankroll*kelly_t3/2:.2f}", f"{bankroll*kelly_t3:.2f}"],
                "% от банката": [f"{kelly_t3/4:.1%}", f"{kelly_t3/2:.1%}", f"{kelly_t3:.1%}"],
                "Потенциална печалба": [
                    f"{bankroll*kelly_t3/4*(bet_odds_t3-1):.2f} лв.",
                    f"{bankroll*kelly_t3/2*(bet_odds_t3-1):.2f} лв.",
                    f"{bankroll*kelly_t3*(bet_odds_t3-1):.2f} лв."
                ],
            }
            st.dataframe(pd.DataFrame(comp), hide_index=True, use_container_width=True)


# ╔══════════════════════════════════════════════════════╗
# ║  ТАБ 4 — МОДЕЛИ (тест + преобучаване)              ║
# ╚══════════════════════════════════════════════════════╝
with tab4:
    st.header("🤖 Управление на моделите")

    st.info("""
    **Как работи:**
    - **Backtest** → тества точността на текущите модели на последните N месеца реални мачове
    - **Преобучаване** → тренира моделите с новите данни от базата на Джеф
    - **Rollback** → връща се към предишната запазена версия при лоши резултати
    """)

    # ── Backtest ──────────────────────────────────────
    st.subheader("📊 Backtest на моделите")
    bt_months = st.slider("Период (месеци назад)", 1, 12, 6, key="bt_months")

    if st.button("▶️ Стартирай Backtest", use_container_width=True):
        with st.spinner("Тества моделите... (може да отнеме 1-2 минути)"):
            results = backtest_models(MODELS, MODELS.get('bo3'), MASTER, STATS, months=bt_months)

        st.subheader("Резултати")

        btr = []
        for surf in ['Hard', 'Clay', 'Grass']:
            if surf in results:
                r = results[surf]
                acc = r['accuracy']
                color = "🟢" if acc >= 0.62 else ("🟡" if acc >= 0.58 else "🔴")
                btr.append({
                    "Настилка": surf,
                    "Точност": f"{acc:.1%}",
                    "Верни / Общо": f"{r['correct']} / {r['total']}",
                    "Оценка": f"{color} {'Добра' if acc>=0.62 else ('Средна' if acc>=0.58 else 'Слаба')}"
                })

        if btr:
            st.dataframe(pd.DataFrame(btr), hide_index=True, use_container_width=True)

        if 'BO3' in results:
            r = results['BO3']
            mae = r['mae']
            color = "🟢" if mae < 2.5 else ("🟡" if mae < 3.0 else "🔴")
            st.metric("BO3 MAE (геймове)",
                      f"{mae:.2f}",
                      delta=f"{color} {'Добър' if mae<2.5 else ('Среден' if mae<3.0 else 'Слаб')} резултат")

        # Предупреждение при лоши резултати
        bad_surfs = [s for s in ['Hard','Clay','Grass']
                     if s in results and results[s]['accuracy'] < 0.55]
        if bad_surfs:
            st.error(f"⚠️ Слаба точност при: **{', '.join(bad_surfs)}** — помисли за rollback или преобучаване!")

    st.divider()

    # ── Преобучаване ────────────────────────────────
    st.subheader("🔄 Преобучаване на моделите")
    st.warning("⚠️ Преди да преобучаваш, автоматично се запазва backup на текущите модели.")

    retrain_surfs = st.multiselect(
        "Кои модели да преобучим?",
        ["Hard", "Clay", "Grass", "BO3"],
        default=["Hard", "Clay", "Grass", "BO3"]
    )

    if st.button("🚀 Преобучи избраните модели", type="primary", use_container_width=True):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        progress_bar = st.progress(0, text="Подготвя...")

        for i, surf in enumerate(retrain_surfs):
            pct = int((i / len(retrain_surfs)) * 100)
            progress_bar.progress(pct, text=f"Преобучава {surf}...")

            if surf == "BO3":
                # Запазва backup
                src = f"{MODEL_DIR}/tennis_model_bo3_final.pkl"
                if os.path.exists(src):
                    joblib.dump(MODELS['bo3'], f"{BACKUP_DIR}/tennis_model_bo3_final.pkl")

                with st.spinner("Преобучава BO3..."):
                    new_bo3, mae = retrain_bo3_model(MASTER)
                if new_bo3 is not None:
                    joblib.dump(new_bo3, src)
                    MODELS['bo3'] = new_bo3
                    st.success(f"✅ BO3 преобучен — MAE: {mae:.2f}" if mae else "✅ BO3 преобучен")
                else:
                    st.error("❌ BO3 преобучаването неуспешно.")
            else:
                # Запазва backup
                src = f"{MODEL_DIR}/winner_model_{surf}.joblib"
                if os.path.exists(src) and MODELS.get(surf):
                    joblib.dump(MODELS[surf], f"{BACKUP_DIR}/winner_model_{surf}.joblib")

                with st.spinner(f"Преобучава {surf}..."):
                    new_m = retrain_winner_model(MODELS[surf], surf, MASTER, STATS)
                if new_m is not None:
                    joblib.dump(new_m, src)
                    MODELS[surf] = new_m
                    st.success(f"✅ {surf} модел преобучен успешно")
                else:
                    st.warning(f"⚠️ {surf}: недостатъчно данни за преобучаване")

        progress_bar.progress(100, text="Готово!")
        st.cache_resource.clear()
        st.balloons()
        st.success("🎉 Всички избрани модели са преобучени! Backup е запазен в `models/backup/`")

    st.divider()

    # ── Rollback ─────────────────────────────────────
    st.subheader("⏪ Rollback към предишна версия")

    backup_hard = os.path.exists(f"{BACKUP_DIR}/winner_model_Hard.joblib")
    backup_clay = os.path.exists(f"{BACKUP_DIR}/winner_model_Clay.joblib")
    backup_grass = os.path.exists(f"{BACKUP_DIR}/winner_model_Grass.joblib")
    backup_bo3  = os.path.exists(f"{BACKUP_DIR}/tennis_model_bo3_final.pkl")

    has_backup = any([backup_hard, backup_clay, backup_grass, backup_bo3])

    if not has_backup:
        st.info("Няма запазен backup. Backup се създава автоматично при следващото преобучаване.")
    else:
        available = []
        if backup_hard:  available.append("Hard")
        if backup_clay:  available.append("Clay")
        if backup_grass: available.append("Grass")
        if backup_bo3:   available.append("BO3")

        rollback_sel = st.multiselect("Върни към backup:", available, default=available)

        if st.button("⏪ Rollback", type="secondary", use_container_width=True):
            for surf in rollback_sel:
                if surf == "BO3":
                    src = f"{BACKUP_DIR}/tennis_model_bo3_final.pkl"
                    dst = f"{MODEL_DIR}/tennis_model_bo3_final.pkl"
                else:
                    src = f"{BACKUP_DIR}/winner_model_{surf}.joblib"
                    dst = f"{MODEL_DIR}/winner_model_{surf}.joblib"

                if os.path.exists(src):
                    import shutil
                    shutil.copy2(src, dst)
                    st.success(f"✅ {surf} върнат към backup версия")

            st.cache_resource.clear()
            st.info("🔄 Моделите са върнати. Страницата ще се презареди...")
            time.sleep(1)
            st.rerun()

    st.divider()

    # ── Информация за текущите модели ─────────────────
    with st.expander("ℹ️ Информация за текущите модели"):
        for surf in ['Hard', 'Clay', 'Grass']:
            m = MODELS.get(surf)
            if m:
                st.markdown(f"**{surf}:** `RandomForestClassifier` | "
                            f"Trees: `{m.n_estimators}` | "
                            f"Max depth: `{m.max_depth}` | "
                            f"Features: `{list(m.feature_names_in_)}`")
        if MODELS.get('bo3'):
            m = MODELS['bo3']
            st.markdown(f"**BO3:** `RandomForestRegressor` | "
                        f"Trees: `{m.n_estimators}` | "
                        f"Max depth: `{m.max_depth}` | "
                        f"Features: `['rank_diff', 'w_roll_hold', 'l_roll_hold']`")

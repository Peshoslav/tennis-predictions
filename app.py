"""
app.py  —  Tennis Predictions
Таб 1: Ръчна проверка + залог калкулатор
Таб 2: Дневна програма (TheOddsAPI, auto-refresh 6h)
Таб 3: Backtest + Преобучаване + Rollback
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, time
import requests
from datetime import datetime, timedelta
from scipy import stats as scipy_stats

from data_utils  import load_master, build_player_stats, build_features, SURFACE_FEATURES
from retrain_utils import (
    retrain_all, apply_new_models, rollback_models, run_backtest,
    load_meta, save_meta
)

# ══════════════════════════════════════════════════════════════
#  КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🎾 Tennis Predictions",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
  .block-container{padding-top:1.5rem}
  [data-testid="stMetric"]{background:#1e1e2e;border-radius:10px;padding:8px}
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  ЗАРЕЖДАНЕ НА МОДЕЛИ
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🤖 Зарежда моделите...")
def load_models():
    m = {}
    for surf in ['Hard','Grass','Clay']:
        path = f"models/winner_model_{surf}.joblib"
        if os.path.exists(path):
            m[surf] = joblib.load(path)
    for bo3_path in ["models/tennis_model_bo3_final.pkl"]:
        if os.path.exists(bo3_path):
            try:   m['bo3'] = joblib.load(bo3_path)
            except: pass
    return m

MODELS  = load_models()
MASTER  = load_master()
STATS   = build_player_stats(MASTER)
PLAYERS = sorted(STATS.index.dropna().tolist())

ODDS_KEY = st.secrets.get("ODDS_API_KEY","")


# ──────────────────────────────────────────────────────────────
#  ПОМОЩНИ ФУНКЦИИ
# ──────────────────────────────────────────────────────────────
def predict_match(p1, p2, surface):
    feats     = build_features(p1, p2, surface, STATS, MASTER)
    feat_list = SURFACE_FEATURES[surface]
    X         = pd.DataFrame([[feats[f] for f in feat_list]], columns=feat_list)
    prob_p1   = float(MODELS[surface].predict_proba(X)[0][1])

    bo3_pred = None
    if MODELS.get('bo3'):
        Xb = pd.DataFrame([[feats['rank_diff'],feats['w_roll_hold'],feats['l_roll_hold']]],
                          columns=['rank_diff','w_roll_hold','l_roll_hold'])
        bo3_pred = float(MODELS['bo3'].predict(Xb)[0])
    return prob_p1, bo3_pred, feats


def kelly(prob, odds):
    k = (prob * odds - 1) / (odds - 1)
    return max(k, 0.0)

def edge_badge(e):
    if e > 0.05:  return "🟢"
    if e > 0.02:  return "🟡"
    return "🔴"


# ══════════════════════════════════════════════════════════════
#  ТАБОВЕ
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🔍 Ръчна проверка",
    "📅 Дневна програма",
    "🧪 Модели & Тестове",
])


# ╔══════════════════════════════════════════════════════════╗
# ║  ТАБ 1 — РЪЧНА ПРОВЕРКА + ЗАЛОГ КАЛКУЛАТОР             ║
# ╚══════════════════════════════════════════════════════════╝
with tab1:
    st.header("🔍 Ръчна проверка")

    # ── Избор на играчи ───────────────────────────────────────
    c1, c2, c3 = st.columns([3,3,2])
    with c1:
        p1 = st.selectbox("👤 Играч 1", [""]+PLAYERS, key="t1p1")
    with c2:
        p2 = st.selectbox("👤 Играч 2", [""]+PLAYERS, key="t1p2")
    with c3:
        surface = st.selectbox("🏟 Настилка", ["Hard","Clay","Grass"])

    # ── Коефициенти ───────────────────────────────────────────
    l1 = p1 or "Играч 1"
    l2 = p2 or "Играч 2"
    ca, cb = st.columns(2)
    with ca:
        odds1 = st.number_input(f"Коеф. {l1}", 1.01, 50.0, 1.85, 0.05, "%.2f")
    with cb:
        odds2 = st.number_input(f"Коеф. {l2}", 1.01, 50.0, 2.10, 0.05, "%.2f")

    # ── Банка ─────────────────────────────────────────────────
    bankroll = st.number_input("💳 Банка (лв.)", 1.0, 1_000_000.0, 1000.0, 100.0, "%.0f")

    ok = bool(p1 and p2 and p1 != p2)
    if st.button("🎯 Анализирай", type="primary", disabled=not ok, use_container_width=True):

        with st.spinner("Изчислява..."):
            prob_p1, bo3_pred, feats = predict_match(p1, p2, surface)

        prob_p2  = 1 - prob_p1
        margin   = 1/odds1 + 1/odds2
        impl1    = (1/odds1) / margin
        impl2    = (1/odds2) / margin
        edge1    = prob_p1 - impl1
        edge2    = prob_p2 - impl2

        # ─── Победител ────────────────────────────────────────
        st.divider()
        st.subheader("📊 Предикция — Победител")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 🎾 {p1}")
            st.metric("Модел",       f"{prob_p1:.1%}")
            st.metric("Букмейкър",   f"{impl1:.1%}", f"{edge1:+.1%}")
            if edge1 > 0.05:
                st.success(f"🟢 **Добра стойност** — ръб {edge1:.1%}")
            elif edge1 > 0.02:
                st.info(f"🟡 Малка стойност ({edge1:.1%})")
            else:
                st.error(f"🔴 Без стойност ({edge1:.1%})")

        with col2:
            st.markdown(f"#### 🎾 {p2}")
            st.metric("Модел",       f"{prob_p2:.1%}")
            st.metric("Букмейкър",   f"{impl2:.1%}", f"{edge2:+.1%}")
            if edge2 > 0.05:
                st.success(f"🟢 **Добра стойност** — ръб {edge2:.1%}")
            elif edge2 > 0.02:
                st.info(f"🟡 Малка стойност ({edge2:.1%})")
            else:
                st.error(f"🔴 Без стойност ({edge2:.1%})")

        # ─── Over/Under Геймове ───────────────────────────────
        if bo3_pred:
            st.divider()
            st.subheader("🎲 Предикция — Брой геймове")
            cg1, cg2 = st.columns([1,2])
            with cg1:
                st.metric("Очаквани геймове", f"{bo3_pred:.1f}")
                game_line = st.number_input("Линия O/U", 14.0, 40.0, 22.5, 0.5, "%.1f", key="t1gl")
            with cg2:
                std     = 3.0
                p_over  = float(1 - scipy_stats.norm.cdf(game_line, bo3_pred, std))
                p_under = 1 - p_over
                oc1, oc2 = st.columns(2)
                oc1.metric(f"Over  {game_line}", f"{p_over:.1%}")
                oc2.metric(f"Under {game_line}", f"{p_under:.1%}")
                st.progress(int(p_over * 100), text="Over ←──────── Under")

        # ─── Залог Калкулатор ─────────────────────────────────
        st.divider()
        st.subheader("💰 Препоръка за залог")

        best_prob  = max(prob_p1, prob_p2)
        best_odds  = odds1 if prob_p1 >= prob_p2 else odds2
        best_edge  = max(edge1, edge2)
        best_label = (p1 if prob_p1 >= prob_p2 else p2)

        k_full  = kelly(best_prob, best_odds)
        k_half  = k_full / 2
        k_qrtr  = k_full / 4

        method = st.radio(
            "Метод",
            ["½ Kelly (препоръчан)","Full Kelly","¼ Kelly","Фиксиран %"],
            horizontal=True, key="t1method"
        )

        if method == "Full Kelly":         frac = k_full
        elif method == "½ Kelly (препоръчан)": frac = k_half
        elif method == "¼ Kelly":          frac = k_qrtr
        else:
            pct = st.slider("% от банката", 1, 20, 3, key="t1pct")
            frac = pct / 100

        bet_amt = bankroll * frac
        profit  = bet_amt * (best_odds - 1)
        ev      = best_edge * bet_amt * best_odds

        km1, km2, km3, km4 = st.columns(4)
        km1.metric("Препоръчан залог",   f"{bet_amt:.2f} лв.", f"{frac:.1%} от банката")
        km2.metric(f"Залагай на {best_label}", f"@ {best_odds:.2f}")
        km3.metric("Потенциална печалба",f"+{profit:.2f} лв.")
        km4.metric("EV (очаквана стойност)", f"{ev:.2f} лв.")

        if best_edge <= 0:
            st.error("❌ Отрицателна стойност — не залагай!")
        elif best_edge > 0.05:
            st.success(f"🟢 Ръб {best_edge:.1%} — **залагай {method}**")
        else:
            st.warning(f"🟡 Малка стойност ({best_edge:.1%}) — залагай консервативно")

        # ─── Детайли ──────────────────────────────────────────
        with st.expander("📈 Входни данни"):
            s1 = STATS.loc[p1] if p1 in STATS.index else pd.Series(dtype=float)
            s2 = STATS.loc[p2] if p2 in STATS.index else pd.Series(dtype=float)
            detail = pd.DataFrame({
                "Метрика": ["Ранглиста",f"ELO ({surface})","1st Serve %","Форма last-10","Почивка (дни)","Възраст"],
                p1: [int(s1.get('rank',999) or 999), f"{s1.get(f'elo_{surface}',1500):.0f}",
                     f"{s1.get('roll_first',0.65):.1%}", f"{s1.get('roll_form',0.5):.1%}",
                     int(s1.get('rest_days',0) or 0), str(round(float(s1.get('age',0) or 0),1))],
                p2: [int(s2.get('rank',999) or 999), f"{s2.get(f'elo_{surface}',1500):.0f}",
                     f"{s2.get('roll_first',0.65):.1%}", f"{s2.get('roll_form',0.5):.1%}",
                     int(s2.get('rest_days',0) or 0), str(round(float(s2.get('age',0) or 0),1))],
            })
            st.dataframe(detail, hide_index=True, use_container_width=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  ТАБ 2 — ДНЕВНА ПРОГРАМА (auto-refresh 6h)             ║
# ╚══════════════════════════════════════════════════════════╝
with tab2:
    st.header("📅 Дневна програма — ATP")

    # ── Auto-refresh logic ────────────────────────────────────
    if "last_odds_fetch" not in st.session_state:
        st.session_state.last_odds_fetch = 0.0

    now_ts       = time.time()
    SIX_HOURS    = 6 * 3600
    age_secs     = now_ts - st.session_state.last_odds_fetch
    next_refresh = SIX_HOURS - age_secs

    col_info, col_btn, col_surf = st.columns([2,1,2])
    with col_info:
        if st.session_state.last_odds_fetch > 0:
            next_str = str(timedelta(seconds=int(max(next_refresh, 0))))
            st.caption(f"⏱ Следващо авт. обновяване след **{next_str}**")
        else:
            st.caption("⏱ Все още не е заредено")
    with col_btn:
        manual_refresh = st.button("🔄 Обнови сега", use_container_width=True)
    with col_surf:
        sched_surf = st.selectbox("Настилка по подразбиране",
                                  ["Hard","Clay","Grass"], key="t2surf")

    need_reload = manual_refresh or (age_secs >= SIX_HOURS)

    if not ODDS_KEY:
        st.error("❌ Липсва ODDS_API_KEY в Streamlit Secrets.")
        st.code('ODDS_API_KEY = "afee5fd3fc3f673d0534dff3f3367364"', language="toml")
        st.info("Streamlit Cloud → Settings → Secrets → постави горното")
    else:
        @st.cache_data(ttl=SIX_HOURS, show_spinner="📡 Зарежда мачове...")
        def _fetch_odds(key, _bust):
            r = requests.get(
                "https://api.the-odds-api.com/v4/sports/tennis_atp/odds",
                params={"apiKey": key, "regions":"eu",
                        "markets":"h2h", "oddsFormat":"decimal"},
                timeout=12
            )
            return r.json()

        bust = int(now_ts // SIX_HOURS) if not need_reload else int(now_ts)
        if need_reload:
            st.session_state.last_odds_fetch = now_ts

        try:
            odds_data = _fetch_odds(ODDS_KEY, bust)
        except Exception as e:
            st.error(f"Грешка: {e}"); odds_data = []

        if isinstance(odds_data, dict) and "message" in odds_data:
            st.error(f"API грешка: {odds_data['message']}")
        elif not odds_data:
            st.warning("Няма мачове в момента.")
        else:
            st.success(f"✅ {len(odds_data)} мача")

            def find_player(name):
                last = name.split()[-1].lower()
                hits = [p for p in PLAYERS if last in p.lower()]
                return hits[0] if hits else None

            for match in odds_data:
                home       = match.get("home_team","")
                away       = match.get("away_team","")
                start_time = match.get("commence_time","")[:16].replace("T"," ")

                best_h = best_a = 1.0
                for bk in match.get("bookmakers",[]):
                    for mkt in bk.get("markets",[]):
                        if mkt["key"] == "h2h":
                            for o in mkt["outcomes"]:
                                if o["name"] == home: best_h = max(best_h, o["price"])
                                elif o["name"] == away: best_a = max(best_a, o["price"])

                with st.expander(f"🎾  {home}  vs  {away}   |   {start_time} UTC"):
                    mc1, mc2 = st.columns(2)
                    mc1.metric(home, f"@ {best_h:.2f}")
                    mc2.metric(away, f"@ {best_a:.2f}")

                    p1f = find_player(home)
                    p2f = find_player(away)

                    if p1f and p2f:
                        surf_k = f"surf_{home}_{away}"
                        surf_c = st.selectbox("Настилка", ["Hard","Clay","Grass"],
                                              key=surf_k,
                                              index=["Hard","Clay","Grass"].index(sched_surf))
                        with st.spinner("Предсказва..."):
                            prob, bo3p, _ = predict_match(p1f, p2f, surf_c)
                        impl_h = (1/best_h) / (1/best_h + 1/best_a)
                        edge_h = prob - impl_h
                        mc3, mc4 = st.columns(2)
                        mc3.metric(f"Модел — {p1f}", f"{prob:.1%}")
                        mc4.metric("Ръб", f"{edge_h:+.1%}")
                        if bo3p:
                            st.caption(f"📊 Очаквани геймове: **{bo3p:.1f}**")
                        st.caption(f"Разпознати: *{p1f}* vs *{p2f}*")
                    else:
                        miss = [n for n,f in [(home,p1f),(away,p2f)] if not f]
                        st.caption(f"⚠️ Не са намерени: {', '.join(miss)}")

        # Авт. презареждане след 6 часа
        if age_secs < SIX_HOURS:
            time.sleep(0)   # non-blocking placeholder
        else:
            st.rerun()


# ╔══════════════════════════════════════════════════════════╗
# ║  ТАБ 3 — МОДЕЛИ: ТЕСТОВЕ, ПРЕОБУЧАВАНЕ, ROLLBACK       ║
# ╚══════════════════════════════════════════════════════════╝
with tab3:
    st.header("🧪 Управление на моделите")

    meta = load_meta()
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Версия на модела", f"v{meta.get('version','?')}")
    mc2.metric("Трениран на",      meta.get('trained_on','original'))
    mc3.metric("Дата",             meta.get('created_at','—')[:10])

    st.divider()

    # ════════════════════════════════════════
    #  БЕКТЕСТ
    # ════════════════════════════════════════
    st.subheader("📈 Бектест на текущите модели")
    bt_months = st.slider("Период за тест (месеци)", 1, 12, 3, key="bt_months")

    if st.button("▶️ Стартирай бектест", use_container_width=True):
        with st.spinner("Изчислява точност върху исторически данни..."):
            bt_df = run_backtest(MASTER, MODELS, months=bt_months)

        if bt_df.empty:
            st.warning("Няма достатъчно данни за избрания период.")
        else:
            total   = len(bt_df)
            correct = bt_df['correct'].sum()
            acc     = correct / total

            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Общо мача",      total)
            bc2.metric("Верни предикции", int(correct))
            bc3.metric("Точност",         f"{acc:.1%}",
                       delta="✅ Добро" if acc >= 0.60 else "⚠️ Под 60%",
                       delta_color="normal" if acc >= 0.60 else "inverse")

            st.subheader("Точност по настилка")
            surf_acc = (bt_df.groupby('surface')['correct']
                         .agg(['sum','count'])
                         .rename(columns={'sum':'Верни','count':'Мача'}))
            surf_acc['Точност'] = (surf_acc['Верни'] / surf_acc['Мача']).map("{:.1%}".format)
            st.dataframe(surf_acc, use_container_width=True)

            st.subheader("Точност по месец")
            monthly = (bt_df.groupby('month')['correct']
                        .agg(['mean','count'])
                        .rename(columns={'mean':'Точност','count':'Мача'}))
            monthly['Точност'] = monthly['Точност'].map("{:.1%}".format)
            st.dataframe(monthly, use_container_width=True)

            if acc < 0.55:
                st.error("❗ Точността е под 55% — препоръчително е преобучаване или rollback!")
            elif acc < 0.60:
                st.warning("⚠️ Точността е между 55-60% — обмисли преобучаване.")
            else:
                st.success("✅ Моделите работят добре!")

    # ════════════════════════════════════════
    #  ПРЕОБУЧАВАНЕ
    # ════════════════════════════════════════
    st.divider()
    st.subheader("🔄 Преобучаване с нови данни")

    st.info("""
    Преобучаването:
    - Тренира с всички данни до **4 седмици** назад
    - Тества на последните 4 седмици
    - Показва сравнение стар ↔ нов модел
    - Ти решаваш дали да приемеш новия
    """)

    if st.button("🚀 Преобучи моделите сега", use_container_width=True, key="retrain_btn"):
        prog = st.progress(0, "Изгражда тренировъчни данни...")
        with st.spinner("Може да отнеме 2-3 минути..."):
            try:
                prog.progress(20, "Тренира...")
                new_models, metrics_new, metrics_old = retrain_all(MASTER)
                prog.progress(90, "Оценява...")
                st.session_state['_new_models']   = new_models
                st.session_state['_metrics_new']  = metrics_new
                st.session_state['_metrics_old']  = metrics_old
                prog.progress(100, "Готово!")
            except Exception as e:
                st.error(f"Грешка при преобучаване: {e}")

    if '_new_models' in st.session_state:
        metrics_new = st.session_state['_metrics_new']
        metrics_old = st.session_state['_metrics_old']

        st.subheader("Сравнение: Стар ↔ Нов модел")
        rows_cmp = []
        for key in metrics_new:
            r = {"Модел": key}
            for k, v in metrics_new[key].items():
                r[f"Нов ({k})"] = v
            if key in metrics_old:
                for k, v in metrics_old[key].items():
                    r[f"Стар ({k})"] = v
                    # Сравнение
                    is_better = (v > metrics_new[key].get(k,0) if k == 'mae'
                                 else metrics_new[key].get(k,0) >= v)
                    r["Подобрение"] = "✅ Да" if is_better else "❌ Не"
            rows_cmp.append(r)

        st.dataframe(pd.DataFrame(rows_cmp), hide_index=True, use_container_width=True)

        col_apply, col_discard = st.columns(2)
        with col_apply:
            if st.button("✅ Приеми новите модели", type="primary", use_container_width=True):
                apply_new_models(st.session_state['_new_models'])
                st.cache_resource.clear()
                del st.session_state['_new_models']
                st.success("✅ Новите модели са приложени! Страницата ще се презареди.")
                st.rerun()
        with col_discard:
            if st.button("❌ Откажи — запази старите", use_container_width=True):
                del st.session_state['_new_models']
                st.info("Старите модели са запазени.")

    # ════════════════════════════════════════
    #  ROLLBACK
    # ════════════════════════════════════════
    st.divider()
    st.subheader("⏪ Rollback към предишна версия")
    st.warning("Използвай само ако новите модели дават лоши резултати!")

    if st.button("⏪ Върни предишните модели", use_container_width=True, key="rollback_btn"):
        ok = rollback_models()
        if ok:
            st.cache_resource.clear()
            st.success("✅ Успешен rollback! Страницата ще се презареди.")
            st.rerun()
        else:
            st.error("Няма запазени backup модели.")

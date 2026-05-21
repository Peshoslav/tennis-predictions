import pandas as pd
import numpy as np
import streamlit as st

# ──────────────────────────────────────────────
# КОЛОНИ ОТ БАЗАТА НА ДЖЕФ
# ──────────────────────────────────────────────
NEEDED_COLS = [
    'tourney_date', 'winner_name', 'loser_name', 'surface',
    'winner_rank', 'loser_rank', 'best_of',
    'w_1stWon', 'w_svpt', 'l_1stWon', 'l_svpt',
    'w_ace', 'l_ace', 'winner_age', 'loser_age'
]

SURFACE_FEATURES = {
    'Hard':  ['elo_surf_diff', '1st_won_diff', 'age_diff', 'form_diff', 'rest_days_diff', 'h2h_cum', 'h2h_surf_roll'],
    'Grass': ['elo_surf_diff', '1st_won_diff', 'age_diff', 'form_diff', 'ace_diff', 'h2h_surf_roll'],
    'Clay':  ['elo_surf_diff', '1st_won_diff', 'age_diff', 'form_diff', 'rest_days_diff', 'h2h_cum'],
}

# ──────────────────────────────────────────────
# 1. ЗАРЕЖДАНЕ НА ДАННИТЕ (кешира се за 12 часа)
# ──────────────────────────────────────────────
@st.cache_data(ttl=43200, show_spinner="📥 Зарежда базата на Джеф (1 минута)...")
def load_master():
    dfs = []
    for year in range(2018, 2027):
        url = (f"https://raw.githubusercontent.com/JeffSackmann/"
               f"tennis_atp/master/atp_matches_{year}.csv")
        try:
            df = pd.read_csv(url, usecols=lambda c: c in NEEDED_COLS)
            df['tourney_date'] = pd.to_datetime(
                df['tourney_date'], format='%Y%m%d', errors='coerce')
            dfs.append(df)
        except Exception:
            pass
    master = (pd.concat(dfs, ignore_index=True)
                .sort_values('tourney_date')
                .reset_index(drop=True))
    return master


# ──────────────────────────────────────────────
# 2. ELO ПО НАСТИЛКА
# ──────────────────────────────────────────────
def _compute_elo(master, K=32, start=1500):
    """Връща речник {играч: {настилка: текущ_elo}}"""
    elo = {}
    for row in master[['winner_name', 'loser_name', 'surface']].itertuples(index=False):
        w, l, surf = row.winner_name, row.loser_name, row.surface
        if pd.isna(surf):
            surf = 'Hard'
        w_e = elo.setdefault(w, {}).get(surf, start)
        l_e = elo.setdefault(l, {}).get(surf, start)
        exp_w = 1 / (1 + 10 ** ((l_e - w_e) / 400))
        elo[w][surf] = w_e + K * (1 - exp_w)
        elo[l][surf] = l_e + K * (0 - (1 - exp_w))
    return elo


# ──────────────────────────────────────────────
# 3. ИЗГРАЖДАНЕ НА СТАТИСТИКИ (кешира се)
# ──────────────────────────────────────────────
@st.cache_data(ttl=43200, show_spinner="⚙️ Изчислява статистики...")
def build_player_stats(_master):
    master = _master  # underscore prefix = не се хешира аргумента

    # ── Elo ──────────────────────────────────
    current_elo = _compute_elo(master)
    elo_rows = []
    for player, surf_map in current_elo.items():
        row = {'player': player}
        for surf, val in surf_map.items():
            row[f'elo_{surf}'] = val
        elo_rows.append(row)
    elo_df = pd.DataFrame(elo_rows).set_index('player')

    # ── Сервис статистики ────────────────────
    serve = pd.concat([
        master[['tourney_date', 'winner_name', 'w_1stWon', 'w_svpt', 'w_ace']].rename(
            columns={'winner_name': 'player',
                     'w_1stWon': 'first_won', 'w_svpt': 'svpt', 'w_ace': 'ace'}),
        master[['tourney_date', 'loser_name', 'l_1stWon', 'l_svpt', 'l_ace']].rename(
            columns={'loser_name': 'player',
                     'l_1stWon': 'first_won', 'l_svpt': 'svpt', 'l_ace': 'ace'})
    ]).sort_values(['player', 'tourney_date'])

    serve['first_pct'] = serve['first_won'] / serve['svpt']
    serve['ace_pct']   = serve['ace'] / serve['svpt']

    serve['roll_first'] = serve.groupby('player')['first_pct'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    serve['roll_ace'] = serve.groupby('player')['ace_pct'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    serve['roll_hold'] = serve['roll_first']  # hold ≈ first serve won %

    latest_serve = serve.groupby('player')[['roll_first', 'roll_ace', 'roll_hold']].last()

    # ── Форма (last 10) ──────────────────────
    form = pd.concat([
        master[['tourney_date', 'winner_name']].assign(res=1).rename(columns={'winner_name': 'player'}),
        master[['tourney_date', 'loser_name']].assign(res=0).rename(columns={'loser_name': 'player'})
    ]).sort_values(['player', 'tourney_date'])

    form['roll_form'] = form.groupby('player')['res'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    latest_form = form.groupby('player')[['roll_form']].last()

    # ── Възраст ──────────────────────────────
    ages = pd.concat([
        master[['tourney_date', 'winner_name', 'winner_age']].rename(
            columns={'winner_name': 'player', 'winner_age': 'age'}),
        master[['tourney_date', 'loser_name', 'loser_age']].rename(
            columns={'loser_name': 'player', 'loser_age': 'age'})
    ]).sort_values(['player', 'tourney_date'])
    latest_age = ages.groupby('player')[['age']].last()

    # ── Дни почивка ──────────────────────────
    dates = pd.concat([
        master[['tourney_date', 'winner_name']].rename(columns={'winner_name': 'player'}),
        master[['tourney_date', 'loser_name']].rename(columns={'loser_name': 'player'})
    ]).sort_values(['player', 'tourney_date'])
    dates['prev'] = dates.groupby('player')['tourney_date'].shift(1)
    dates['rest_days'] = (dates['tourney_date'] - dates['prev']).dt.days
    latest_rest = dates.groupby('player')[['rest_days']].last()

    # ── Ранглиста ─────────────────────────────
    ranks = pd.concat([
        master[['tourney_date', 'winner_name', 'winner_rank']].rename(
            columns={'winner_name': 'player', 'winner_rank': 'rank'}),
        master[['tourney_date', 'loser_name', 'loser_rank']].rename(
            columns={'loser_name': 'player', 'loser_rank': 'rank'})
    ]).sort_values(['player', 'tourney_date'])
    latest_rank = ranks.groupby('player')[['rank']].last()

    # ── Комбиниране ──────────────────────────
    stats = (latest_serve
             .join(latest_form,  how='outer')
             .join(latest_age,   how='outer')
             .join(latest_rest,  how='outer')
             .join(latest_rank,  how='outer')
             .join(elo_df,       how='outer'))

    return stats


# ──────────────────────────────────────────────
# 4. H2H МЕЖДУ ДВАМА ИГРАЧА
# ──────────────────────────────────────────────
def compute_h2h(master, p1, p2, surface):
    mask = (
        ((master['winner_name'] == p1) & (master['loser_name'] == p2)) |
        ((master['winner_name'] == p2) & (master['loser_name'] == p1))
    )
    h2h = master[mask].copy()

    p1_wins_total = (h2h['winner_name'] == p1).sum()
    p2_wins_total = (h2h['winner_name'] == p2).sum()
    h2h_cum = int(p1_wins_total - p2_wins_total)

    surf_h2h = h2h[h2h['surface'] == surface].copy()
    if len(surf_h2h) > 0:
        surf_h2h['p1_win'] = (surf_h2h['winner_name'] == p1).astype(int)
        h2h_surf_roll = float(surf_h2h['p1_win'].rolling(5, min_periods=1).mean().iloc[-1])
    else:
        h2h_surf_roll = 0.5

    return h2h_cum, h2h_surf_roll


# ──────────────────────────────────────────────
# 5. ИЗГРАЖДАНЕ НА ПРИЗНАЦИ ЗА МОДЕЛА
# ──────────────────────────────────────────────
def build_features(p1, p2, surface, stats, master):
    s1 = stats.loc[p1] if p1 in stats.index else pd.Series(dtype=float)
    s2 = stats.loc[p2] if p2 in stats.index else pd.Series(dtype=float)

    elo1 = s1.get(f'elo_{surface}', 1500.0)
    elo2 = s2.get(f'elo_{surface}', 1500.0)

    h2h_cum, h2h_surf_roll = compute_h2h(master, p1, p2, surface)

    r1 = float(s1.get('rank', 200) or 200)
    r2 = float(s2.get('rank', 200) or 200)

    feats = {
        'elo_surf_diff':  float(elo1) - float(elo2),
        '1st_won_diff':   float(s1.get('roll_first', 0.65) or 0.65) - float(s2.get('roll_first', 0.65) or 0.65),
        'age_diff':       float(s1.get('age', 26) or 26) - float(s2.get('age', 26) or 26),
        'form_diff':      float(s1.get('roll_form', 0.5) or 0.5) - float(s2.get('roll_form', 0.5) or 0.5),
        'rest_days_diff': float(s1.get('rest_days', 3) or 3) - float(s2.get('rest_days', 3) or 3),
        'h2h_cum':        h2h_cum,
        'h2h_surf_roll':  h2h_surf_roll,
        'ace_diff':       float(s1.get('roll_ace', 0.05) or 0.05) - float(s2.get('roll_ace', 0.05) or 0.05),
        # BO3 признаци
        'rank_diff':      abs(r1 - r2),
        'w_roll_hold':    float(s1.get('roll_hold', 0.6) or 0.6) if r1 <= r2 else float(s2.get('roll_hold', 0.6) or 0.6),
        'l_roll_hold':    float(s2.get('roll_hold', 0.6) or 0.6) if r1 <= r2 else float(s1.get('roll_hold', 0.6) or 0.6),
    }
    return feats

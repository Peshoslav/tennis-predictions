import pandas as pd
import numpy as np
import re
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import clone

# ──────────────────────────────────────────────────────────────────────────────
NEEDED_COLS = [
    'tourney_date', 'winner_name', 'loser_name', 'surface',
    'winner_rank', 'loser_rank', 'best_of',
    'w_1stWon', 'w_svpt', 'l_1stWon', 'l_svpt',
    'w_ace', 'l_ace', 'winner_age', 'loser_age', 'score'
]

SURFACE_FEATURES = {
    'Hard':  ['elo_surf_diff', '1st_won_diff', 'age_diff', 'form_diff', 'rest_days_diff', 'h2h_cum', 'h2h_surf_roll'],
    'Grass': ['elo_surf_diff', '1st_won_diff', 'age_diff', 'form_diff', 'ace_diff', 'h2h_surf_roll'],
    'Clay':  ['elo_surf_diff', '1st_won_diff', 'age_diff', 'form_diff', 'rest_days_diff', 'h2h_cum'],
}

BO3_FEATURES = ['rank_diff', 'w_roll_hold', 'l_roll_hold']

# ──────────────────────────────────────────────────────────────────────────────
# 1. ЗАРЕЖДАНЕ НА БАЗАТА НА ДЖЕФ
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=21600, show_spinner="📥 Зарежда базата на Джеф...")
def load_master():
    dfs = []
    for year in range(2018, 2027):
        url = (f"https://raw.githubusercontent.com/JeffSackmann/"
               f"tennis_atp/master/atp_matches_{year}.csv")
        try:
            df = pd.read_csv(url, usecols=lambda c: c in NEEDED_COLS)
            df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
            dfs.append(df)
        except Exception:
            pass
    master = pd.concat(dfs, ignore_index=True).sort_values('tourney_date').reset_index(drop=True)
    master['total_games'] = master['score'].apply(
        lambda x: sum([int(w)+int(l) for w, l in re.findall(r'(\d+)-(\d+)', str(x))])
        if '-' in str(x) else np.nan
    )
    return master

# ──────────────────────────────────────────────────────────────────────────────
# 2. ELO
# ──────────────────────────────────────────────────────────────────────────────
def _compute_elo(master, K=32, start=1500):
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

# ──────────────────────────────────────────────────────────────────────────────
# 3. СТАТИСТИКИ НА ИГРАЧИТЕ
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=21600, show_spinner="⚙️ Изчислява статистики...")
def build_player_stats(_master):
    master = _master

    current_elo = _compute_elo(master)
    elo_rows = [{'player': p, **{f'elo_{s}': v for s, v in sv.items()}}
                for p, sv in current_elo.items()]
    elo_df = pd.DataFrame(elo_rows).set_index('player')

    serve = pd.concat([
        master[['tourney_date', 'winner_name', 'w_1stWon', 'w_svpt', 'w_ace']].rename(
            columns={'winner_name': 'player', 'w_1stWon': 'first_won', 'w_svpt': 'svpt', 'w_ace': 'ace'}),
        master[['tourney_date', 'loser_name', 'l_1stWon', 'l_svpt', 'l_ace']].rename(
            columns={'loser_name': 'player', 'l_1stWon': 'first_won', 'l_svpt': 'svpt', 'l_ace': 'ace'})
    ]).sort_values(['player', 'tourney_date'])
    serve['first_pct'] = serve['first_won'] / serve['svpt']
    serve['ace_pct']   = serve['ace'] / serve['svpt']
    serve['roll_first'] = serve.groupby('player')['first_pct'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    serve['roll_ace'] = serve.groupby('player')['ace_pct'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    serve['roll_hold'] = serve['roll_first']
    latest_serve = serve.groupby('player')[['roll_first', 'roll_ace', 'roll_hold']].last()

    form = pd.concat([
        master[['tourney_date', 'winner_name']].assign(res=1).rename(columns={'winner_name': 'player'}),
        master[['tourney_date', 'loser_name']].assign(res=0).rename(columns={'loser_name': 'player'})
    ]).sort_values(['player', 'tourney_date'])
    form['roll_form'] = form.groupby('player')['res'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    latest_form = form.groupby('player')[['roll_form']].last()

    ages = pd.concat([
        master[['tourney_date', 'winner_name', 'winner_age']].rename(
            columns={'winner_name': 'player', 'winner_age': 'age'}),
        master[['tourney_date', 'loser_name', 'loser_age']].rename(
            columns={'loser_name': 'player', 'loser_age': 'age'})
    ]).sort_values(['player', 'tourney_date'])
    latest_age = ages.groupby('player')[['age']].last()

    dates = pd.concat([
        master[['tourney_date', 'winner_name']].rename(columns={'winner_name': 'player'}),
        master[['tourney_date', 'loser_name']].rename(columns={'loser_name': 'player'})
    ]).sort_values(['player', 'tourney_date'])
    dates['prev'] = dates.groupby('player')['tourney_date'].shift(1)
    dates['rest_days'] = (dates['tourney_date'] - dates['prev']).dt.days
    latest_rest = dates.groupby('player')[['rest_days']].last()

    ranks = pd.concat([
        master[['tourney_date', 'winner_name', 'winner_rank']].rename(
            columns={'winner_name': 'player', 'winner_rank': 'rank'}),
        master[['tourney_date', 'loser_name', 'loser_rank']].rename(
            columns={'loser_name': 'player', 'loser_rank': 'rank'})
    ]).sort_values(['player', 'tourney_date'])
    latest_rank = ranks.groupby('player')[['rank']].last()

    stats = (latest_serve.join(latest_form, how='outer')
                         .join(latest_age,  how='outer')
                         .join(latest_rest, how='outer')
                         .join(latest_rank, how='outer')
                         .join(elo_df,      how='outer'))
    return stats

# ──────────────────────────────────────────────────────────────────────────────
# 4. H2H
# ──────────────────────────────────────────────────────────────────────────────
def compute_h2h(master, p1, p2, surface):
    mask = (
        ((master['winner_name'] == p1) & (master['loser_name'] == p2)) |
        ((master['winner_name'] == p2) & (master['loser_name'] == p1))
    )
    h2h = master[mask]
    h2h_cum = int((h2h['winner_name'] == p1).sum() - (h2h['winner_name'] == p2).sum())
    surf_h2h = h2h[h2h['surface'] == surface]
    if len(surf_h2h):
        p1_win = (surf_h2h['winner_name'] == p1).astype(int)
        h2h_surf_roll = float(p1_win.rolling(5, min_periods=1).mean().iloc[-1])
    else:
        h2h_surf_roll = 0.5
    return h2h_cum, h2h_surf_roll

# ──────────────────────────────────────────────────────────────────────────────
# 5. ПРИЗНАЦИ ЗА МОДЕЛА
# ──────────────────────────────────────────────────────────────────────────────
def build_features(p1, p2, surface, stats, master):
    s1 = stats.loc[p1] if p1 in stats.index else pd.Series(dtype=float)
    s2 = stats.loc[p2] if p2 in stats.index else pd.Series(dtype=float)
    h2h_cum, h2h_surf_roll = compute_h2h(master, p1, p2, surface)
    r1 = float(s1.get('rank', 200) or 200)
    r2 = float(s2.get('rank', 200) or 200)
    top = s1 if r1 <= r2 else s2
    bot = s2 if r1 <= r2 else s1
    return {
        'elo_surf_diff':  float(s1.get(f'elo_{surface}', 1500) or 1500) - float(s2.get(f'elo_{surface}', 1500) or 1500),
        '1st_won_diff':   float(s1.get('roll_first', 0.65) or 0.65) - float(s2.get('roll_first', 0.65) or 0.65),
        'age_diff':       float(s1.get('age', 26) or 26) - float(s2.get('age', 26) or 26),
        'form_diff':      float(s1.get('roll_form', 0.5) or 0.5) - float(s2.get('roll_form', 0.5) or 0.5),
        'rest_days_diff': float(s1.get('rest_days', 3) or 3) - float(s2.get('rest_days', 3) or 3),
        'h2h_cum':        h2h_cum,
        'h2h_surf_roll':  h2h_surf_roll,
        'ace_diff':       float(s1.get('roll_ace', 0.05) or 0.05) - float(s2.get('roll_ace', 0.05) or 0.05),
        'rank_diff':      abs(r1 - r2),
        'w_roll_hold':    float(top.get('roll_hold', 0.6) or 0.6),
        'l_roll_hold':    float(bot.get('roll_hold', 0.6) or 0.6),
    }

# ──────────────────────────────────────────────────────────────────────────────
# 6. BACKTEST
# ──────────────────────────────────────────────────────────────────────────────
def backtest_models(winner_models, bo3_model, master, stats, months=6):
    """
    Тества моделите на последните `months` месеца данни.
    Връща dict с метрики за всяка настилка + BO3.
    """
    cutoff = pd.Timestamp.today() - pd.DateOffset(months=months)
    test   = master[master['tourney_date'] >= cutoff].copy()

    results = {}

    for surface, feat_list in SURFACE_FEATURES.items():
        model = winner_models.get(surface)
        if model is None:
            results[surface] = {'error': 'Моделът не е зареден', 'total': 0}
            continue

        df_surf = test[test['surface'] == surface].dropna(subset=['winner_name', 'loser_name'])

        if df_surf.empty:
            results[surface] = {
                'error': f'Няма {surface} мачове в последните {months} месеца (сезонен проблем)',
                'total': 0
            }
            continue

        correct, total, errors = 0, 0, 0
        for _, row in df_surf.iterrows():
            try:
                feats = build_features(row['winner_name'], row['loser_name'], surface, stats, master)
                X     = pd.DataFrame([[feats[f] for f in feat_list]], columns=feat_list)
                prob  = float(model.predict_proba(X)[0][1])
                if prob > 0.5:
                    correct += 1
                total += 1
            except Exception:
                errors += 1

        if total == 0:
            results[surface] = {
                'error': f'Всички {len(df_surf)} мача върнаха грешка при изчисление',
                'total': 0
            }
        else:
            results[surface] = {
                'accuracy': correct / total,
                'correct':  correct,
                'total':    total,
                'skipped':  errors,
            }

    # BO3 MAE
    if bo3_model is not None:
        df_bo3 = test[(test['best_of'] == 3) & test['total_games'].notna()]
        if not df_bo3.empty:
            preds, actuals = [], []
            for _, row in df_bo3.iterrows():
                try:
                    s1 = stats.loc[row['winner_name']] if row['winner_name'] in stats.index else pd.Series(dtype=float)
                    s2 = stats.loc[row['loser_name']]  if row['loser_name']  in stats.index else pd.Series(dtype=float)
                    r1 = float(s1.get('rank', 200) or 200)
                    r2 = float(s2.get('rank', 200) or 200)
                    X  = pd.DataFrame([[
                        abs(r1 - r2),
                        float(s1.get('roll_hold', 0.6) or 0.6) if r1 <= r2 else float(s2.get('roll_hold', 0.6) or 0.6),
                        float(s2.get('roll_hold', 0.6) or 0.6) if r1 <= r2 else float(s1.get('roll_hold', 0.6) or 0.6),
                    ]], columns=BO3_FEATURES)
                    preds.append(float(bo3_model.predict(X)[0]))
                    actuals.append(float(row['total_games']))
                except Exception:
                    pass
            if preds:
                mae = float(np.mean(np.abs(np.array(preds) - np.array(actuals))))
                results['BO3'] = {'mae': mae, 'total': len(preds)}

    return results

# ──────────────────────────────────────────────────────────────────────────────
# 7. RETRAIN
# ──────────────────────────────────────────────────────────────────────────────
def retrain_winner_model(original_model, surface, master, stats, train_until_year=2025):
    feat_list = SURFACE_FEATURES[surface]
    df = master[
        (master['surface'] == surface) &
        (master['tourney_date'].dt.year <= train_until_year)
    ].dropna(subset=['winner_name', 'loser_name'])

    rows = []
    for _, row in df.iterrows():
        try:
            w, l = row['winner_name'], row['loser_name']
            if w not in stats.index or l not in stats.index:
                continue
            sw, sl = stats.loc[w], stats.loc[l]
            h2h_c, h2h_s = compute_h2h(master, w, l, surface)
            feats = {
                'elo_surf_diff':  float(sw.get(f'elo_{surface}', 1500) or 1500) - float(sl.get(f'elo_{surface}', 1500) or 1500),
                '1st_won_diff':   float(sw.get('roll_first', 0.65) or 0.65) - float(sl.get('roll_first', 0.65) or 0.65),
                'age_diff':       float(sw.get('age', 26) or 26) - float(sl.get('age', 26) or 26),
                'form_diff':      float(sw.get('roll_form', 0.5) or 0.5) - float(sl.get('roll_form', 0.5) or 0.5),
                'rest_days_diff': float(sw.get('rest_days', 3) or 3) - float(sl.get('rest_days', 3) or 3),
                'h2h_cum':        h2h_c,
                'h2h_surf_roll':  h2h_s,
                'ace_diff':       float(sw.get('roll_ace', 0.05) or 0.05) - float(sl.get('roll_ace', 0.05) or 0.05),
            }
            feat_row = [feats.get(f, 0) for f in feat_list]
            rows.append(feat_row + [1])
            rows.append([-v if isinstance(v, (int, float)) else v for v in feat_row] + [0])
        except Exception:
            pass

    if len(rows) < 50:
        return None

    data = pd.DataFrame(rows, columns=feat_list + ['label'])
    X, y = data[feat_list], data['label']
    new_model = clone(original_model)
    new_model.fit(X, y)
    return new_model


def retrain_bo3_model(master):
    import gc
    feat_list = BO3_FEATURES

    serve = pd.concat([
        master[['tourney_date', 'winner_name', 'w_1stWon', 'w_svpt']].rename(
            columns={'winner_name': 'player', 'w_1stWon': 'first_won', 'w_svpt': 'svpt'}),
        master[['tourney_date', 'loser_name', 'l_1stWon', 'l_svpt']].rename(
            columns={'loser_name': 'player', 'l_1stWon': 'first_won', 'l_svpt': 'svpt'})
    ]).sort_values(['player', 'tourney_date'])
    serve['hold'] = serve['first_won'] / serve['svpt']
    serve['roll_hold'] = serve.groupby('player')['hold'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    latest_hold = serve.groupby('player')['roll_hold'].last()

    df = master[(master['best_of'] == 3) & master['total_games'].notna()].copy()
    df['rank_diff']   = abs(df['winner_rank'].fillna(500) - df['loser_rank'].fillna(500))
    df['w_roll_hold'] = df['winner_name'].map(latest_hold).fillna(0.6)
    df['l_roll_hold'] = df['loser_name'].map(latest_hold).fillna(0.6)

    train = df[df['tourney_date'].dt.year < 2026]
    test  = df[df['tourney_date'].dt.year >= 2026]

    model = RandomForestRegressor(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)
    model.fit(train[feat_list], train['total_games'])

    mae = None
    if not test.empty:
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(test['total_games'], model.predict(test[feat_list]))

    del serve
    gc.collect()
    return model, mae

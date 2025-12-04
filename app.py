import streamlit as st
import pandas as pd
import numpy as np
import time

# ----------------------------
# Roster template
# ----------------------------
ROSTER_TEMPLATE = {
    "qb": 1,
    "rb": 1,
    "wr": 2,
    "te": 1,
    "flex": 1  # rb/wr/te
}

# ----------------------------
# Utility functions
# ----------------------------
def normalize_series(s):
    s = s.astype(float)
    if s.nunique() <= 1:
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def snake_order(num_teams, rounds):
    order = []
    base = list(range(num_teams))
    for r in range(rounds):
        if r % 2 == 0:
            order.extend([(r, t) for t in base])
        else:
            order.extend([(r, t) for t in base[::-1]])
    return order

def init_team_roster():
    return {
        "positions": {"qb": [], "rb": [], "wr": [], "te": [], "flex": []},
        "team_counts": {}
    }

def can_add_player(player_row, team_roster):
    pos = str(player_row.get("position", "")).lower()
    if pos in ["qb","rb","wr","te"]:
        if len(team_roster["positions"][pos]) < ROSTER_TEMPLATE[pos]:
            return True
    if pos in ["rb","wr","te"]:
        if len(team_roster["positions"]["flex"]) < ROSTER_TEMPLATE["flex"]:
            return True
    return False

def assign_player(player_row, team_roster):
    pos = str(player_row.get("position", "")).lower()
    player = player_row.get("player", "")
    team = player_row.get("nflteam", "")
    if pos in ["qb","rb","wr","te"] and len(team_roster["positions"][pos]) < ROSTER_TEMPLATE[pos]:
        team_roster["positions"][pos].append(player)
    else:
        team_roster["positions"]["flex"].append(player)
    team_roster["team_counts"][(pos, team)] = team_roster["team_counts"].get((pos, team), 0) + 1

def compute_stack_bonus(player_row, team_roster):
    pos = str(player_row.get("position", "")).lower()
    team = player_row.get("nflteam", None)
    if not team or pd.isna(team):
        return 0.0
    has_qb = team_roster["team_counts"].get(("qb", team), 0) > 0
    has_wrte = team_roster["team_counts"].get(("wr", team), 0) > 0 or team_roster["team_counts"].get(("te", team), 0) > 0
    if pos in ["wr", "te"] and has_qb:
        return 0.1
    if pos == "qb" and has_wrte:
        return 0.1
    return 0.0

def pick_player(pool_df, team_roster, w_proj=1.0, w_adp=1.0, noise_scale=0.05):
    df = pool_df[pool_df.apply(lambda r: can_add_player(r, team_roster), axis=1)].copy()
    if df.empty:
        return None
    df["basescore"] = w_proj * df["projnorm"] + w_adp * df["adpnorm"]
    df["stackbonus"] = df.apply(lambda r: compute_stack_bonus(r, team_roster), axis=1)
    df["score"] = df["basescore"] + df["stackbonus"] + np.random.normal(0, noise_scale, len(df))
    choice = df.loc[df["score"].idxmax()]
    return choice

def simulate_draft(pool_df, num_teams=12, rounds=6, w_proj=1.0, w_adp=1.0):
    available = pool_df.copy()
    teams = [init_team_roster() for _ in range(num_teams)]
    order = snake_order(num_teams, rounds)
    picks = []
    for r, t in order:
        if available.empty:
            break
        choice = pick_player(available, teams[t], w_proj, w_adp)
        if choice is None:
            continue
        assign_player(choice, teams[t])
        picks.append({
            "Round": r+1,
            "Team": t+1,
            "Player": choice.get("player", None),
            "Position": choice.get("position", None),
            "NFLTeam": choice.get("nflteam", None),
            "ADP": choice.get("adp", None),
            "ETRProj": choice.get("etrproj", None),
            "UDProj": choice.get("udproj", None)
        })
        available = available[available["player"] != choice.get("player", None)]
    return pd.DataFrame(picks)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("12-Man Draft Simulator (Half PPR)")
st.sidebar.header("Upload CSVs")

ud_file = st.sidebar.file_uploader("Upload 12 Man UD CSV", type=["csv"])
etr_file = st.sidebar.file_uploader("Upload ETR Proj CSV", type=["csv"])

if ud_file and etr_file:
    ud_df = pd.read_csv(ud_file)
    etr_df = pd.read_csv(etr_file)

    # Normalize column names
    ud_df.columns = ud_df.columns.str.strip().str.lower()
    etr_df.columns = etr_df.columns.str.strip().str.lower()

    # Clean UD
    ud_df["player"] = ud_df["firstname"] + " " + ud_df["lastname"]
    ud_df.rename(columns={"adp":"adp","projectedpoints":"udproj","teamname":"nflteam"}, inplace=True)

    # Clean ETR — use Half PPR Proj
    etr_df.rename(columns={"pos":"position","team":"nflteam","half ppr proj":"etrproj"}, inplace=True)

    # Merge safely (only include columns that exist)
    merge_cols = [c for c in ["player","position","nflteam","etrproj"] if c in etr_df.columns]
    pool_df = pd.merge(ud_df, etr_df[merge_cols], on="player", how="left")

    # Normalize
    pool_df["projnorm"] = normalize_series(pool_df["etrproj"].fillna(pool_df["udproj"]))
    inv_adp = pool_df["adp"].max() - pool_df["adp"]
    pool_df["adpnorm"] = normalize_series(inv_adp)

    st.subheader("Merged Player Pool (Half PPR)")
    expected_cols = ["player","position","nflteam","adp","etrproj","udproj"]
    available_cols = [c for c in expected_cols if c in pool_df.columns]
    st.dataframe(pool_df[available_cols].head(20))

    # Settings
    num_teams = st.sidebar.number_input("Teams", 2, 20, 12)
    rounds = st.sidebar.number_input("Rounds", 1, 20, 6)
    w_proj = st.sidebar.slider("Projection weight", 0.0, 2.0, 1.0, 0.1)
    w_adp = st.sidebar.slider("ADP weight", 0.0, 2.0, 1.0, 0.1)
    sims = st.sidebar.number_input("Number of simulations", 1, 50, 1)

    if st.sidebar.button("Run Simulation"):
        all_picks = []
        for i in range(sims):
            picks_df = simulate_draft(pool_df, num_teams, rounds, w_proj, w_adp)
            picks_df["Sim"] = i+1
            all_picks.append(picks_df)
        result_df = pd.concat(all_picks, ignore_index=True)

        st.subheader("Draft Results")
        st.dataframe(result_df)

        st.download_button("Download Draft CSV", result_df.to_csv(index=False).encode("utf-8"), file_name=f"drafts_{int(time.time())}.csv", mime="text/csv")

else:
    st.info("Upload both 12 Man UD.csv and ETR Proj.csv to start.")

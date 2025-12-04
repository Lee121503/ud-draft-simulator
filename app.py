import streamlit as st
import pandas as pd
import numpy as np
import time

# ----------------------------
# Roster template
# ----------------------------
ROSTER_TEMPLATE = {
    "QB": 1,
    "RB": 1,
    "WR": 2,
    "TE": 1,
    "FLEX": 1  # RB/WR/TE
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
        "positions": {"QB": [], "RB": [], "WR": [], "TE": [], "FLEX": []},
        "team_counts": {}
    }

def can_add_player(player_row, team_roster):
    pos = player_row["Position"]
    # direct slots
    if pos in ["QB","RB","WR","TE"]:
        if len(team_roster["positions"][pos]) < ROSTER_TEMPLATE[pos]:
            return True
    # flex slot
    if pos in ["RB","WR","TE"]:
        if len(team_roster["positions"]["FLEX"]) < ROSTER_TEMPLATE["FLEX"]:
            return True
    return False

def assign_player(player_row, team_roster):
    pos = player_row["Position"]
    if pos in ["QB","RB","WR","TE"] and len(team_roster["positions"][pos]) < ROSTER_TEMPLATE[pos]:
        team_roster["positions"][pos].append(player_row["Player"])
    else:
        team_roster["positions"]["FLEX"].append(player_row["Player"])
    team_roster["team_counts"][(pos, player_row["NFLTeam"])] = team_roster["team_counts"].get((pos, player_row["NFLTeam"]), 0) + 1

def compute_stack_bonus(player_row, team_roster):
    pos = player_row["Position"]
    team = player_row["NFLTeam"]
    if pd.isna(team) or team == "":
        return 0.0
    has_qb = team_roster["team_counts"].get(("QB", team), 0) > 0
    has_wrte = team_roster["team_counts"].get(("WR", team), 0) > 0 or team_roster["team_counts"].get(("TE", team), 0) > 0
    if pos in ["WR", "TE"] and has_qb:
        return 0.1
    if pos == "QB" and has_wrte:
        return 0.1
    return 0.0

def pick_player(pool_df, team_roster, w_proj=1.0, w_adp=1.0, noise_scale=0.05):
    df = pool_df[pool_df.apply(lambda r: can_add_player(r, team_roster), axis=1)].copy()
    if df.empty:
        return None
    df["BaseScore"] = w_proj * df["ProjNorm"] + w_adp * df["ADPNorm"]
    df["StackBonus"] = df.apply(lambda r: compute_stack_bonus(r, team_roster), axis=1)
    df["Score"] = df["BaseScore"] + df["StackBonus"] + np.random.normal(0, noise_scale, len(df))
    choice = df.loc[df["Score"].idxmax()]
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
            "Player": choice["Player"],
            "Position": choice["Position"],
            "NFLTeam": choice["NFLTeam"],
            "ADP": choice.get("ADP", None),
            "ETRProj": choice.get("ETRProj", None),
            "UDProj": choice.get("UDProj", None)
        })
        available = available[available["Player"] != choice["Player"]]
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

    # Clean UD
    ud_df["Player"] = ud_df["firstName"] + " " + ud_df["lastName"]
    ud_df.rename(columns={"adp":"ADP","projectedPoints":"UDProj","teamName":"NFLTeam"}, inplace=True)

    # Clean ETR — now using Half PPR Proj
    etr_df.rename(columns={"Pos":"Position","Team":"NFLTeam","Half PPR Proj":"ETRProj"}, inplace=True)

    # Merge
    pool_df = pd.merge(ud_df, etr_df[["Player","Position","NFLTeam","ETRProj"]], on="Player", how="left")

    # Normalize
    pool_df["ProjNorm"] = normalize_series(pool_df["ETRProj"].fillna(pool_df["UDProj"]))
    inv_adp = pool_df["ADP"].max() - pool_df["ADP"]
    pool_df["ADPNorm"] = normalize_series(inv_adp)

    st.subheader("Merged Player Pool (Half PPR)")
    expected_cols = ["Player","Position","NFLTeam","ADP","ETRProj","UDProj"]
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

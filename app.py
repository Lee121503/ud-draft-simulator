import streamlit as st
import pandas as pd
import numpy as np
import time
import random

ROSTER_TEMPLATE = {"qb":1,"rb":1,"wr":2,"te":1,"flex":1}

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
    return {"positions":{"qb":[],"rb":[],"wr":[],"te":[],"flex":[]}, "team_counts":{}}

def can_add_player(player_row, team_roster):
    pos = str(player_row.get("position","")).lower()
    if pos in ["qb","rb","wr","te"]:
        if len(team_roster["positions"][pos]) < ROSTER_TEMPLATE[pos]:
            return True
    if pos in ["rb","wr","te"]:
        if len(team_roster["positions"]["flex"]) < ROSTER_TEMPLATE["flex"]:
            return True
    return False

def assign_player(player_row, team_roster):
    pos = str(player_row.get("position","")).lower()
    player = player_row.get("player","")
    team = player_row.get("nflteam","")
    if pos in ["qb","rb","wr","te"] and len(team_roster["positions"][pos]) < ROSTER_TEMPLATE[pos]:
        team_roster["positions"][pos].append(player)
    else:
        team_roster["positions"]["flex"].append(player)
    team_roster["team_counts"][(pos, team)] = team_roster["team_counts"].get((pos, team),0)+1

def compute_stack_bonus(player_row, team_roster):
    pos = str(player_row.get("position","")).lower()
    team = player_row.get("nflteam",None)
    if not team or pd.isna(team):
        return 0.0
    has_qb = team_roster["team_counts"].get(("qb",team),0)>0
    has_wrte = team_roster["team_counts"].get(("wr",team),0)>0 or team_roster["team_counts"].get(("te",team),0)>0
    if pos in ["wr","te"] and has_qb: return 0.1
    if pos=="qb" and has_wrte: return 0.1
    return 0.0

def pick_player(pool_df, team_roster, w_proj=1.0, w_adp=1.0, noise_scale=0.05):
    df = pool_df[pool_df.apply(lambda r: can_add_player(r, team_roster), axis=1)].copy()
    if df.empty: return None
    df["basescore"] = w_proj*df["vornorm"] + w_adp*df["adpnorm"]
    df["stackbonus"] = df.apply(lambda r: compute_stack_bonus(r, team_roster), axis=1)
    df["score"] = df["basescore"] + df["stackbonus"] + np.random.normal(0,noise_scale,len(df))
    return df.loc[df["score"].idxmax()]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("12-Man Draft Simulator (Manual Draft Mode + Draft Board)")
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

    merge_cols = [c for c in ["player","position","nflteam","etrproj"] if c in etr_df.columns]
    pool_df = pd.merge(ud_df, etr_df[merge_cols], on="player", how="left")

    # Replacement-level cutoffs (12-team defaults)
    replacement_cutoffs = {"qb":12,"rb":24,"wr":36,"te":12}
    vorp_values={}
    for pos,cutoff in replacement_cutoffs.items():
        pos_df = pool_df[pool_df["position"].str.lower()==pos].sort_values("etrproj",ascending=False)
        if len(pos_df)>=cutoff:
            vorp_values[pos] = pos_df.iloc[cutoff-1]["etrproj"]
        else:
            vorp_values[pos] = pos_df["etrproj"].min() if not pos_df.empty else 0

    pool_df["vorp"] = pool_df.apply(lambda r: r["etrproj"]-vorp_values.get(str(r.get("position","")).lower(),0), axis=1)

    # Normalize VORP and ADP
    pool_df["vornorm"] = normalize_series(pool_df["vorp"].fillna(0))
    inv_adp = pool_df["adp"].max()-pool_df["adp"]
    pool_df["adpnorm"] = normalize_series(inv_adp)

    st.subheader("Merged Player Pool (Half PPR + VORP)")
    expected_cols=["player","position","nflteam","adp","etrproj","udproj","vorp"]
    available_cols=[c for c in expected_cols if c in pool_df.columns]
    st.dataframe(pool_df[available_cols].head(20))

    # Settings
    num_teams=st.sidebar.number_input("Teams",2,20,12)
    rounds=st.sidebar.number_input("Rounds",1,20,6)
    w_proj=st.sidebar.slider("Projection weight",0.0,2.0,1.0,0.1)
    w_adp=st.sidebar.slider("ADP weight",0.0,2.0,1.0,0.1)

    # --- Persistent state ---
    if "my_team" not in st.session_state:
        st.session_state.my_team = random.randint(0, num_teams-1)
    st.sidebar.write(f"You are Team {st.session_state.my_team+1}")

    if "picks" not in st.session_state:
        st.session_state.picks = []
    if "available" not in st.session_state:
        st.session_state.available = pool_df.copy()
    if "teams" not in st.session_state:
        st.session_state.teams = [init_team_roster() for _ in range(num_teams)]
    if "order" not in st.session_state:
        st.session_state.order = snake_order(num_teams, rounds)
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "awaiting_pick" not in st.session_state:
        st.session_state.awaiting_pick = False

    # --- Advance Draft ---
    if st.sidebar.button("Advance Draft"):
        if st.session_state.current_index < len(st.session_state.order):
            r,t = st.session_state.order[st.session_state.current_index]
            if t == st.session_state.my_team:
                st.session_state.awaiting_pick = True
            else:
                choice = pick_player(st.session_state.available, st.session_state.teams[t], w_proj, w_adp)
                if choice is not None:
                    assign_player(choice, st.session_state.teams[t])
                    st.session_state.picks.append({
                        "Round": r+1, "Team": t+1,
                        "Player": choice.get("player", None),
                        "Position": choice.get("position", None),
                        "NFLTeam": choice.get("nflteam", None),
                        "ADP": choice.get("adp", None),
                        "ETRProj": choice.get("etrproj", None),
                        "UDProj": choice.get("udproj", None),
                        "VORP": choice.get("vorp", None)
                    })
                    st.session_state.available = st.session_state.available[
                        st.session_state.available["player"] != choice.get("player", None)
                    ]
                st.session_state.current_index += 1

    # --- Manual Pick UI ---
    if st.session_state.awaiting_pick:
        r, t = st.session_state.order[st.session_state.current_index]
        st.subheader(f"Round {r+1}, Your Pick (Team {t+1})")
        options = st.session_state.available["player"].tolist()
        choice_name = st.selectbox("Select your player:", options, key=f"pick_{r}_{t}")
    
        if st.button("Confirm Pick", key=f"confirm_{r}_{t}"):
            choice = st.session_state.available[
                st.session_state.available["player"] == choice_name
            ].iloc[0]
    
            if can_add_player(choice, st.session_state.teams[t]):
                # ✅ valid pick
                assign_player(choice, st.session_state.teams[t])
                st.session_state.picks.append({
                    "Round": r+1, "Team": t+1,
                    "Player": choice.get("player", None),
                    "Position": choice.get("position", None),
                    "NFLTeam": choice.get("nflteam", None),
                    "ADP": choice.get("adp", None),
                    "ETRProj": choice.get("etrproj", None),
                    "UDProj": choice.get("udproj", None),
                    "VORP": choice.get("vorp", None)
                })
                st.session_state.available = st.session_state.available[
                    st.session_state.available["player"] != choice_name
                ]
                st.session_state.current_index += 1
                st.session_state.awaiting_pick = False
            else:
                # ❌ invalid pick — stay on your turn
                st.warning("Roster restriction prevents adding this player. Please select another.")
                # Do NOT advance index, do NOT reset awaiting_pick


    # --- Show results so far ---
    result_df = pd.DataFrame(st.session_state.picks)
    if not result_df.empty:
        st.subheader("Draft Results")
        st.dataframe(result_df)

        # Draft board view (Rounds × Teams grid)
        board = result_df.pivot(index="Round", columns="Team", values="Player")
        st.subheader("Draft Board (Rounds × Teams)")
        st.dataframe(board)

        # Download button
        st.download_button(
            "Download Draft CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name=f"drafts_{int(time.time())}.csv",
            mime="text/csv"
        )

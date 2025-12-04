import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import re
from rapidfuzz import process, fuzz
st.set_page_config(layout="wide")

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

    # Dedicated slot check
    if pos in ROSTER_TEMPLATE and len(team_roster["positions"][pos]) < ROSTER_TEMPLATE[pos]:
        return True

    # Flex slot check (RB/WR/TE eligible)
    if pos in ["rb","wr","te"] and len(team_roster["positions"]["flex"]) < ROSTER_TEMPLATE["flex"]:
        return True

    return False


def assign_player(player_row, team_roster):
    pos = str(player_row.get("position","")).lower()
    player = player_row.get("player","")
    team = player_row.get("nflteam","")

    # Try dedicated slot first
    if pos in ROSTER_TEMPLATE and len(team_roster["positions"][pos]) < ROSTER_TEMPLATE[pos]:
        team_roster["positions"][pos].append(player)
    # If RB/WR/TE, try flex slot
    elif pos in ["rb","wr","te"] and len(team_roster["positions"]["flex"]) < ROSTER_TEMPLATE["flex"]:
        team_roster["positions"]["flex"].append(player)
    else:
        # No valid slot available
        return False

    # Update team counts
    team_roster["team_counts"][(pos, team)] = team_roster["team_counts"].get((pos, team), 0) + 1
    return True


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

    # --- Name normalization helper ---
    def normalize_name(name: str) -> str:
        if not isinstance(name, str):
            return ""
        name = name.lower().strip()
        name = re.sub(r"[^\w\s]", "", name)  # remove punctuation
        name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)  # remove suffixes
        # common nickname harmonization
        name = name.replace("ken ", "kenneth ")
        name = name.replace("dj ", "deejay ")
        name = name.replace("aj ", "anthony ")
        name = re.sub(r"\s+", " ", name).strip()
        return name
    
    # --- Normalize names ---
    ud_df["player"] = ud_df["firstname"] + " " + ud_df["lastname"]
    ud_df["player_norm"] = ud_df["player"].apply(normalize_name)
    etr_df["player_norm"] = etr_df["player"].apply(normalize_name)
    
    # --- Fuzzy match UD -> ETR ---
    etr_names = etr_df["player_norm"].tolist()
    ud_df["etr_match_norm"] = ud_df["player_norm"].apply(
        lambda x: process.extractOne(x, etr_names, scorer=fuzz.token_sort_ratio)[0]
    )
    
    # --- Merge: keep UD’s position/team, bring in ETR projection ---
    pool_df = pd.merge(
        ud_df[["player","player_norm","etr_match_norm","adp","udproj","slotname","nflteam"]],
        etr_df[["player_norm","position","nflteam","etrproj"]],
        left_on="etr_match_norm",
        right_on="player_norm",
        how="left",
        suffixes=("_ud","_etr")
    )
    
    # Rename for clarity
    pool_df.rename(columns={
        "slotname":"position_ud",
        "position":"position_etr",
        "nflteam_ud":"nflteam_ud",
        "nflteam_etr":"nflteam_etr"
    }, inplace=True)
    
    # Backfill: use UD team if ETR missing
    pool_df["position"] = pool_df["position_ud"].fillna(pool_df["position_etr"])
    pool_df["nflteam"] = pool_df["nflteam_ud"].fillna(pool_df["nflteam_etr"])
    
    # Fill missing projections with 0
    pool_df["etrproj"] = pool_df["etrproj"].fillna(0)
    
    # Keep original UD display name
    pool_df["player_display"] = pool_df["player"]




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

    # --- Team slot selection ---
    slot_mode = st.sidebar.radio("Choose draft slot mode:", ["Random", "Manual"])
    
    if slot_mode == "Random":
        if "my_team" not in st.session_state:
            st.session_state.my_team = random.randint(0, num_teams-1)
    else:  # Manual
        manual_slot = st.sidebar.number_input("Select your draft slot (1–12)", 1, num_teams, 1)
        st.session_state.my_team = manual_slot - 1  # zero-based index
    
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
        while st.session_state.current_index < len(st.session_state.order):
            r, t = st.session_state.order[st.session_state.current_index]
    
            if t == st.session_state.my_team:
                # Stop when it's your turn
                st.session_state.awaiting_pick = True
                break
            else:
                # Simulate other team pick
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


    # --- Reset Draft ---
    if st.sidebar.button("Reset Draft"):
        st.session_state.picks = []
        st.session_state.available = pool_df.copy()
        st.session_state.teams = [init_team_roster() for _ in range(num_teams)]
        st.session_state.order = snake_order(num_teams, rounds)
        st.session_state.current_index = 0
        st.session_state.awaiting_pick = False
        st.success("Draft has been reset. Start again!")

    # --- Show results so far ---
    result_df = pd.DataFrame(st.session_state.picks)
    if not result_df.empty:
        st.subheader("Draft Results")
        st.dataframe(result_df)

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

        # Draft board view (Rounds × Teams grid)
        board = result_df.pivot(index="Round", columns="Team", values="Player")
        pos_board = result_df.pivot(index="Round", columns="Team", values="Position")

        def color_positions(val, pos):
            if pos is None:
                return ""
            pos = str(pos).lower()
            if pos == "rb":
                return "background-color: lightgreen"
            elif pos == "wr":
                return "background-color: khaki"
            elif pos == "qb":
                return "background-color: plum"
            elif pos == "te":
                return "background-color: lightblue"
            return ""

        # Build a DataFrame of styles with same shape as board
        styles = pd.DataFrame(
            [[color_positions(board.iloc[i, j], pos_board.iloc[i, j])
              for j in range(board.shape[1])]
             for i in range(board.shape[0])],
            index=board.index, columns=board.columns
        )

        styled_board = board.style.apply(lambda _: styles, axis=None)

        st.subheader("Draft Board (Rounds × Teams)")
        st.dataframe(styled_board)

        # Download button
        st.download_button(
            "Download Draft CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name=f"drafts_{int(time.time())}.csv",
            mime="text/csv"
        )

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. Dynamic Team Color Mapping ---
TEAM_COLORS = {
    'Chennai Super Kings': '#FDB913',       # Gold/Yellow
    'Mumbai Indians': '#004BA0',             # Blue
    'Royal Challengers Bengaluru': "#5C0808",# Black/Red
    'Kolkata Knight Riders': "#370B78",      # Purple
    'Delhi Capitals': '#00008B',             # Blue
    'Punjab Kings': '#DD1F2D',               # Red
    'Rajasthan Royals': '#EA1A85',           # Pink
    'Sunrisers Hyderabad': '#FF822A',        # Orange
    'Gujarat Titans': "#0D193F",             # Navy Blue
    'Lucknow Super Giants': "#19A6F1"        # Cyan
}

# --- 2. Page Config ---
st.set_page_config(
    page_title="IPL Prediction Pro",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. Optimized Asset Loading ---
@st.cache_resource
def load_ml_assets():
    pipe = pickle.load(open('pipe_ipl_2nd_innings.pkl', 'rb'))
    teams = pickle.load(open('teams_ipl_2nd_innings.pkl', 'rb'))
    venues = pickle.load(open('venues_ipl_2nd_innings.pkl', 'rb'))
    players = pickle.load(open('players_ipl_2nd_innings.pkl', 'rb'))
    sr_map = pickle.load(open('sr_lookup_ipl_2nd_innings.pkl', 'rb')) 
    role_map = pickle.load(open('role_lookup.pkl', 'rb'))
    venue_chase_map = pickle.load(open('venue_chase_ipl_2nd_innings.pkl', 'rb'))
    team_win_pct_map = pickle.load(open('team_win_pct_map.pkl', 'rb'))
    return pipe, teams, venues, players, sr_map, role_map, venue_chase_map, team_win_pct_map

pipe, teams, venues, players, sr_map, role_map, venue_chase_map, team_win_pct_map = load_ml_assets()

# --- 4. Sidebar: Match Context & Dynamic Theme Selection ---
with st.sidebar:
    st.title("🏟️ Match Setup")
    batting_team = st.selectbox('Batting Team', sorted(teams), index=0)
    bowling_team = st.selectbox('Bowling Team', sorted(teams), index=1)
    
    # Get colors based on selection for dynamic theme
    color1 = TEAM_COLORS.get(batting_team, '#1E3A8A')
    color2 = TEAM_COLORS.get(bowling_team, '#EAB308')
    
    selected_venue = st.selectbox('Stadium Venue', sorted(venues))
    
    # --- Re-added: Venue Chase Percentage Display ---
    v_chase_pct = venue_chase_map.get(selected_venue, 50.0)
    st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); padding: 12px; border-radius: 10px; border: 2px solid {color1}; margin-bottom: 15px; margin-top: 10px;">
            <small style="color: {color1}; font-weight: bold;">Historical Chase Win %</small><br>
            <span style="font-size: 22px; font-weight: bold; color: #ffffff;">{round(v_chase_pct, 1)}%</span>
        </div>
    """, unsafe_allow_html=True)

    target = st.number_input('Target Score', min_value=1, value=180)
    st.divider()
    calc_btn = st.button('🔥 RUN PREDICTION')

# --- 5. DYNAMIC CSS INJECTION (Blended Team Background) ---
st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, {color1} 0%, #0F172A 50%, {color2} 100%);
        color: #ffffff;
        transition: background 0.8s ease;
    }}
    [data-testid="stSidebar"] {{
        background-color: #0F172A;
        border-right: 3px solid {color1};
    }}
    div[data-testid="stMetric"] {{
        background: rgba(15, 23, 42, 0.7);
        padding: 15px;
        border-radius: 12px;
        border: 2px solid {color1};
    }}
    .stButton>button {{
        background: linear-gradient(90deg, {color1} 0%, {color2} 100%);
        color: white;
        border-radius: 10px;
        height: 3.5em;
        width: 100%;
        font-weight: bold;
        border: none;
    }}
    .player-card {{
        background: rgba(15, 23, 42, 0.5);
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid {color2};
        margin-bottom: 10px;
    }}
    h1, h2, h3, .stSubheader, label {{
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 6. Main Page UI ---
st.title("IPL Predictive Analytics Engine")

col_input, col_momentum = st.columns([2, 1])

with col_input:
    st.subheader("📊 Match Progress")
    sc1, sc2, sc3 = st.columns(3)
    score = sc1.number_input('Current Score', min_value=0)
    wickets = sc2.number_input('Wickets Down', min_value=0, max_value=9)
    overs = sc3.number_input('Overs Completed', min_value=7.0, max_value=20.0, value=7.0, step=0.1)
    
    st.subheader("🏏 On-Field Batsmen")
    b1, b2 = st.columns(2)
    def format_player_name(p): return p[1] if isinstance(p, (tuple, list)) else p

    striker = b1.selectbox('Striker', sorted(players), format_func=format_player_name)
    s_name = striker[1] if isinstance(striker, (tuple, list)) else striker
    s_sr = round(sr_map.get(s_name, 120.0), 2)
    s_role = role_map.get(s_name, "Middle Order")
    b1.markdown(f'<div class="player-card"><small>SR: {s_sr}</small><br><b>{s_role}</b></div>', unsafe_allow_html=True)
    
    non_striker = b2.selectbox('Non-Striker', sorted(players), format_func=format_player_name)
    ns_name = non_striker[1] if isinstance(non_striker, (tuple, list)) else non_striker
    ns_sr = round(sr_map.get(ns_name, 120.0), 2)
    ns_role = role_map.get(ns_name, "Middle Order")
    b2.markdown(f'<div class="player-card"><small>SR: {ns_sr}</small><br><b>{ns_role}</b></div>', unsafe_allow_html=True)

with col_momentum:
    st.subheader("📈 Momentum")
    l5_runs = st.number_input('Runs (Last 5 Overs)', min_value=0)
    l5_wickets = st.number_input('Wickets (Last 5 Overs)', min_value=0, max_value=5)

# --- 7. Prediction Logic ---
if calc_btn:
    if batting_team == bowling_team:
        st.error("Select different teams.")
    else:
        runs_left = target - score
        balls_left = 120 - (int(overs) * 6 + int((overs % 1) * 10))
        ball_per_run = balls_left / (runs_left + 1)
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
        
        pressure_index = min(rrr / (crr + 0.1), 10.0)
        bat_win_pct = team_win_pct_map.get(batting_team, 50.0)
        bowl_win_pct = team_win_pct_map.get(bowling_team, 50.0)

        input_df = pd.DataFrame({
            'batting_team_win_pct': [bat_win_pct], 'bowling_team_win_pct': [bowl_win_pct],
            'venue_chase_pct': [v_chase_pct], 'balls_per_run': [ball_per_run], 
            'wickets_remaining': [10-wickets], 'current_run_rate': [crr], 
            'required_run_rate': [rrr], 'last_5_over_runs': [l5_runs],
            'last_5_over_wickets': [l5_wickets], 'pp_wicket_count': [0],
            'striker_sr': [s_sr], 'non_striker_sr': [ns_sr], 
            'partnership_power': [0.8], 'striker_role': [s_role], 
            'non_striker_role': [ns_role], 'pressure_index': [pressure_index]
        })

        res = pipe.predict_proba(input_df)
        win, loss = res[0][1], res[0][0]
        
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Runs Required", int(runs_left))
        m2.metric("Balls Remaining", int(balls_left))
        m3.metric("Required RR", round(rrr, 2))
        m4.metric("Pressure Index", round(pressure_index, 2))

        chart_col, prob_col = st.columns([1.5, 1])
        with chart_col:
            fig = go.Figure(data=[go.Pie(
                labels=[f"{batting_team}", f"{bowling_team}"],
                values=[win, loss], hole=.7,
                marker_colors=[color1, color2]
            )])
            fig.update_layout(showlegend=True, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

        with prob_col:
            st.write(f"### {batting_team}: {round(win*100)}%")
            st.progress(int(win*100))
            st.write(f"### {bowling_team}: {round(loss*100)}%")

            st.progress(int(loss*100))
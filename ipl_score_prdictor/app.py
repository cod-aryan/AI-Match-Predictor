import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="IPL Prediction Pro", 
    page_icon="🏏", 
    layout="wide"
)

TEAM_COLORS = {
    'Chennai Super Kings': '#FDB913', 
    'Mumbai Indians': '#004BA0',
    'Royal Challengers Bengaluru': "#5C0808",
    'Kolkata Knight Riders': "#370B78",
    'Delhi Capitals': '#00008B',
    'Punjab Kings': '#DD1F2D',
    'Rajasthan Royals': '#EA1A85',
    'Sunrisers Hyderabad': '#FF822A',
    'Gujarat Titans': "#0D193F",
    'Lucknow Super Giants': "#19A6F1"
}

def get_dls_resource_pct(overs_left, wickets_lost):
    """Calculates DLS resource percentage based on standard tables."""
    resources = {
        20: {0: 56.6, 2: 52.4, 4: 44.6, 6: 30.8, 8: 15.0, 9: 7.6},
        15: {0: 45.2, 2: 42.5, 4: 37.1, 6: 26.9, 8: 13.8, 9: 7.1},
        10: {0: 32.1, 2: 30.8, 4: 28.3, 6: 22.8, 8: 12.5, 9: 6.7},
        5:  {0: 17.2, 2: 16.8, 4: 16.1, 6: 14.3, 8: 10.1, 9: 5.8},
        0:  {0: 0.0,  2: 0.0,  4: 0.0,  6: 0.0,  8: 0.0,  9: 0.0}
    }
    over_markers = sorted(resources.keys(), reverse=True)
    for i in range(len(over_markers)-1):
        upper, lower = over_markers[i], over_markers[i+1]
        if upper >= overs_left >= lower:
            w_keys = sorted(resources[upper].keys())
            w_idx = next((k for k in w_keys if k >= wickets_lost), 9)
            val_upper, val_lower = resources[upper][w_idx], resources[lower][w_idx]
            return val_lower + (val_upper - val_lower) * ((overs_left - lower) / (upper - lower))
    return 0.0

@st.cache_resource
def load_assets():
    """Loads and caches all ML models and lookup data."""
    assets = {
        'p_cat': pickle.load(open('pipe.pkl', 'rb')),
        'p_xgb': pickle.load(open('pipe2.pkl', 'rb')),
        'p_lr': pickle.load(open('pipe3.pkl', 'rb')),
        'p_lgb': pickle.load(open('pipe4.pkl', 'rb')),
        'p_rf': pickle.load(open('pipe5.pkl', 'rb')),
        'p_svr': pickle.load(open('pipe6.pkl', 'rb')),
        'teams': pickle.load(open('teams.pkl', 'rb')),
        'venues': pickle.load(open('venues.pkl', 'rb')),
        'v_lookup': pickle.load(open('venue_lookup.pkl', 'rb')),
        'bat_stats': pickle.load(open('batting_stats.pkl', 'rb')),
        'bowl_stats': pickle.load(open('bowling_stats.pkl', 'rb')),
        'p_lookup': pickle.load(open('player_lookup.pkl', 'rb')),
        'sr_lookup': pickle.load(open('sr_lookup.pkl', 'rb'))
    }
    return assets

# Initialize assets
a = load_assets()

def get_player_sr(p):
    """Retrieves player strike rate from lookup."""
    vals = [val for k, val in a['sr_lookup'].items() if k[0] == p]
    return round(vals[-1], 2) if vals else 130.0


with st.sidebar:
    st.title("🏟️ Match Setup")
    batting_team = st.selectbox('Batting Team', sorted(a['teams']), index=0)
    bowling_team = st.selectbox('Bowling Team', sorted(a['teams']), index=1)
    
    # Theme colors for dynamic styling
    color1 = TEAM_COLORS.get(batting_team, '#1E3A8A')
    color2 = TEAM_COLORS.get(bowling_team, '#EAB308')

    selected_venue = st.selectbox('Stadium Venue', sorted(a['venues']))
    
    # Venue Stats Display
    v_avg_score = int(a['v_lookup'].get(selected_venue, 160))
    st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); padding: 12px; border-radius: 10px; border: 2px solid {color1}; margin-top: 10px;">
            <small style="color: {color1}; font-weight: bold;">Venue Historical Avg Score</small><br>
            <span style="font-size: 22px; font-weight: bold; color: #ffffff;">{v_avg_score}</span>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    algorithm = st.selectbox("Select Prediction Model", 
                            ("XGBoost", "LightGBM", "CatBoost", "Random Forest", "SVR", "Linear Regression"))
    calc_btn = st.button('🔥 RUN PREDICTION', use_container_width=True)


st.markdown(f"""
    <style>
    .stApp {{ background: linear-gradient(135deg, {color1} 0%, #0F172A 50%, {color2} 100%); color: #ffffff; }}
    [data-testid="stSidebar"] {{ background-color: #0F172A; border-right: 3px solid {color1}; }}
    div[data-testid="stMetric"] {{ background: rgba(15, 23, 42, 0.7); padding: 15px; border-radius: 12px; border: 2px solid {color1}; }}
    h1 {{ text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }}
    </style>
""", unsafe_allow_html=True)

st.title("IPL Predictive Analytics Engine")
col_input, col_momentum = st.columns([2, 1])

with col_input:
    st.subheader("📊 Match Live Data (Mandatory)")
    sc1, sc2, sc3 = st.columns(3)
    score = sc1.number_input('Current Score', min_value=0, step=1)
    wickets = sc2.number_input('Wickets Down', min_value=0, max_value=9)
    overs_input = sc3.number_input('Overs Completed', min_value=0.0, max_value=19.5, value=0.0, step=0.1)
    
    st.subheader("🏏 Batsmen")
    b1, b2 = st.columns(2)
    p_list = sorted(list(a['p_lookup'].keys()))
    striker = b1.selectbox('Striker', p_list)
    non_striker = b2.selectbox('Non-Striker', p_list)

with col_momentum:
    st.subheader("📈 Recent Trend")
    l5_runs = st.number_input('Runs (Last 5 Overs)', min_value=0)
    l5_wickets = st.number_input('Wickets (Last 5 Overs)', min_value=0, max_value=5)


if calc_btn:
    # 7.1 Input Validations
    if score <= 30:
        st.error("⚠️ Current Score must be greater than 30 for an accurate prediction.")
        st.stop()
    if overs_input <= 0.0:
        st.error("⚠️ Overs Completed is a mandatory field.")
        st.stop()
    if batting_team == bowling_team:
        st.error("❌ Batting Team and Bowling Team cannot be the same.")
        st.stop()
    if striker == non_striker:
        st.error("❌ Striker and Non-Striker cannot be the same player.")
        st.stop()

    # 7.2 Calculations
    over_base = int(overs_input)
    ball_rem = int(round((overs_input - over_base) * 10))
    balls_comp = (over_base * 6) + ball_rem
    balls_left = 120 - balls_comp
    wickets_left = 10 - wickets
    crr = score / (balls_comp / 6) if balls_comp > 0 else 0
    
    # DLS Baseline
    res_pct = get_dls_resource_pct(balls_left / 6, wickets)
    dls_score = score + (165 * (res_pct / 100))

    # Aggression Bonus
    aggression_bonus = 0
    if wickets_left >= 3 and over_base >= 8:
        momentum = max(0, crr - 8.8) 
        base_bonus = (wickets_left * 1.3) + (momentum * 1.6)
        penalty = l5_wickets * 2.5
        aggression_bonus = max(0, base_bonus - penalty)

    # 7.3 Model Input Preparation
    s_role = a['p_lookup'].get(striker, 'Top Order')
    ns_role = a['p_lookup'].get(non_striker, 'Top Order')
    
    input_dict = {
        'batting_team_avg_last10': [a['bat_stats'].get(batting_team, {}).get('last10', 160)],
        'bowling_team_avg_conceded_last10': [a['bowl_stats'].get(bowling_team, {}).get('last10', 160)],
        'venue_avg_score': [np.log1p(v_avg_score)], 
        'balls_left': [balls_left], 
        'wicket_left': [wickets_left],
        'current_run_rate': [crr], 
        'last_5_over_runs': [l5_runs], 
        'last_5_over_wickets': [l5_wickets],
        'striker_sr': [get_player_sr(striker)], 
        'non_striker_sr': [get_player_sr(non_striker)],
        'finisher_death_impact': [1 if (over_base >= 15 and (s_role == 'Finisher' or ns_role == 'Finisher')) else 0]
    }
    df_11 = pd.DataFrame(input_dict)
    df_13 = df_11.copy()
    df_13['striker_role'] = [s_role]
    df_13['non_striker_role'] = [ns_role]

    # 7.4 Model Prediction
    try:
        if algorithm == "XGBoost": pred = a['p_xgb'].predict(df_13)[0] + 1
        elif algorithm == "LightGBM": pred = a['p_lgb'].predict(df_13)[0] + 2
        elif algorithm == "CatBoost": pred = a['p_cat'].predict(df_11)[0]
        elif algorithm == "Random Forest": pred = a['p_rf'].predict(df_13)[0] + 5
        elif algorithm == "SVR": pred = a['p_svr'].predict(df_11)[0] + 10
        else: pred = score + a['p_lr'].predict(df_11)[0] + 10 - aggression_bonus

        final_score = int(max(pred + aggression_bonus, dls_score + 5, score))
        
        # Display Results
        st.divider()
        st.markdown(f"<h1>Predicted Total: <span style='color:#00ff00;'>{final_score}</span></h1>", unsafe_allow_html=True)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("DLS Baseline", int(dls_score) + 5)
        m_col2.metric(f"{algorithm} Model", int(pred + aggression_bonus))
        
    except Exception as e:

        st.error(f"Prediction Error: {e}")


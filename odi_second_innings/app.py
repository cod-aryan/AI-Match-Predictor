import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="ODI Win Predictor Pro", page_icon="🏏", layout="wide")

# International Team Colors (ICC World Cup 2026 Qualified/Major Teams)
TEAM_COLORS = {
    'India': '#1D4ED8', 'Sri Lanka': '#002060', 'Afghanistan': '#0058A8',
    'Australia': '#FFCD00', 'Bangladesh': '#006A4E', 'England': '#E61B2E',
    'South Africa': '#007A33', 'USA': '#002868', 'West Indies': '#7B0031',
    'Ireland': '#169B62', 'New Zealand': '#000000', 'Pakistan': '#01411C',
    'Netherlands': '#F36C21', 'Namibia': '#003580', 'Zimbabwe': '#E4002B',
    'Nepal': '#DC143C', 'Oman': '#DA291C', 'Canada': '#FF0000', 'United Arab Emirates': '#00732F'
}

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_odi_assets():
    try:
        pipe = pickle.load(open('pipe_odi_second_innings.pkl', 'rb'))
        teams = pickle.load(open('team_odi_second_innings.pkl', 'rb'))
        cities = pickle.load(open('city_odi_second_innings.pkl', 'rb'))
        return pipe, teams, cities
    except FileNotFoundError:
        st.error("Model files not found! Ensure pipe_odi_second_innings.pkl, team_odi_second_innings.pkl, and city_odi_second_innings.pkl are present.")
        return None, None, None

pipe, teams, cities = load_odi_assets()

# --- 3. SIDEBAR SETUP (Match Setup & Target) ---
st.sidebar.title("🏟️ Match Setup")
st.sidebar.divider()
batting_team = st.sidebar.selectbox('Batting Team', sorted(teams) if teams else [])
bowling_team = st.sidebar.selectbox('Bowling Team', sorted(teams) if teams else [])
selected_city = st.sidebar.selectbox('Select City', sorted(cities) if cities else [])

st.sidebar.subheader("🎯 Target Settings")
target = st.sidebar.number_input('Target Score', min_value=1, step=1, value=250)

# Set Dynamic Colors for Alignment
color1 = TEAM_COLORS.get(batting_team, '#1E3A8A')
color2 = TEAM_COLORS.get(bowling_team, '#1E293B')

# --- 4. PERFECTLY ALIGNED DYNAMIC CSS ---
st.markdown(f"""
    <style>
    /* Main Background Alignment */
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, {color1} 0%, #0F172A 50%, {color2} 100%) !important;
        color: #ffffff;
    }}

    /* Sidebar Background Alignment - Perfectly Blended */
    [data-testid="stSidebarContent"] {{
        background: linear-gradient(180deg, {color1} 0%, {color2} 100%) !important;
        border-right: 2px solid rgba(255, 255, 255, 0.2);
    }}

    /* Sidebar Text/Labels */
    [data-testid="stSidebar"] * {{
        color: white !important;
        font-weight: bold;
    }}
    
    h1, h2, h3 {{
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 5. MAIN INTERFACE ---
st.title('🏏 ODI Win Probability Predictor')

col_main, col_chart = st.columns([1, 1])

with col_main:
    st.subheader("📝 Live Scorecard")
    sc1, sc2 = st.columns(2)
    score = sc1.number_input('Current Score', min_value=0, step=1, value=100)
    wickets = sc2.number_input('Wickets Fallen', min_value=0, max_value=9, step=1, value=3)
    
    st.divider()
    
    m1, m2 = st.columns(2)
    overs = m1.number_input('Overs Completed', min_value=0.0, max_value=49.5, step=0.1, value=25.0)
    
    st.subheader("📊 Last 10 Overs Statistics")
    l10_runs = st.number_input('Runs scored in last 10 overs', min_value=0, step=1, value=50)
    l10_wickets = st.number_input('Wickets lost in last 10 overs', min_value=0, max_value=9, step=1, value=1)
    
    predict_btn = st.button('🔥 PREDICT WIN PROBABILITY', use_container_width=True)

# --- 6. PREDICTION LOGIC & PIE CHART ---
if predict_btn:
    # Logic Validations
    if batting_team == bowling_team:
        st.error(f"❌ Selection Error: Batting and Bowling teams must be different.")
    elif score >= target:
        st.success(f"{batting_team} has already won!")
    elif l10_wickets > wickets:
        st.error(f"❌ Logic Error: Wickets in last 10 overs cannot exceed total wickets.")
    elif l10_runs > score:
        st.error(f"❌ Logic Error: Runs in last 10 overs cannot exceed total score.")
    else:
        try:
            # 1. Feature Engineering (ODI: 300 Balls)
            runs_left = target - score
            balls_bowled = (int(overs) * 6) + int((overs % 1) * 10)
            balls_remaining = 300 - balls_bowled
            wickets_left = 10 - wickets
            
            # Current and Required Run Rates
            crr = (score * 6) / balls_bowled if balls_bowled > 0 else 0
            rr = (runs_left * 6) / balls_remaining if balls_remaining > 0 else 0

            # 2. Input DataFrame
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'current_score': [score],
                'balls_remaining': [balls_remaining],
                'wickets_left': [wickets_left],
                'runs_left': [runs_left],
                'crr': [crr],
                'rr': [rr],
                'runs_last_10': [l10_runs],
                'wickets_last_10': [l10_wickets]
            })

            # 3. Predict
            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]

            with col_chart:
                st.subheader("📊 Win Probability")
                
                # Plotly Pie Chart (Donut Style)
                fig = go.Figure(data=[go.Pie(
                    labels=[batting_team, bowling_team],
                    values=[win, loss],
                    hole=.5,
                    marker_colors=['#00ff00', '#ff4b4b'],
                    textinfo='percent+label'
                )])
                
                fig.update_layout(
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white"),
                    margin=dict(t=30, b=0, l=0, r=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"### {batting_team}: {round(win * 100)}%")
                st.markdown(f"### {bowling_team}: {round(loss * 100)}%")

        except Exception as e:
            st.error(f"⚠️ Internal Prediction Error: {e}")
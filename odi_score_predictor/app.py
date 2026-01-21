import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. PERFORMANCE METRICS (Recalibrated for ODI) ---
model_performance = {
    "XGBoost": {"rmse": 14.52, "file": 'xgb_model_odi_score.pkl'},
    "LightGBM": {"rmse": 13.44, "file": 'lgbm_model_odi_score.pkl'},
    "CatBoost": {"rmse": 14.78, "file": 'cat_model_odi_score.pkl'},
    "Random Forest": {"rmse": 22.13, "file": 'rf_model_odi_score.pkl'},
    "Linear Regression": {"rmse": 28.71, "file": 'lr_model_odi_score.pkl'}
}

# Major ODI Nations Colors
TEAM_COLORS = {
    'India': '#1D4ED8', 'Australia': '#FFCD00', 'Pakistan': '#01411C',
    'England': '#E61B2E', 'South Africa': '#007A33', 'New Zealand': '#000000',
    'West Indies': '#7B0031', 'Sri Lanka': '#002060', 'Afghanistan': '#0058A8',
    'Bangladesh': '#006A4E'
}

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_all_assets():
    # Use your ODI specific pickle files here
    teams = pickle.load(open('teams_odi_score.pkl', 'rb'))
    cities = pickle.load(open('cities_odi_score.pkl', 'rb'))
    models = {name: pickle.load(open(info['file'], 'rb')) for name, info in model_performance.items()}
    return teams, cities, models

teams, cities, loaded_models = load_all_assets()

# --- 3. SIDEBAR NAVIGATION & MATCH SETUP ---
st.sidebar.title("🏏 ODI World Engine")
st.sidebar.divider()
st.sidebar.header("Match Setup")
batting_team = st.sidebar.selectbox('Batting Team', sorted(teams))
bowling_team = st.sidebar.selectbox('Bowling Team', sorted(teams))
city = st.sidebar.selectbox('Venue City', sorted(cities))
algorithm = st.sidebar.selectbox("Prediction Algorithm", list(model_performance.keys()))

color1 = TEAM_COLORS.get(batting_team, '#1E3A8A')
color2 = TEAM_COLORS.get(bowling_team, '#1E293B')

# --- 4. PERFECTLY ALIGNED DYNAMIC CSS ---
st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, {color1} 0%, #0F172A 50%, {color2} 100%) !important;
        color: #ffffff;
    }}
    [data-testid="stSidebarContent"] {{
        background: linear-gradient(180deg, {color1} 0%, {color2} 100%) !important;
        border-right: 2px solid rgba(255, 255, 255, 0.2);
    }}
    [data-testid="stSidebar"] * {{ color: white !important; font-weight: bold; }}
    div[data-testid="stMetric"] {{ 
        background: rgba(255, 255, 255, 0.1); 
        border: 1px solid rgba(255, 255, 255, 0.2); 
        border-radius: 10px; 
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 5. MAIN INTERFACE ---
st.title(f"🏆 ODI Score Predictor")

col_input, col_trend = st.columns([2, 1])

with col_input:
    st.subheader("📝 Live Match Data")
    sc1, sc2, sc3 = st.columns(3)
    score = sc1.number_input('Current Score', min_value=0, step=1, value=120)
    wickets = sc2.number_input('Total Wickets Down', min_value=0, max_value=9, step=1, value=3)
    overs = sc3.number_input('Overs Completed', min_value=0.0, max_value=49.5, step=0.1, value=25.0)

with col_trend:
    st.subheader("📈 Recent Momentum")
    l10_runs = st.number_input('Runs in Last 10 Overs', min_value=0, step=1, value=55)
    l10_wickets = st.number_input('Wickets in Last 10 Overs', min_value=0, max_value=9, step=1, value=1)

# --- 6. PREDICTION LOGIC ---
if st.button('🔥 RUN PREDICTION'):
    # Validation
    if batting_team == bowling_team:
        st.error(f"❌ Selection Error: Teams must be different.")
    elif overs < 5.0:
        st.error("❌ Data Error: Prediction requires at least 5 overs of play.")
    elif l10_wickets > wickets or l10_runs > score:
        st.error("❌ Logic Error: Recent stats cannot exceed total match stats.")
    else:
        try:
            # 1. ODI Feature Engineering (300 Balls)
            balls_left = 300 - (int(overs) * 6 + int((overs % 1) * 10))
            wickets_left = 10 - wickets
            balls_bowled = 300 - balls_left
            crr = score / (balls_bowled / 6)
            
            # 2. ODI Aggression Bonus Calculation
            # In ODIs, acceleration depends on wickets left after the 35th or 40th over.
            agg_bonus = 0
            if overs >= 25 and wickets_left > 4:
                # Bonus multiplier based on wickets in hand
                wicket_factor = (wickets_left ** 1.5) / 5
                # Acceleration factor based on remaining balls
                phase_factor = (balls_left / 300) * 5
                # Momentum check
                momentum_factor =  (l10_runs / 10) - 5.5
                
                
                agg_bonus = (wicket_factor * 1.3) + (momentum_factor * 2.6) + (phase_factor * 2.5) - (l10_wickets * 7)

            # 3. Create DataFrame
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'current_score': [score],
                'balls_left': [balls_left],
                'wickets_left': [wickets_left],
                'crr': [crr],
                'last_ten_runs': [l10_runs], # Note: Model likely trained on last 10 for ODI
                'last_ten_wickets': [l10_wickets]
            })

            # 4. Predict
            pipe = loaded_models[algorithm]
            raw_pred = pipe.predict(input_df)[0]
            
            # 5. Final Result
            final_score = int(max(raw_pred + max(0, agg_bonus), score))
            rmse = model_performance[algorithm]['rmse']

            # 6. Display Output
            st.divider()
            st.markdown(f"<h1 style='text-align: center; color: #00ff00;'>Predicted ODI Total: {final_score}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Range (±RMSE): {final_score-int(rmse)} — {final_score+int(rmse)}</h3>", unsafe_allow_html=True)
            
            # Contextual Insight 
            st.info(f"Insight: Predicted run rate for remaining {balls_left} balls is approx {((final_score-score)/(balls_left/6)):.2f} rpo.")

        except Exception as e:

            st.error(f"⚠️ Internal Prediction Error: {e}")


from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# ============== FIX FOR SKLEARN 1.5.2 COMPATIBILITY ==============
def fix_sklearn_model(model):
    """Fix compatibility issues for sklearn 1.5.2"""
    try:
        # Handle pipeline objects
        if hasattr(model, 'steps'):
            for name, step in model.steps:
                fix_sklearn_model(step)
        
        # Handle LogisticRegression - ensure multi_class is set properly for 1.5.2
        if hasattr(model, '__class__') and model.__class__.__name__ == 'LogisticRegression':
            # In sklearn 1.5.2, multi_class should be 'auto', 'ovr', or 'multinomial'
            if not hasattr(model, 'multi_class') or model.multi_class is None:
                model.multi_class = 'auto'
        
        # Also check for the classifier inside a pipeline
        if hasattr(model, 'named_steps'):
            for step_name, step in model.named_steps.items():
                fix_sklearn_model(step)
        
        # Handle ColumnTransformer
        if hasattr(model, 'transformers_'):
            for name, transformer, columns in model.transformers_:
                fix_sklearn_model(transformer)
        if hasattr(model, 'transformers'):
            for item in model.transformers:
                if len(item) >= 2:
                    fix_sklearn_model(item[1])
                    
        # Handle any nested estimators
        if hasattr(model, 'estimator'):
            fix_sklearn_model(model.estimator)
        if hasattr(model, 'estimators_'):
            for est in model.estimators_:
                fix_sklearn_model(est)
                
    except Exception as e:
        print(f"Warning in fix_sklearn_model: {e}")
    return model

# ============== MODEL-SPECIFIC TEAMS ==============
# IPL Teams (for IPL Score & IPL Win)
IPL_TEAMS = [
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bengaluru',
    'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
    'Rajasthan Royals', 'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants'
]

# ODI Score - Major ODI Nations (10 teams)
ODI_SCORE_TEAMS = [
    'India', 'Australia', 'England', 'Pakistan', 'South Africa', 'New Zealand',
    'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan'
]

# ODI Win - ICC World Cup Qualified Teams (19 teams)
ODI_WIN_TEAMS = [
    'India', 'Sri Lanka', 'Afghanistan', 'Australia', 'Bangladesh', 'England',
    'South Africa', 'USA', 'West Indies', 'Ireland', 'New Zealand', 'Pakistan',
    'Netherlands', 'Namibia', 'Zimbabwe', 'Nepal', 'Oman', 'Canada', 'United Arab Emirates'
]

# T20 Score & Win - 2026 T20 World Cup Teams (20 teams)
T20_TEAMS = [
    'India', 'Sri Lanka', 'Afghanistan', 'Australia', 'Bangladesh', 'England',
    'South Africa', 'USA', 'West Indies', 'Ireland', 'New Zealand', 'Pakistan',
    'Netherlands', 'Italy', 'Namibia', 'Zimbabwe', 'Nepal', 'Oman', 'Canada', 
    'United Arab Emirates'
]

# ============== TEAM COLORS ==============
IPL_TEAM_COLORS = {
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

# ODI Score - Major ODI Nations Colors
ODI_SCORE_TEAM_COLORS = {
    'India': '#1D4ED8', 'Australia': '#FFCD00', 'Pakistan': '#01411C',
    'England': '#E61B2E', 'South Africa': '#007A33', 'New Zealand': '#000000',
    'West Indies': '#7B0031', 'Sri Lanka': '#002060', 'Afghanistan': '#0058A8',
    'Bangladesh': '#006A4E'
}

# ODI Win - Extended Teams Colors
ODI_WIN_TEAM_COLORS = {
    'India': '#1D4ED8', 'Sri Lanka': '#002060', 'Afghanistan': '#0058A8',
    'Australia': '#FFCD00', 'Bangladesh': '#006A4E', 'England': '#E61B2E',
    'South Africa': '#007A33', 'USA': '#002868', 'West Indies': '#7B0031',
    'Ireland': '#169B62', 'New Zealand': '#000000', 'Pakistan': '#01411C',
    'Netherlands': '#F36C21', 'Namibia': '#003580', 'Zimbabwe': '#E4002B',
    'Nepal': '#DC143C', 'Oman': '#DA291C', 'Canada': '#FF0000', 'United Arab Emirates': '#00732F'
}

# T20 Teams Colors (Score & Win)
T20_TEAM_COLORS = {
    'India': '#1D4ED8', 'Sri Lanka': '#002060', 'Afghanistan': '#0058A8',
    'Australia': '#FFCD00', 'Bangladesh': '#006A4E', 'England': '#E61B2E',
    'South Africa': '#007A33', 'USA': '#002868', 'West Indies': '#7B0031',
    'Ireland': '#169B62', 'New Zealand': '#000000', 'Pakistan': '#01411C',
    'Netherlands': '#F36C21', 'Italy': '#008C45', 'Namibia': '#003580',
    'Zimbabwe': '#E4002B', 'Nepal': '#DC143C', 'Oman': '#DA291C',
    'Canada': '#FF0000', 'United Arab Emirates': '#00732F'
}

# ============== ASSET LOADING FUNCTIONS ==============

def load_ipl_score_assets():
    """Load IPL Score Predictor assets"""
    base_path = 'ipl_score_prdictor/'
    try:
        assets = {
            'p_cat': pickle.load(open(base_path + 'pipe.pkl', 'rb')),
            'p_xgb': pickle.load(open(base_path + 'pipe2.pkl', 'rb')),
            'p_lr': pickle.load(open(base_path + 'pipe3.pkl', 'rb')),
            'p_lgb': pickle.load(open(base_path + 'pipe4.pkl', 'rb')),
            'p_rf': pickle.load(open(base_path + 'pipe5.pkl', 'rb')),
            'p_svr': pickle.load(open(base_path + 'pipe6.pkl', 'rb')),
            'teams': pickle.load(open(base_path + 'teams.pkl', 'rb')),
            'venues': pickle.load(open(base_path + 'venues.pkl', 'rb')),
            'v_lookup': pickle.load(open(base_path + 'venue_lookup.pkl', 'rb')),
            'bat_stats': pickle.load(open(base_path + 'batting_stats.pkl', 'rb')),
            'bowl_stats': pickle.load(open(base_path + 'bowling_stats.pkl', 'rb')),
            'p_lookup': pickle.load(open(base_path + 'player_lookup.pkl', 'rb')),
            'sr_lookup': pickle.load(open(base_path + 'sr_lookup.pkl', 'rb'))
        }
        return assets
    except Exception as e:
        print(f"Error loading IPL score assets: {e}")
        return None

def load_ipl_win_assets():
    """Load IPL Win Predictor assets"""
    base_path = 'ipl_win_predictor_second_innings/'
    try:
        pipe = pickle.load(open(base_path + 'pipe_ipl_2nd_innings.pkl', 'rb'))
        fix_sklearn_model(pipe)
        assets = {
            'pipe': pipe,
            'teams': pickle.load(open(base_path + 'teams_ipl_2nd_innings.pkl', 'rb')),
            'venues': pickle.load(open(base_path + 'venues_ipl_2nd_innings.pkl', 'rb')),
            'players': pickle.load(open(base_path + 'players_ipl_2nd_innings.pkl', 'rb')),
            'sr_map': pickle.load(open(base_path + 'sr_lookup_ipl_2nd_innings.pkl', 'rb')),
            'role_map': pickle.load(open(base_path + 'role_lookup.pkl', 'rb')),
            'venue_chase_map': pickle.load(open(base_path + 'venue_chase_ipl_2nd_innings.pkl', 'rb')),
            'team_win_pct_map': pickle.load(open(base_path + 'team_win_pct_map.pkl', 'rb'))
        }
        return assets
    except Exception as e:
        print(f"Error loading IPL win assets: {e}")
        return None

def load_odi_score_assets():
    """Load ODI Score Predictor assets"""
    base_path = 'odi_score_predictor/'
    model_files = {
        "XGBoost": 'xgb_model_odi_score.pkl',
        "LightGBM": 'lgbm_model_odi_score.pkl',
        "CatBoost": 'cat_model_odi_score.pkl',
        "Random Forest": 'rf_model_odi_score.pkl',
        "Gradient Boosting": 'gb_model_odi_score.pkl',
        "Linear Regression": 'lr_model_odi_score.pkl'
    }
    try:
        assets = {
            'teams': pickle.load(open(base_path + 'teams_odi_score.pkl', 'rb')),
            'cities': pickle.load(open(base_path + 'cities_odi_score.pkl', 'rb')),
            'models': {name: pickle.load(open(base_path + fname, 'rb')) for name, fname in model_files.items()}
        }
        return assets
    except Exception as e:
        print(f"Error loading ODI score assets: {e}")
        return None

def load_odi_win_assets():
    """Load ODI Win Predictor assets"""
    base_path = 'odi_second_innings/'
    try:
        pipe = pickle.load(open(base_path + 'pipe_odi_second_innings.pkl', 'rb'))
        fix_sklearn_model(pipe)
        assets = {
            'pipe': pipe,
            'teams': pickle.load(open(base_path + 'team_odi_second_innings.pkl', 'rb')),
            'cities': pickle.load(open(base_path + 'city_odi_second_innings.pkl', 'rb'))
        }
        return assets
    except Exception as e:
        print(f"Error loading ODI win assets: {e}")
        return None

def load_t20_score_assets():
    """Load T20 Score Predictor assets"""
    base_path = 't20_score_predictor/'
    model_files = {
        "XGBoost": 'xgb_model_t20_score.pkl',
        "LightGBM": 'lgbm_model_t20_score.pkl',
        "CatBoost": 'cat_model_t20_score.pkl',
        "Random Forest": 'rf_model_t20_score.pkl',
        "Gradient Boosting": 'gb_model_t20_score.pkl',
        "Linear Regression": 'lr_model_t20_score.pkl'
    }
    try:
        assets = {
            'teams': pickle.load(open(base_path + 'teams_t20_score.pkl', 'rb')),
            'cities': pickle.load(open(base_path + 'cities_t20_score.pkl', 'rb')),
            'models': {name: pickle.load(open(base_path + fname, 'rb')) for name, fname in model_files.items()}
        }
        return assets
    except Exception as e:
        print(f"Error loading T20 score assets: {e}")
        return None

def load_t20_win_assets():
    """Load T20 Win Predictor assets"""
    base_path = 't20_second_inning/'
    try:
        pipe = pickle.load(open(base_path + 'pipe_t20_second_inning.pkl', 'rb'))
        fix_sklearn_model(pipe)
        assets = {
            'pipe': pipe,
            'teams': pickle.load(open(base_path + 'team_t20_second_inning.pkl', 'rb')),
            'cities': pickle.load(open(base_path + 'city_t20_second_inning.pkl', 'rb'))
        }
        return assets
    except Exception as e:
        print(f"Error loading T20 win assets: {e}")
        return None

# ============== HELPER FUNCTIONS ==============

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

# ============== ROUTES ==============

@app.route('/')
def home():
    return render_template('index.html')

# ---------- IPL SCORE PREDICTOR ----------
@app.route('/ipl-score')
def ipl_score():
    assets = load_ipl_score_assets()
    if assets is None:
        return render_template('error.html', message="Failed to load IPL Score Predictor assets")
    
    # Use IPL-specific teams from pickle or fallback to defined list
    asset_teams = set(assets['teams']) if assets['teams'] else set()
    all_teams = sorted(set(IPL_TEAMS) | asset_teams)
    venues = sorted(assets['venues'])
    players = sorted(list(assets['p_lookup'].keys()))
    algorithms = ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "SVR", "Linear Regression"]
    
    return render_template('ipl_score.html', 
                         teams=all_teams, 
                         venues=venues, 
                         players=players,
                         algorithms=algorithms,
                         team_colors=IPL_TEAM_COLORS)

@app.route('/ipl-score/predict', methods=['POST'])
def ipl_score_predict():
    try:
        assets = load_ipl_score_assets()
        data = request.json
        
        batting_team = data['batting_team']
        bowling_team = data['bowling_team']
        venue = data['venue']
        score = int(data['score'])
        wickets = int(data['wickets'])
        overs = float(data['overs'])
        striker = data['striker']
        non_striker = data['non_striker']
        l5_runs = int(data['l5_runs'])
        l5_wickets = int(data['l5_wickets'])
        algorithm = data['algorithm']
        
        # Validations
        if score <= 30:
            return jsonify({'error': 'Current Score must be greater than 30 for an accurate prediction.'})
        if overs <= 0.0:
            return jsonify({'error': 'Overs Completed is a mandatory field.'})
        if batting_team == bowling_team:
            return jsonify({'error': 'Batting Team and Bowling Team cannot be the same.'})
        if striker == non_striker:
            return jsonify({'error': 'Striker and Non-Striker cannot be the same player.'})
        
        # Calculations
        over_base = int(overs)
        ball_rem = int(round((overs - over_base) * 10))
        balls_comp = (over_base * 6) + ball_rem
        balls_left = 120 - balls_comp
        wickets_left = 10 - wickets
        crr = score / (balls_comp / 6) if balls_comp > 0 else 0
        
        # DLS Baseline
        v_avg_score = int(assets['v_lookup'].get(venue, 160))
        res_pct = get_dls_resource_pct(balls_left / 6, wickets)
        dls_score = score + (165 * (res_pct / 100))
        
        # Aggression Bonus
        aggression_bonus = 0
        if wickets_left >= 3 and over_base >= 8:
            momentum = max(0, crr - 8.8)
            base_bonus = (wickets_left * 1.3) + (momentum * 1.6)
            penalty = l5_wickets * 2.5
            aggression_bonus = max(0, base_bonus - penalty)
        
        # Get player SR
        def get_player_sr(p):
            vals = [val for k, val in assets['sr_lookup'].items() if k[0] == p]
            return round(vals[-1], 2) if vals else 130.0
        
        s_role = assets['p_lookup'].get(striker, 'Top Order')
        ns_role = assets['p_lookup'].get(non_striker, 'Top Order')
        
        input_dict = {
            'batting_team_avg_last10': [assets['bat_stats'].get(batting_team, {}).get('last10', 160)],
            'bowling_team_avg_conceded_last10': [assets['bowl_stats'].get(bowling_team, {}).get('last10', 160)],
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
        
        if algorithm == "XGBoost":
            pred = assets['p_xgb'].predict(df_13)[0] + 1
        elif algorithm == "LightGBM":
            pred = assets['p_lgb'].predict(df_13)[0] + 2
        elif algorithm == "CatBoost":
            pred = assets['p_cat'].predict(df_11)[0]
        elif algorithm == "Random Forest":
            pred = assets['p_rf'].predict(df_13)[0] + 5
        elif algorithm == "SVR":
            pred = assets['p_svr'].predict(df_11)[0] + 10
        else:  # Linear Regression
            pred = score + assets['p_lr'].predict(df_11)[0] + 10 - aggression_bonus
        
        final_score = int(max(pred + aggression_bonus, dls_score + 5, score))
        
        return jsonify({
            'success': True,
            'final_score': final_score,
            'dls_baseline': int(dls_score) + 5,
            'model_prediction': int(pred + aggression_bonus),
            'algorithm': algorithm
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# ---------- IPL WIN PREDICTOR ----------
@app.route('/ipl-win')
def ipl_win():
    assets = load_ipl_win_assets()
    if assets is None:
        return render_template('error.html', message="Failed to load IPL Win Predictor assets")
    
    # Use IPL-specific teams from pickle or fallback to defined list
    asset_teams = set(assets['teams']) if assets['teams'] else set()
    all_teams = sorted(set(IPL_TEAMS) | asset_teams)
    venues = sorted(assets['venues'])
    players = sorted(assets['players'])
    
    return render_template('ipl_win.html',
                         teams=all_teams,
                         venues=venues,
                         players=players,
                         team_colors=IPL_TEAM_COLORS)

@app.route('/ipl-win/predict', methods=['POST'])
def ipl_win_predict():
    try:
        assets = load_ipl_win_assets()
        data = request.json
        
        batting_team = data['batting_team']
        bowling_team = data['bowling_team']
        venue = data['venue']
        target = int(data['target'])
        score = int(data['score'])
        wickets = int(data['wickets'])
        overs = float(data['overs'])
        striker = data['striker']
        non_striker = data['non_striker']
        l5_runs = int(data['l5_runs'])
        l5_wickets = int(data['l5_wickets'])
        
        if batting_team == bowling_team:
            return jsonify({'error': 'Select different teams.'})
        if overs < 7.0:
            return jsonify({'error': 'Prediction requires at least 7 overs of play.'})
        
        runs_left = target - score
        balls_left = 120 - (int(overs) * 6 + int((overs % 1) * 10))
        ball_per_run = balls_left / (runs_left + 1)
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
        
        pressure_index = min(rrr / (crr + 0.1), 10.0)
        bat_win_pct = assets['team_win_pct_map'].get(batting_team, 50.0)
        bowl_win_pct = assets['team_win_pct_map'].get(bowling_team, 50.0)
        v_chase_pct = assets['venue_chase_map'].get(venue, 50.0)
        
        s_name = striker[1] if isinstance(striker, (tuple, list)) else striker
        ns_name = non_striker[1] if isinstance(non_striker, (tuple, list)) else non_striker
        s_sr = round(assets['sr_map'].get(s_name, 120.0), 2)
        ns_sr = round(assets['sr_map'].get(ns_name, 120.0), 2)
        s_role = assets['role_map'].get(s_name, "Middle Order")
        ns_role = assets['role_map'].get(ns_name, "Middle Order")
        
        input_df = pd.DataFrame({
            'batting_team_win_pct': [bat_win_pct],
            'bowling_team_win_pct': [bowl_win_pct],
            'venue_chase_pct': [v_chase_pct],
            'balls_per_run': [ball_per_run],
            'wickets_remaining': [10 - wickets],
            'current_run_rate': [crr],
            'required_run_rate': [rrr],
            'last_5_over_runs': [l5_runs],
            'last_5_over_wickets': [l5_wickets],
            'pp_wicket_count': [0],
            'striker_sr': [s_sr],
            'non_striker_sr': [ns_sr],
            'partnership_power': [0.8],
            'striker_role': [s_role],
            'non_striker_role': [ns_role],
            'pressure_index': [pressure_index]
        })
        
        res = assets['pipe'].predict_proba(input_df)
        win, loss = res[0][1], res[0][0]
        
        return jsonify({
            'success': True,
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'win_probability': round(win * 100, 1),
            'loss_probability': round(loss * 100, 1),
            'runs_left': int(runs_left),
            'balls_left': int(balls_left),
            'required_rr': round(rrr, 2),
            'pressure_index': round(pressure_index, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# ---------- ODI SCORE PREDICTOR ----------
@app.route('/odi-score')
def odi_score():
    assets = load_odi_score_assets()
    if assets is None:
        return render_template('error.html', message="Failed to load ODI Score Predictor assets")
    
    # Use ODI Score specific teams from pickle or fallback to defined list
    asset_teams = set(assets['teams']) if assets['teams'] else set()
    all_teams = sorted(set(ODI_SCORE_TEAMS) | asset_teams)
    cities = sorted(assets['cities'])
    algorithms = list(assets['models'].keys())
    
    return render_template('odi_score.html',
                         teams=all_teams,
                         cities=cities,
                         algorithms=algorithms,
                         team_colors=ODI_SCORE_TEAM_COLORS)

@app.route('/odi-score/predict', methods=['POST'])
def odi_score_predict():
    try:
        assets = load_odi_score_assets()
        data = request.json
        
        batting_team = data['batting_team']
        bowling_team = data['bowling_team']
        city = data['city']
        score = int(data['score'])
        wickets = int(data['wickets'])
        overs = float(data['overs'])
        l10_runs = int(data['l10_runs'])
        l10_wickets = int(data['l10_wickets'])
        algorithm = data['algorithm']
        
        model_rmse = {
            "XGBoost": 14.52, "LightGBM": 13.44, "CatBoost": 14.78,
            "Random Forest": 22.13, "Gradient Boosting": 16.35, "Linear Regression": 28.71
        }
        
        if batting_team == bowling_team:
            return jsonify({'error': 'Teams must be different.'})
        if overs < 5.0:
            return jsonify({'error': 'Prediction requires at least 5 overs of play.'})
        if l10_wickets > wickets or l10_runs > score:
            return jsonify({'error': 'Recent stats cannot exceed total match stats.'})
        
        balls_left = 300 - (int(overs) * 6 + int((overs % 1) * 10))
        wickets_left = 10 - wickets
        balls_bowled = 300 - balls_left
        crr = score / (balls_bowled / 6)
        
        agg_bonus = 0
        if overs >= 25 and wickets_left > 4:
            # Bonus multiplier based on wickets in hand
            wicket_factor = (wickets_left ** 1.5) / 5
            # Acceleration factor based on remaining balls
            phase_factor = (balls_left / 300) * 5
            # Momentum check
            momentum_factor = (l10_runs / 10) - 5.5
            agg_bonus = (wicket_factor * 1.5) + (momentum_factor * 3.4) + (phase_factor * 2.7) - (l10_wickets * 7)
        
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'current_score': [score],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'crr': [crr],
            'last_ten_runs': [l10_runs],
            'last_ten_wickets': [l10_wickets]
        })
        
        pipe = assets['models'][algorithm]
        raw_pred = pipe.predict(input_df)[0]
        final_score = int(max(raw_pred + max(0, agg_bonus), score))
        rmse = model_rmse[algorithm]
        
        return jsonify({
            'success': True,
            'final_score': final_score,
            'range_low': final_score - int(rmse),
            'range_high': final_score + int(rmse),
            'algorithm': algorithm,
            'predicted_rpo': round((final_score - score) / (balls_left / 6), 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# ---------- ODI WIN PREDICTOR ----------
@app.route('/odi-win')
def odi_win():
    assets = load_odi_win_assets()
    if assets is None:
        return render_template('error.html', message="Failed to load ODI Win Predictor assets")
    
    # Use ODI Win specific teams from pickle or fallback to defined list
    asset_teams = set(assets['teams']) if assets['teams'] else set()
    all_teams = sorted(set(ODI_WIN_TEAMS) | asset_teams)
    cities = sorted(assets['cities'])
    
    return render_template('odi_win.html',
                         teams=all_teams,
                         cities=cities,
                         team_colors=ODI_WIN_TEAM_COLORS)

@app.route('/odi-win/predict', methods=['POST'])
def odi_win_predict():
    try:
        assets = load_odi_win_assets()
        data = request.json
        
        batting_team = data['batting_team']
        bowling_team = data['bowling_team']
        city = data['city']
        target = int(data['target'])
        score = int(data['score'])
        wickets = int(data['wickets'])
        overs = float(data['overs'])
        l10_runs = int(data['l10_runs'])
        l10_wickets = int(data['l10_wickets'])
        
        if batting_team == bowling_team:
            return jsonify({'error': 'Batting and Bowling teams must be different.'})
        if score >= target:
            return jsonify({'success': True, 'win_probability': 100, 'loss_probability': 0, 
                          'message': f'{batting_team} has already won!'})
        if l10_wickets > wickets:
            return jsonify({'error': 'Wickets in last 10 overs cannot exceed total wickets.'})
        if l10_runs > score:
            return jsonify({'error': 'Runs in last 10 overs cannot exceed total score.'})
        
        runs_left = target - score
        balls_bowled = (int(overs) * 6) + int((overs % 1) * 10)
        balls_remaining = 300 - balls_bowled
        wickets_left = 10 - wickets
        
        crr = (score * 6) / balls_bowled if balls_bowled > 0 else 0
        rr = (runs_left * 6) / balls_remaining if balls_remaining > 0 else 0
        
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
        
        result = assets['pipe'].predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        
        return jsonify({
            'success': True,
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'win_probability': round(win * 100, 1),
            'loss_probability': round(loss * 100, 1),
            'runs_left': int(runs_left),
            'balls_remaining': int(balls_remaining)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# ---------- T20 SCORE PREDICTOR ----------
@app.route('/t20-score')
def t20_score():
    assets = load_t20_score_assets()
    if assets is None:
        return render_template('error.html', message="Failed to load T20 Score Predictor assets")
    
    # Use T20-specific teams from pickle or fallback to defined list
    asset_teams = set(assets['teams']) if assets['teams'] else set()
    all_teams = sorted(set(T20_TEAMS) | asset_teams)
    cities = sorted(assets['cities'])
    algorithms = list(assets['models'].keys())
    
    return render_template('t20_score.html',
                         teams=all_teams,
                         cities=cities,
                         algorithms=algorithms,
                         team_colors=T20_TEAM_COLORS)

@app.route('/t20-score/predict', methods=['POST'])
def t20_score_predict():
    try:
        assets = load_t20_score_assets()
        data = request.json
        
        batting_team = data['batting_team']
        bowling_team = data['bowling_team']
        city = data['city']
        score = int(data['score'])
        wickets = int(data['wickets'])
        overs = float(data['overs'])
        l5_runs = int(data['l5_runs'])
        l5_wickets = int(data['l5_wickets'])
        algorithm = data['algorithm']
        
        model_rmse = {
            "XGBoost": 8.52, "LightGBM": 8.44, "CatBoost": 8.78,
            "Random Forest": 16.13, "Gradient Boosting": 12.28, "Linear Regression": 22.71
        }
        
        if batting_team == bowling_team:
            return jsonify({'error': 'Batting and Bowling teams must be different.'})
        if overs <= 5.0:
            return jsonify({'error': 'Please enter more than 5.0 overs for an accurate prediction.'})
        if l5_wickets > wickets:
            return jsonify({'error': f'Wickets in last 5 overs ({l5_wickets}) cannot be greater than total wickets ({wickets}).'})
        if l5_runs > score:
            return jsonify({'error': f'Runs in last 5 overs ({l5_runs}) cannot exceed total score ({score}).'})
        
        balls_left = 120 - (int(overs) * 6 + int((overs % 1) * 10))
        wickets_left = 10 - wickets
        balls_bowled = 120 - balls_left
        crr = score / (balls_bowled / 6)
        
        agg_bonus = 0
        if wickets_left >= 3 and overs >= 7:
            agg_bonus = (wickets_left * 1.4) + (crr - 8.9) * 1.70 + (balls_left * 0.05) - (l5_wickets * 2.0) + (l5_runs * 0.2)
        
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'current_score': [score],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'crr': [crr],
            'last_five_runs': [l5_runs],
            'last_five_wickets': [l5_wickets]
        })
        
        pipe = assets['models'][algorithm]
        raw_pred = pipe.predict(input_df)[0]
        final_score = int(max(raw_pred + max(0, agg_bonus), score))
        rmse = model_rmse[algorithm]
        
        return jsonify({
            'success': True,
            'final_score': final_score,
            'range_low': final_score - int(rmse),
            'range_high': final_score + int(rmse),
            'algorithm': algorithm
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# ---------- T20 WIN PREDICTOR ----------
@app.route('/t20-win')
def t20_win():
    assets = load_t20_win_assets()
    if assets is None:
        return render_template('error.html', message="Failed to load T20 Win Predictor assets")
    
    # Use T20-specific teams from pickle or fallback to defined list
    asset_teams = set(assets['teams']) if assets['teams'] else set()
    all_teams = sorted(set(T20_TEAMS) | asset_teams)
    cities = sorted(assets['cities'])
    
    return render_template('t20_win.html',
                         teams=all_teams,
                         cities=cities,
                         team_colors=T20_TEAM_COLORS)

@app.route('/t20-win/predict', methods=['POST'])
def t20_win_predict():
    try:
        assets = load_t20_win_assets()
        data = request.json
        
        batting_team = data['batting_team']
        bowling_team = data['bowling_team']
        city = data['city']
        target = int(data['target'])
        score = int(data['score'])
        wickets = int(data['wickets'])
        overs = float(data['overs'])
        l5_runs = int(data['l5_runs'])
        l5_wickets = int(data['l5_wickets'])
        
        if batting_team == bowling_team:
            return jsonify({'error': 'Batting and Bowling teams must be different.'})
        if score >= target:
            return jsonify({'success': True, 'win_probability': 100, 'loss_probability': 0,
                          'message': f'{batting_team} has already won!'})
        if l5_wickets > wickets:
            return jsonify({'error': 'Wickets in last 5 overs cannot exceed total wickets.'})
        if l5_runs > score:
            return jsonify({'error': 'Runs in last 5 overs cannot exceed total score.'})
        
        runs_left = target - score
        balls_left = 120 - (int(overs) * 6 + int((overs % 1) * 10))
        wickets_left = 10 - wickets
        balls_bowled = 120 - balls_left
        crr = (score * 6) / balls_bowled if balls_bowled > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
        
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'current_score': [score],
            'balls_remaining': [balls_left],
            'wickets_left': [wickets_left],
            'runs_left': [runs_left],
            'crr': [crr],
            'rr': [rrr],
            'runs_last_5': [l5_runs],
            'wickets_last_5': [l5_wickets]
        })
        
        result = assets['pipe'].predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        
        return jsonify({
            'success': True,
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'win_probability': round(win * 100, 1),
            'loss_probability': round(loss * 100, 1),
            'runs_left': int(runs_left),
            'balls_left': int(balls_left),
            'required_rr': round(rrr, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)

import fixture_quality as fq
import pandas as pd


import statsmodels.formula.api as smf
import numpy as np


#config
gameweek = 10
GAMES_TO_CHECK = 5
form_weight = 0.2


threat_weight = 0.6
creativity_weight = 0.4

team_val = 100





####################    pull fixture data       ##########################


# Fixtures to df
df_fixtures = pd.read_csv("epl_fixtures_2025.csv")


# ---  fixture related setup ---
df_table = fq.prem_table()
df_unplayed = df_fixtures[df_fixtures['HomeTeamScore'].isna()].copy() #take unplayed fixture from all fixtures table


# --- Call tables ---
df_attack = fq.calculate_attack_score(df_table, df_unplayed, num_games=GAMES_TO_CHECK)
df_defense = fq.calculate_defense_score(df_table, df_unplayed, num_games=GAMES_TO_CHECK)












####################    pull player data       ##########################


df_player = pd.read_csv('player_statistics.csv')


#####################    print fixture quality tables      ##########################




#print the attaxck and defense tables too for reference
print("")
print("---- Fixture Quality Tables ----")   

print("---- Attack Table ----")

print(df_attack.sort_values(by='Attack multiplier'))

print("")
print("---- Defense Table ----")
print(df_defense.sort_values(by='Defense multiplier'))









##################  aggregate data sets  #########################



# Isolate the multipliers from the attack and defense tables
df_attack_multi = df_attack[['HomeTeam', 'Attack multiplier']]
df_defense_multi = df_defense[['HomeTeam', 'Defense multiplier']]

# --- Merge multipliers into the main player table ---

# 1. Merge Attack Multiplier
# This joins df_player with df_attack_multi where player 'team_name' matches the 'HomeTeam'
df_merged = pd.merge(
    df_player,
    df_attack_multi,
    left_on='team_name',
    right_on='HomeTeam',
    how='left'  # Use 'left' to keep all players even if a team match isn't found
)

# 2. Merge Defense Multiplier
# This joins the *newly merged* table with the defense multipliers
df_merged = pd.merge(
    df_merged,
    df_defense_multi,
    left_on='team_name',
    right_on='HomeTeam',  # Use the same 'HomeTeam' key
    how='left'
)




# --- Create the final table ---
# Select only the columns we discussed, plus the new multipliers
df_players_final = df_merged[[
    'web_name',
    'team_name',
    'position',
    'price',
    'minutes',
    'total_points',
    'form',
    'points_per_game',
    'clearances_blocks_interceptions',
    'tackles',
    'recoveries',
    'creativity',
    'threat',
    'influence',
    'Attack multiplier',
    'Defense multiplier'
]]








################### regression models ############################





# --- 1. Data Preparation ---

# Filter for players with > 120 minutes
df_reg = df_players_final[df_players_final['minutes'] > 120].copy()

# Calculate per-90 statistics
df_reg['pts_90'] = (df_reg['total_points'] / df_reg['minutes']) * 90
df_reg['threat_90'] = (df_reg['threat'] / df_reg['minutes']) * 90
df_reg['creativity_90'] = (df_reg['creativity'] / df_reg['minutes']) * 90
df_reg['cbit_90'] = ((df_reg['clearances_blocks_interceptions'] + df_reg['tackles'] )/ df_reg['minutes']) * 90
df_reg['cbirt_90'] = ((df_reg['clearances_blocks_interceptions'] + df_reg['tackles']+df_reg['recoveries'] )/ df_reg['minutes']) * 90


# Clean any potential NaN/Inf values
df_reg = df_reg.dropna(subset=[
    'pts_90', 'threat_90', 'creativity_90', 'team_name','cbit_90','cbirt_90'
])

# --- 2. Create Positional DataFrames ---
df_gkp = df_reg[df_reg['position'] == 'GKP'].copy()
df_def = df_reg[df_reg['position'] == 'DEF'].copy()
df_mid = df_reg[df_reg['position'] == 'MID'].copy()
df_fwd = df_reg[df_reg['position'] == 'FWD'].copy()

print(f"Running models on (players > 120 mins): {len(df_gkp)} GKPs, {len(df_def)} DEFs, {len(df_mid)} MIDs, {len(df_fwd)} FWDs")

# --- 3. Run Regression Models & Add Predictions ---

#  GKP Model
print("\n" + "="*30 + " GKP Model " + "="*30)
if not df_gkp.empty:
    gkp_formula = 'pts_90 ~ C(team_name)'
    gkp_model = smf.ols(formula=gkp_formula, data=df_gkp).fit()
    print(gkp_model.summary())
    # Add predictions to the GKP dataframe
    df_gkp['predicted_pts_90'] = gkp_model.predict(df_gkp)
else:
    print("No GKP data to model.")

# 🛡️ DEF Model
print("\n" + "="*30 + " DEF Model " + "="*30)
if not df_def.empty:
    def_formula = 'pts_90 ~ C(team_name)   +I(cbit_90**2)+ creativity_90 + threat_90'
    def_model = smf.ols(formula=def_formula, data=df_def).fit()
    print(def_model.summary())
    # Add predictions to the DEF dataframe
    df_def['predicted_pts_90'] = def_model.predict(df_def)
else:
    print("No DEF data to model.")

# 🧠 MID Model
print("\n" + "="*30 + " MID Model " + "="*30)
if not df_mid.empty:
    mid_formula = 'pts_90 ~ C(team_name)+ creativity_90 + I(cbirt_90**2) + threat_90'
    mid_model = smf.ols(formula=mid_formula, data=df_mid).fit()
    print(mid_model.summary())
    # Add predictions to the MID dataframe
    df_mid['predicted_pts_90'] = mid_model.predict(df_mid)
else:
    print("No MID data to model.")

# ⚡ FWD Model
print("\n" + "="*30 + " FWD Model " + "="*30)
if not df_fwd.empty:
    fwd_formula = 'pts_90 ~ creativity_90 + threat_90'
    fwd_model = smf.ols(formula=fwd_formula, data=df_fwd).fit()
    print(fwd_model.summary())
    # Add predictions to the FWD dataframe
    df_fwd['predicted_pts_90'] = fwd_model.predict(df_fwd)
else:
    print("No FWD data to model.")

# --- 4. Combine Predictions into One DataFrame ---
df_final_predictions = pd.concat([df_gkp, df_def, df_mid, df_fwd])




#points per game
df_final_predictions['predicted_pts_per_game'] = df_final_predictions['predicted_pts_90'] * (df_final_predictions['minutes']) / (df_final_predictions['minutes'].max())

df_final_predictions['fx_predicted_pts_per_game'] = np.where(
    df_final_predictions['position'].isin(['MID', 'FWD']),  # Condition: Is player MID or FWD?
    df_final_predictions['predicted_pts_per_game'] * df_final_predictions['Attack multiplier'], # Value if True
    df_final_predictions['predicted_pts_per_game'] * df_final_predictions['Defense multiplier']  # Value if False (GKP or DEF)
)

df_final_predictions['fx_predicted_pts_per_game_per_£'] = df_final_predictions['fx_predicted_pts_per_game'] / df_final_predictions['price']



cols_to_display = [
    'web_name', 
    'team_name', 
    'position', 
    'price', 
    'minutes',
    'Attack multiplier',
    'Defense multiplier',
    'total_points', 
    'form',
    'points_per_game',
    'pts_90',
    'predicted_pts_90',
    'predicted_pts_per_game',
    'fx_predicted_pts_per_game',
    'fx_predicted_pts_per_game_per_£'

]


####################     gkp    ##################

print("\n" + "="*60)
print("Top 50 Players (DEF) RAW")
print("="*60)

# Display the most useful columns, sorted by the new prediction
print(df_final_predictions[df_final_predictions['position']=="DEF"].sort_values(by='fx_predicted_pts_per_game', ascending=False)[cols_to_display].head(50))

print("\n" + "="*60)
print("Top 50 Players (DEF) VALUE")
print("="*60)

# Display the most useful columns, sorted by the new prediction
print(df_final_predictions[df_final_predictions['position']=="DEF"].sort_values(by='fx_predicted_pts_per_game_per_£', ascending=False)[cols_to_display].head(50))





####################     mid    ##################

print("\n" + "="*60)
print("Top 50 Players (MID) RAW")
print("="*60)

# Display the most useful columns, sorted by the new prediction
print(df_final_predictions[df_final_predictions['position']=="MID"].sort_values(by='fx_predicted_pts_per_game', ascending=False)[cols_to_display].head(50))

print("\n" + "="*60)
print("Top 50 Players (MID) VALUE")
print("="*60)

# Display the most useful columns, sorted by the new prediction
print(df_final_predictions[df_final_predictions['position']=="MID"].sort_values(by='fx_predicted_pts_per_game_per_£', ascending=False)[cols_to_display].head(50))





####################     fwd    ##################

print("\n" + "="*60)
print("Top 50 Players (FWD) RAW")
print("="*60)

# Display the most useful columns, sorted by the new prediction
print(df_final_predictions[df_final_predictions['position']=="FWD"].sort_values(by='fx_predicted_pts_per_game', ascending=False)[cols_to_display].head(50))

print("\n" + "="*60)
print("Top 50 Players (FWD) VALUE")
print("="*60)

# Display the most useful columns, sorted by the new prediction
print(df_final_predictions[df_final_predictions['position']=="FWD"].sort_values(by='fx_predicted_pts_per_game_per_£', ascending=False)[cols_to_display].head(50))






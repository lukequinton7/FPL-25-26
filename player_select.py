import fixture_quality as fq
import pandas as pd


#config
gameweek = 10
GAMES_TO_CHECK = 1




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
    'total_points',
    'form',
    'points_per_game',
    'Attack multiplier',
    'Defense multiplier'
]]


#attacking stats- predicting points over X games

print("")
print("---- Attackers and Midfield ----")

df_attackers_mids = df_players_final[df_players_final['position'].isin(['MID', 'FWD'])]
df_attackers_mids = df_attackers_mids[[
    'web_name',
    'team_name',
    'position',
    'price',
    'total_points',
    'form',
    'points_per_game',
    'Attack multiplier' 
]]

df_attackers_mids['attack_adjusted_points'] = (((df_attackers_mids['total_points']/(gameweek-1)) *0.5) + (df_attackers_mids['form']*0.5))         *      df_attackers_mids['Attack multiplier']
df_attackers_mids['attack_adj_pts_per_million'] = df_attackers_mids['attack_adjusted_points'] / df_attackers_mids['price']

print("RAW")

df_attackers_mids = df_attackers_mids.sort_values(by='attack_adjusted_points', ascending=False)

print(df_attackers_mids.head(30))

print("")

print("VALUE")

df_attackers_mids = df_attackers_mids.sort_values(by='attack_adj_pts_per_million', ascending=False)

print(df_attackers_mids.head(30))




#defending stats- predicting points over X games

print("")
print("---- Defenders & Goalkeepers ----")

df_gk_def = df_players_final[df_players_final['position'].isin(['GKP', 'DEF'])]
df_gk_def = df_gk_def[[
    'web_name',
    'team_name',
    'position',
    'price',
    'total_points',
    'form',
    'points_per_game',
    'Defense multiplier' 
]]

df_gk_def['defense_adjusted_points'] = (((df_gk_def['total_points']/(gameweek-1)) *0.5) + (df_gk_def['form']*0.5))         *      df_gk_def['Defense multiplier']
df_gk_def['defense_adj_pts_per_million'] = df_gk_def['defense_adjusted_points'] / df_gk_def['price']

print("RAW")

df_gk_def = df_gk_def.sort_values(by='defense_adjusted_points', ascending=False)

print(df_gk_def.head(30))

print("")

print("VALUE")

df_gk_def = df_gk_def.sort_values(by='defense_adj_pts_per_million', ascending=False)

print(df_gk_def.head(30))


































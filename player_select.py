import fixture_quality as fq
import pandas as pd


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
    'creativity',
    'threat',
    'influence',
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
    'minutes',
    'total_points',
    'form',
    'creativity',
    'threat',
    'points_per_game',
    'Attack multiplier' 
]]

df_attackers_mids['attack_adjusted_points'] = (((df_attackers_mids['total_points']/(gameweek-1)) *(1-form_weight)) + (df_attackers_mids['form']*form_weight))         *      df_attackers_mids['Attack multiplier']
df_attackers_mids['attack_adj_pts_per_million'] = df_attackers_mids['attack_adjusted_points'] / df_attackers_mids['price']

df_attackers_mids['creativity+threat']= (df_attackers_mids['creativity']*creativity_weight + df_attackers_mids['threat']* threat_weight)     *      df_attackers_mids['Attack multiplier']
df_attackers_mids['creativity+threat/price']=df_attackers_mids['creativity+threat'] / df_attackers_mids['price']


print("RAW")

df_attackers_mids = df_attackers_mids.sort_values(by='creativity+threat', ascending=False)

print(df_attackers_mids.head(30))

print("")

print("VALUE")

df_attackers_mids = df_attackers_mids.sort_values(by='creativity+threat/price', ascending=False)

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

df_gk_def['defense_adjusted_points'] = (((df_gk_def['total_points']/(gameweek-1)) *(1-form_weight)) + (df_gk_def['form']*form_weight))         *      df_gk_def['Defense multiplier']
df_gk_def['defense_adj_pts_per_million'] = df_gk_def['defense_adjusted_points'] / df_gk_def['price']

print("RAW")

df_gk_def = df_gk_def.sort_values(by='defense_adjusted_points', ascending=False)

print(df_gk_def.head(30))

print("")

print("VALUE")

df_gk_def = df_gk_def.sort_values(by='defense_adj_pts_per_million', ascending=False)

print(df_gk_def.head(30))






mid_data = df_attackers_mids[df_attackers_mids['position'].isin(['MID'])]
fwd_data = df_attackers_mids[df_attackers_mids['position'].isin(['FWD'])]
gkp_data = df_gk_def[df_gk_def['position'].isin(['GKP'])]
def_data = df_gk_def[df_gk_def['position'].isin(['DEF'])]






################### create starting lineup  #########################

def start_lineup():



    # Sort by creativity+threat
    mid_data_5 = mid_data.sort_values(by='creativity+threat', ascending=False).head(25)

    fwd_data_3 = fwd_data.sort_values(by='creativity+threat', ascending=False).head(25)

    def_data_5 = def_data.sort_values(by='defense_adjusted_points', ascending=False).head(25)

    gkp_data_2 = gkp_data.sort_values(by='defense_adjusted_points', ascending=False).head(25)

    start_15_df =pd.concat([
        mid_data_5.head(5),
        fwd_data_3.head(3),
        def_data_5.head(5),
        gkp_data_2.head(2)
    ])




    print(start_15_df) #best 15

    current_val = start_15_df['price'].sum() #price of best 15

    print("best 15- total price:   ", current_val) 



    print("over budget- removing lowest value player")

    #remove lowest value attack/mid player
    lowest_value_player = start_15_df.loc[start_15_df['creativity+threat/price'].idxmin()]


    print("removed: ", lowest_value_player)

        #replace with next best mid/forward
    if lowest_value_player['position'] == 'MID':
        replacement_pool = df_mids_sorted[~df_mids_sorted.index.isin(start_15_df.index)]
        next_best = replacement_pool.sort_values(by='creativity+threat/price', ascending=False).head(1)
        start_15_df = pd.concat([start_15_df, next_best])
    else:
        ""
    

    start_15_df = start_15_df.drop(lowest_value_player.name)














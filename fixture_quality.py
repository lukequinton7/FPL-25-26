import pandas as pd
import numpy as np

# Fixtures to df
df_fixtures = pd.read_csv("epl_fixtures_2025.csv")

def prem_table():

    # Use the correct column names: 'HomeTeamScore' and 'AwayTeamScore'
    df_played = df_fixtures.dropna(subset=['HomeTeamScore', 'AwayTeamScore'])

    home_stats = df_played.groupby('HomeTeam').agg(
        GS=('HomeTeamScore', 'sum'),
        GC=('AwayTeamScore', 'sum'),
        GP=('HomeTeamScore', 'count')
    )

    away_stats = df_played.groupby('AwayTeam').agg(
        GS=('AwayTeamScore', 'sum'),
        GC=('HomeTeamScore', 'sum'),
        GP=('AwayTeamScore', 'count')
    )

    df_table = home_stats.add(away_stats, fill_value=0)

    df_table['GD'] = df_table['GS'] - df_table['GC']


    df_table['GS_score'] = (df_table['GS']/df_table['GP'])

    df_table['GS_concede'] = (df_table['GC']/df_table['GP'])

    df_table = df_table.astype(float).round(2)
    df_table = df_table.sort_values(by='GD', ascending=False)
    df_table = df_table.reset_index().rename(columns={'index': 'Team'})

    df_table = df_table[['HomeTeam', 'GP', 'GS', 'GC', 'GD', 'GS_score','GS_concede']]
 
    return df_table






def calculate_attack_score(df_table, df_unplayed, num_games):
    """
    Calculates the attack difficulty score for each team based on their
    next 'num_games' fixtures.
    """
    
    # Create the lookup maps using 'HomeTeam' as the key
    opponent_concede_map = df_table.set_index('HomeTeam')['GS_concede']
    team_score_map = df_table.set_index('HomeTeam')['GS_score']

    # Create new lists to store the results
    all_difficulty_scores = []
    all_avg_opponent_gc = []  

    for team in df_table['HomeTeam']:
        
        counter = 0
        opponent_gc_scores = []
        team_gs_score = team_score_map.get(team)

        # --- Inner loop ---
        for index, fixture in df_unplayed.iterrows():
            
            if fixture['HomeTeam'] == team or fixture['AwayTeam'] == team:
                
                opponent = ""
                if fixture['HomeTeam'] == team:
                    opponent = fixture['AwayTeam']
                    opponent_gc = opponent_concede_map.get(opponent) * 1.15

                else:
                    opponent = fixture['HomeTeam']
                    opponent_gc = opponent_concede_map.get(opponent) * 0.85
                
                #opponent_gc = opponent_concede_map.get(opponent)
                opponent_gc_scores.append(opponent_gc)
                
                counter += 1
                
                if counter == num_games: # <-- Use the input parameter
                    break
        
        # --- After the inner loop ---
        final_score = pd.NA
        avg_opponent_gc = pd.NA  
        
        if opponent_gc_scores: 
            avg_opponent_gc = pd.Series(opponent_gc_scores).mean()
            
            if avg_opponent_gc > 0:
                final_score = (0.00 * team_gs_score + 1.00 * avg_opponent_gc)
                
        all_difficulty_scores.append(final_score)
        all_avg_opponent_gc.append(avg_opponent_gc) 

    # --- Add new columns to the table ---
    df_table['Attack Score'] = all_difficulty_scores
    df_table['Opponent_GC_Avg'] = all_avg_opponent_gc

    scale_factor =  (df_table['Attack Score'] - df_table['Attack Score'].min()) / (df_table['Attack Score'].max() - df_table['Attack Score'].min())
    df_table['Attack multiplier'] = 0.75 + (0.5* scale_factor)

    # Round the new columns
    df_table['Attack Score'] = df_table['Attack Score'].round(2)
    df_table['Opponent_GC_Avg'] = df_table['Opponent_GC_Avg'].round(2)
    df_table['Attack multiplier'] = df_table['Attack multiplier'].round(2)
    
    return df_table


def calculate_defense_score(df_table, df_unplayed, num_games):
    """
    Calculates the defense difficulty score for each team based on their
    next 'num_games' fixtures.
    """
    
    # Create the lookup maps using 'HomeTeam' as the key
    opponent_attack_map = df_table.set_index('HomeTeam')['GS_score']
    team_defense_map = df_table.set_index('HomeTeam')['GS_concede']

    # Create new lists to store the results
    all_defense_scores = []
    all_avg_opponent_gs = []  

    for team in df_table['HomeTeam']:
        
        counter = 0
        opponent_gs_scores = [] 
        team_gc_score = team_defense_map.get(team) 

        # --- Inner loop ---
        for index, fixture in df_unplayed.iterrows():
            
            if fixture['HomeTeam'] == team or fixture['AwayTeam'] == team:
                
                opponent = ""
                if fixture['HomeTeam'] == team:
                    opponent = fixture['AwayTeam']
                    opponent_gs = opponent_attack_map.get(opponent) * 0.85
                else:
                    opponent = fixture['HomeTeam']
                    opponent_gs = opponent_attack_map.get(opponent) * 1.15
                
                #opponent_gs = opponent_attack_map.get(opponent) 
                opponent_gs_scores.append(opponent_gs) 
                
                counter += 1
                
                if counter == num_games: 
                    break 
        
        # --- After the inner loop ---
        final_score = pd.NA
        avg_opponent_gs = pd.NA  
        
        if opponent_gs_scores: 
            avg_opponent_gs = pd.Series(opponent_gs_scores).mean() 
            
            if avg_opponent_gs > 0:
                final_score = (0.0 * team_gc_score) + (1.0 * avg_opponent_gs)
                
        all_defense_scores.append(final_score)
        all_avg_opponent_gs.append(avg_opponent_gs) 

    # --- Add new columns to the table ---
    df_table['Defense Score'] = all_defense_scores
    df_table['Opponent_GS_Avg'] = all_avg_opponent_gs

    scale_factor_d =  (df_table['Defense Score'].max() - df_table['Defense Score']) / (df_table['Defense Score'].max() - df_table['Defense Score'].min())
    df_table['Defense multiplier'] = 0.75 + (0.5* scale_factor_d)
    

    # Round the new columns
    df_table['Defense Score'] = df_table['Defense Score'].round(2)
    df_table['Opponent_GS_Avg'] = df_table['Opponent_GS_Avg'].round(2)
    df_table['Defense multiplier'] = df_table['Defense multiplier'].round(2)

    return df_table


import pandas as pd
import os
import requests



def refresh_player_data():

    # The direct URL to the raw CSV data
    raw_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/cleaned_players.csv"

    local_filename = "player_stats.csv"

    print(f"Downloading data from {raw_url}...")

    try:
        df = pd.read_csv(raw_url)

        df.to_csv(local_filename, index=False)

        print(f"\nSuccessfully downloaded and saved data as '{local_filename}'")
        print("\n--- First 5 rows ---")
        print(df.head())
        print("--------------------")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check the URL or your internet connection.")


def refresh_fixture_data():
    URL = "https://fixturedownload.com/feed/json/epl-2025"
    FILENAME = "epl_fixtures_2025.csv"

    print(f"Attempting to download data from {URL}...")

    try:
        # 1. Pandas can read a JSON URL directly into a DataFrame
        df = pd.read_json(URL)
        
        print("Download successful. First 5 rows of data:")
        print(df.head())
        print("\n------------------------------\n")

        # 2. Save the DataFrame to a CSV
        df.to_csv(FILENAME, index=False)
        
        print(f"Successfully saved data to {FILENAME}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check the URL and your internet connection.")






# Execute the data refresh functions
refresh_player_data()
refresh_fixture_data()








def fetch_fpl_player_stats(filename='fpl_player_stats.csv'):
    """
    Fetches ALL FPL players' TOTAL SEASON STATS and saves to CSV.
    
    Columns include: ID, Name, Team, Position, Price, Status, 
    Minutes, Goals, Assists, Clean Sheets, Saves, Bonus, Total Points, PPG, etc.
    
    Returns: pandas DataFrame (sorted by total_points DESC)
    """
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(url)
    response.raise_for_status()  # Raises error if fetch fails
    
    data = response.json()
    
    # Position & Team maps
    position_map = {p['id']: p['singular_name_short'] for p in data['element_types']}
    team_map = {t['id']: t['name'] for t in data['teams']}
    
    # Extract player data
    players_data = []
    for p in data['elements']:
        player = {
            'id': p['id'],
            'web_name': p['web_name'],
            'full_name': f"{p.get('first_name', '')} {p.get('second_name', '')}".strip(),
            'team_name': team_map.get(p['team'], 'Unknown'),
            'position': position_map.get(p['element_type'], 'Unknown'),
            'status': p.get('status', 'a'),  # 'a'=active
            'price': p['now_cost'] / 10.0,
            'total_points': p['total_points'],
            'minutes': p['minutes'],
            'goals_scored': p['goals_scored'],
            'assists': p['assists'],
            'clean_sheets': p['clean_sheets'],
            'goals_conceded': p['goals_conceded'],
            'saves': p['saves'],
            'bonus': p['bonus'],
            'yellow_cards': p['yellow_cards'],
            'red_cards': p['red_cards'],
            'penalties_missed': p.get('penalties_missed', 0),
            'penalties_saved': p.get('penalties_saved', 0),
            'own_goals': p.get('own_goals', 0),
            'bps': p['bps'],
            'influence': round(float(p.get('influence', 0)), 2),
            'creativity': round(float(p.get('creativity', 0)), 2),
            'threat': round(float(p.get('threat', 0)), 2),
            'ict_index': round(float(p.get('ict_index', 0)), 2),
            'form': p.get('form', 0),
            'points_per_game': round(float(p.get('points_per_game', 0)), 2),
            'ep_this': p.get('ep_this', 0),  # Expected pts this GW
            'ep_next': p.get('ep_next', 0),  # Expected pts next GW
            'selected_pct': round(float(p.get('selected_by_percent', 0)), 2),
            'in_dreamteam': p['in_dreamteam']
        }
        players_data.append(player)
    
    # Create & sort DataFrame
    df = pd.DataFrame(players_data)
    df = df.sort_values('total_points', ascending=False).reset_index(drop=True)
    
    # Save CSV
    df.to_csv(filename, index=False)
    print(f"✅ Saved **{len(df)} players** to '{filename}' (sorted by Total Points)")
    print("\nTop 5 players preview:")
    print(df[['web_name', 'team_name', 'total_points', 'goals_scored', 'assists']].head())
    
    return df

df = fetch_fpl_player_stats('player_statistics.csv')  # Saves CSV + returns DataFrame
df.head()  # Preview
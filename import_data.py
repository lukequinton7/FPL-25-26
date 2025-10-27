import pandas as pd
import os


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



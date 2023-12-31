import sqlite3

import pandas as pd

import settings
import quiniela.aux_funcs as aux

def load_matchday(season, division, matchday):
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        data = pd.read_sql(f"""
            SELECT score, home_team, away_team, season, division, matchday FROM Matches
                WHERE season = '{season}'
                  AND division = {division}
                  AND matchday = {matchday}
        """, conn)
    
        # create the features
        features_df = aux.calculate_single_season_division_standings(season, division, data)
        
        data = aux.create_final_dataset(data, features_df)    
    if data.empty:
        raise ValueError("There is no matchday data for the values given")
    return data

# We adapt this function to grab our desired features.
def load_historical_data(seasons):
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        if seasons == "all":
            data = pd.read_sql("SELECT score, home_team, away_team, season, division, matchday FROM Matches", conn)

            data = aux.prepare_data(data)
        else:
            data = pd.read_sql(f"""
                SELECT score, home_team, away_team, season, division, matchday FROM Matches
                    WHERE season IN {tuple(seasons)}
            """, conn)
            
            data = aux.prepare_data(data)
    if data.empty:
        raise ValueError(f"No data for seasons {seasons}")
    return data


def save_predictions(predictions):
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        predictions.to_sql(name="Predictions", con=conn, if_exists="append", index=False)

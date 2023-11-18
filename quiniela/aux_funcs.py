import pandas as pd

# Helper function to determine match outcomes 
def encode_last_5(results_list):
    # Define points for win, draw, loss
    points = {'W': 3, 'D': 1, 'L': 0}
    
    # Calculate the total points for the last 5 matches
    total_points = sum(points[result] for result in results_list if result in points)
    
    return total_points

def determine_outcome(home_goals, away_goals):
    if home_goals > away_goals:
        return 'W', 'L'
    elif home_goals < away_goals:
        return 'L', 'W'
    else:
        return 'T', 'T'
    
# Function to calculate relative strength
def calculate_relative_strength(standings_df, matchday):
    # Calculate the maximum possible points so far for each team
    max_points_so_far = matchday * 3
    # Calculate the relative strength for each team as the percentage of points obtained out of the total possible
    standings_df['relative_strength'] = standings_df['PTS'] / max_points_so_far * 100
    return standings_df


def calculate_single_season_division_standings(season, division, matches_df):
    # Filter matches for the given season and division
    season_matches = matches_df[(matches_df['season'] == season) & (matches_df['division'] == division)].sort_values('matchday')
    matchdays = season_matches['matchday'].unique()

    # Initialize the standings dict with teams
    teams = pd.unique(season_matches[['home_team', 'away_team']].values.ravel('K'))
    standings_tracker = {team: {'GF_home': 0, 'GA_home': 0, 'GF_away': 0, 'GA_away': 0, 'W': 0, 'L': 0, 'T': 0, 'PTS': 0, 'last_5': []} for team in teams}

    # List to collect matchday standings
    all_standings = []

    # Process each matchday
    for matchday in matchdays:
        matchday_matches = season_matches[season_matches['matchday'] == matchday]
        for index, match in matchday_matches.iterrows():
            home_team, away_team = match['home_team'], match['away_team']
            home_goals, away_goals = map(int, match['score'].split(':'))
            home_outcome, away_outcome = determine_outcome(home_goals, away_goals)

            # Update goals for and against at home and away
            standings_tracker[home_team]['GF_home'] += home_goals
            standings_tracker[away_team]['GF_away'] += away_goals
            standings_tracker[home_team]['GA_home'] += away_goals
            standings_tracker[away_team]['GA_away'] += home_goals

            # Update last 5 matches
            if matchday > 1:  # Only update if it's not the first matchday
                standings_tracker[home_team]['last_5'].insert(0, home_outcome)
                standings_tracker[away_team]['last_5'].insert(0, away_outcome)

            # Ensure last_5 lists do not exceed 5 matches
            standings_tracker[home_team]['last_5'] = standings_tracker[home_team]['last_5'][:5]
            standings_tracker[away_team]['last_5'] = standings_tracker[away_team]['last_5'][:5]

            # Update wins, losses, ties, and points
            if home_goals > away_goals:  # Home win
                standings_tracker[home_team]['W'] += 1
                standings_tracker[home_team]['PTS'] += 3
                standings_tracker[away_team]['L'] += 1
            elif home_goals < away_goals:  # Away win
                standings_tracker[away_team]['W'] += 1
                standings_tracker[away_team]['PTS'] += 3
                standings_tracker[home_team]['L'] += 1
            else:  # Tie
                standings_tracker[home_team]['T'] += 1
                standings_tracker[home_team]['PTS'] += 1
                standings_tracker[away_team]['T'] += 1
                standings_tracker[away_team]['PTS'] += 1

        # Calculate goal difference for each team
        for team in teams:
            standings_tracker[team]['GD'] = standings_tracker[team]['GF_home'] + standings_tracker[team]['GF_away'] - \
                                             standings_tracker[team]['GA_home'] - standings_tracker[team]['GA_away']

        # Create standings DataFrame for the current matchday
        matchday_standings = (pd.DataFrame.from_dict(standings_tracker, orient='index')
                                .reset_index()
                                .rename(columns={'index': 'team'}))
        matchday_standings['matchday'] = matchday
        matchday_standings['season'] = season
        matchday_standings['division'] = division

        # Sort standings
        matchday_standings.sort_values(by=['PTS', 'GD', 'GF_home', 'GF_away'], ascending=[False, False, False, False], inplace=True)
        matchday_standings['rank'] = matchday_standings.reset_index(drop=True).index + 1

        # Calculate the relative strength
        matchday_standings = calculate_relative_strength(matchday_standings, matchday)

        # Append to the list
        all_standings.append(matchday_standings)

    # Concatenate all matchday standings
    final_standings = pd.concat(all_standings, ignore_index=True)
    # Reorder columns
    final_standings = final_standings[['season', 'division', 'matchday', 'rank', 'team', 'GD', 'GF_home', 'GA_home', 'GF_away', 'GA_away', 'W', 'L', 'T', 'PTS', 'last_5', 'relative_strength']]
    
    return final_standings

def calculate_all_seasons_divisions_standings(matches_df):
    # Initialize the final DataFrame
    final_all_standings = pd.DataFrame()

    # Process each season and division without explicit loops
    for (season, division), group_df in matches_df.groupby(['season', 'division']):
        season_division_standings = calculate_single_season_division_standings(season, division, group_df)
        final_all_standings = pd.concat([final_all_standings, season_division_standings], ignore_index=True)

    return final_all_standings

def encode_match_outcome(score):
    home_goals, away_goals = map(int, score.split(':'))
    if home_goals > away_goals:
        return '1'  # Home win
    elif home_goals < away_goals:
        return '2'  # Away win
    else:
        return 'X'  # Draw

def create_final_dataset(matches_df, features_df):
    # First merge for Home Team features
    matches_df = pd.merge(
        left=matches_df,
        right=features_df,
        how='left',
        left_on=['season', 'division', 'matchday', 'home_team'],
        right_on=['season', 'division', 'matchday', 'team'],
        suffixes=('', '_home')
    )

    # Rename the merged columns for the Home Team
    home_feature_columns = {
        'GD_home' : 'GD_HomeTeam',
        'GF_home_home': 'GFH_HomeTeam',
        'GA_home_home': 'GAH_HomeTeam',
        'GF_away_home': 'GFA_HomeTeam',
        'GA_away_home': 'GAA_HomeTeam',
        'PTS_home': 'PTS_HomeTeam',
        'last_5_home': 'last_5_HomeTeam',
        'relative_strength_home': 'relative_strength_HomeTeam',
        'rank_home': 'rank_HomeTeam'
    }
    matches_df.rename(columns=home_feature_columns, inplace=True)

    # Second merge for Away Team features
    matches_df = pd.merge(
        left=matches_df,
        right=features_df,
        how='left',
        left_on=['season', 'division', 'matchday', 'away_team'],
        right_on=['season', 'division', 'matchday', 'team'],
        suffixes=('', '_away')
    )

    # Rename the merged columns for the Away Team
    away_feature_columns = {
        'GD_away' : 'GD_AwayTeam',
        'GF_home_away': 'GFH_AwayTeam',
        'GA_home_away': 'GAH_AwayTeam',
        'GF_away_away': 'GFA_AwayTeam',
        'GA_away_away': 'GAA_AwayTeam',
        'PTS_away': 'PTS_AwayTeam',
        'last_5_away': 'last_5_AwayTeam',
        'relative_strength_away': 'relative_strength_AwayTeam',
        'rank_away': 'rank_AwayTeam'
    }
    matches_df.rename(columns=away_feature_columns, inplace=True)
    matches_df.drop(['team_away'], axis=1, inplace=True)

    matches_df['season_start'] = matches_df['season'].apply(lambda x: int(x.split('-')[0]))
    min_season = matches_df['season_start'].min()
    matches_df['season_since_start'] = matches_df['season_start'] - min_season
    
    # Now apply this function to the last_5 column of your DataFrame
    matches_df['last_5_AwayTeam'] = matches_df['last_5_AwayTeam'].apply(encode_last_5)
    matches_df['last_5_HomeTeam'] = matches_df['last_5'].apply(encode_last_5)

    matches_df.drop(['last_5', 'score', 'team', 'PTS', 'season'], axis=1, inplace=True)
    
    return matches_df

def prepare_data(data):
    data.dropna(subset=['score'], inplace=True)
    data['outcome'] = data['score'].apply(encode_match_outcome)    
    features_df = calculate_all_seasons_divisions_standings(data)
    
    matches_df = data

    return create_final_dataset(matches_df, features_df)
    
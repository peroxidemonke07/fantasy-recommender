"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.9
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def calculate_batting_scorecard(deliveries_df: pd.DataFrame, matches_df: pd.DataFrame)->pd.DataFrame:
    """
    Calculate the batting scorecard per match, including runs, fours, sixes, balls faced, 
    strike rate, and batting points. The batting points are awarded based on runs, boundaries, 
    strike rate, and specific milestones. Additionally, match information such as match date, 
    opponent team, and venue are added to the scorecard.

    Parameters:
    -----------
    deliveries_df : pd.DataFrame
        A DataFrame containing ball-by-ball data of the match with columns such as 
        'match_id', 'batter', 'batsman_runs', and 'batting_team'.
    matches_df : pd.DataFrame
        A DataFrame containing match details with columns such as 'id', 'season', 'city', 
        'date', 'match_type', 'venue', 'team1', 'team2', etc.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the batting summary per player with columns such as 
        'batter', 'runs', 'fours', 'sixes', 'balls_faced', 'batting_points', 'strike_rate', 
        'match_date', 'opponent_team', and 'venue'.
    """
    # Group by match_id and batsman
    grouped_df = deliveries_df.groupby(['match_id', 'batter'])
    
    # Aggregate batting stats
    batsman_summary = grouped_df.agg(
        runs=pd.NamedAgg(column='batsman_runs', aggfunc='sum'),
        fours=pd.NamedAgg(column='batsman_runs', aggfunc=lambda x: (x == 4).sum()),
        sixes=pd.NamedAgg(column='batsman_runs', aggfunc=lambda x: (x == 6).sum()),
        balls_faced=pd.NamedAgg(column='batsman_runs', aggfunc='count'),
        player_team=pd.NamedAgg(column='batting_team', aggfunc='first'),
        opponent_team=pd.NamedAgg(column='bowling_team', aggfunc='first')
    )

    # Batting Points Calculation
    batsman_summary['strike_rate'] = (batsman_summary['runs'] / batsman_summary['balls_faced'].replace(0, 1)) * 100
    batsman_summary['batting_points'] = (
        batsman_summary['runs'] + batsman_summary['fours'] + batsman_summary['sixes'] * 2
    )
    batsman_summary['batting_points'] += (
        (batsman_summary['runs'] >= 100) * 8 +
        (batsman_summary['runs'] >= 50) * 4 +
        (batsman_summary['runs'] >= 30) * 4
    )
    batsman_summary.loc[batsman_summary['runs'] == 0, 'batting_points'] = -2

    batsman_summary['strike_rate_points'] = 0
    batsman_summary.loc[batsman_summary['balls_faced'] > 10, 'strike_rate_points'] = (
        (batsman_summary['strike_rate'] > 170) * 6 +
        ((batsman_summary['strike_rate'] > 150) & (batsman_summary['strike_rate'] <= 170)) * 4 +
        ((batsman_summary['strike_rate'] >= 130) & (batsman_summary['strike_rate'] <= 150)) * 2 +
        ((batsman_summary['strike_rate'] >= 60) & (batsman_summary['strike_rate'] <= 70)) * -2 +
        ((batsman_summary['strike_rate'] >= 50) & (batsman_summary['strike_rate'] < 60)) * -4 +
        (batsman_summary['strike_rate'] < 50) * -6
    )

    batsman_summary['batting_points'] += batsman_summary['strike_rate_points']
    batsman_summary.drop(columns=['strike_rate_points'], inplace=True)
    
    # Merge with match data
    batsman_summary.reset_index(inplace=True)
    batsman_summary = batsman_summary.merge(
        matches_df[['id', 'date', 'venue']], 
        left_on='match_id', right_on='id', 
        how='left'
    )
    batsman_summary.drop(columns=['id'], inplace=True)
    
    # Sort by batsman
    batsman_summary.sort_values(by=['batter','date'], inplace=True)

    return batsman_summary



def calculate_bowling_scorecard(deliveries_df: pd.DataFrame, matches_df: pd.DataFrame)->pd.DataFrame:
    """
    Calculate the bowling scorecard per match, including runs conceded, balls bowled, 
    wickets taken, economy rate, and bowling points. The bowling points are awarded based on 
    wickets taken, economy rate, and specific milestones. Additionally, match information such as 
    match date, opponent team, and venue are added to the scorecard.

    Parameters:
    -----------
    deliveries_df : pd.DataFrame
        A DataFrame containing ball-by-ball data of the match with columns such as 
        'match_id', 'bowler', 'total_runs', 'dismissal_kind', and 'bowling_team'.
    matches_df : pd.DataFrame
        A DataFrame containing match details with columns such as 'id', 'season', 'city', 
        'date', 'match_type', 'venue', 'team1', 'team2', etc.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the bowling summary per player with columns such as 
        'bowler', 'runs_conceded', 'wickets_taken', 'economy_rate', 'bowling_points', 
        'bowling_average', 'match_date', 'opponent_team', and 'venue'.
    """
    # Group by match_id and bowler
    grouped_df = deliveries_df.groupby(['match_id', 'bowler'])
    
    # Aggregate bowling stats
    bowler_summary = grouped_df.agg(
        runs_conceded=pd.NamedAgg(column='total_runs', aggfunc='sum'),
        fours_conceded=pd.NamedAgg(column='total_runs', aggfunc=lambda x: ((x == 4) | (x == 5)).sum()),
        sixes_conceded=pd.NamedAgg(column='total_runs', aggfunc=lambda x: ((x == 6) | (x == 7)).sum()),
        wickets_taken=pd.NamedAgg(column='dismissal_kind', aggfunc=lambda x: x.isin(['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']).sum()),
        bowled_and_lbw=pd.NamedAgg(column='dismissal_kind', aggfunc=lambda x: x.isin(['bowled', 'lbw']).sum()),
        balls_bowled=pd.NamedAgg(column='over', aggfunc='count'),
        player_team=pd.NamedAgg(column='bowling_team', aggfunc='first'),
        opponent_team=pd.NamedAgg(column='batting_team', aggfunc='first')
    )

    # Calculate bowling average and economy rate
    bowler_summary['bowling_average'] = bowler_summary['runs_conceded'] / bowler_summary['wickets_taken'].replace(0, 1)
    bowler_summary['economy_rate'] = bowler_summary['runs_conceded'] / (bowler_summary['balls_bowled'] / 6).astype('int').replace(0, 1)

    # Bowling points
    bowler_summary['bowling_points'] = bowler_summary['wickets_taken'] * 25 + bowler_summary['bowled_and_lbw'] * 8
    bowler_summary['bowling_points'] += (
        (bowler_summary['wickets_taken'] >= 5) * 8 +
        (bowler_summary['wickets_taken'] >= 4) * 4 +
        (bowler_summary['wickets_taken'] >= 3) * 4
    )

    # Economy rate based points
    bowler_summary['economy_points'] = 0
    bowler_summary.loc[bowler_summary['balls_bowled'] >= 12, 'economy_points'] = (
        (bowler_summary['economy_rate'] < 5) * 6 +
        ((bowler_summary['economy_rate'] >= 5) & (bowler_summary['economy_rate'] < 6)) * 4 +
        ((bowler_summary['economy_rate'] >= 6) & (bowler_summary['economy_rate'] < 7)) * 2 +
        ((bowler_summary['economy_rate'] >= 10) & (bowler_summary['economy_rate'] < 11)) * -2 +
        ((bowler_summary['economy_rate'] >= 11) & (bowler_summary['economy_rate'] < 12)) * -4 +
        (bowler_summary['economy_rate'] >= 12) * -6
    )

    bowler_summary['bowling_points'] += bowler_summary['economy_points']
    bowler_summary.drop(columns=['economy_points'], inplace=True)

    # Merge with match data
    bowler_summary.reset_index(inplace=True)
    bowler_summary = bowler_summary.merge(
        matches_df[['id', 'date', 'venue']], 
        left_on='match_id', right_on='id', 
        how='left'
    )
    bowler_summary.drop(columns=['id'],inplace=True)

    # Sort by bowler
    bowler_summary.sort_values(by=['bowler','date'], inplace=True)

    return bowler_summary


def calculate_fielding_scorecard(deliveries_df: pd.DataFrame, matches_df: pd.DataFrame)->pd.DataFrame:
    """
    Calculate the fielding scorecard per match, including catches, run outs, stumps, 
    and fielding points. The fielding points are awarded based on the number of catches, 
    run outs, and stumps. Additionally, match information such as match date, opponent team, 
    and venue are added to the scorecard.

    Parameters:
    -----------
    deliveries_df : pd.DataFrame
        A DataFrame containing ball-by-ball data of the match with columns such as 
        'match_id', 'fielder', 'dismissal_kind', etc.
    matches_df : pd.DataFrame
        A DataFrame containing match details with columns such as 'id', 'season', 'city', 
        'date', 'match_type', 'venue', 'team1', 'team2', etc.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the fielding summary per player with columns such as 
        'fielder', 'catches', 'run_outs', 'stumps', 'fielding_points', 
        'match_date', 'opponent_team', and 'venue'.
    """
    # Group by match_id and fielder
    grouped_df = deliveries_df.groupby(['match_id', 'fielder'])
    
    # Aggregate fielding stats
    fielder_summary = grouped_df.agg(
        catches=pd.NamedAgg(column='dismissal_kind', aggfunc=lambda x: (x == 'caught').sum()),
        run_outs=pd.NamedAgg(column='dismissal_kind', aggfunc=lambda x: (x == 'run out').sum()),
        stumps=pd.NamedAgg(column='dismissal_kind', aggfunc=lambda x: (x == 'stumped').sum()),
        player_team=pd.NamedAgg(column='bowling_team', aggfunc='first'),
        opponent_team=pd.NamedAgg(column='batting_team', aggfunc='first')
    )

    # Calculate fielding points
    fielder_summary['fielding_points'] = (
        fielder_summary['catches'] * 10 + 
        fielder_summary['run_outs'] * 15 + 
        fielder_summary['stumps'] * 20
    )

    # Multiple catch bonuses
    fielder_summary['fielding_points'] += fielder_summary['catches'].apply(lambda x: 5 if x >= 3 else 0)

    # Merge with match data
    fielder_summary.reset_index(inplace=True)
    fielder_summary = fielder_summary.merge(
        matches_df[['id', 'date', 'venue']], 
        left_on='match_id', right_on='id', 
        how='left'
    )
    fielder_summary.drop(columns=['id'],inplace=True)

    # Sort by fielder
    fielder_summary.sort_values(by=['fielder','date'], inplace=True)

    return fielder_summary


def encode_teams_and_venues(matches_df: pd.DataFrame) -> dict:
    """
    Label encodes the 'venue' and 'team' columns in the matches_df DataFrame using all unique values.
    
    Args:
        matches_df (pd.DataFrame): DataFrame with match data containing 'venue', 'team1', and 'team2' columns.
    
    Returns:
        dict: Dictionary with label encoders for 'venue' and 'team' columns.
    """
    label_encoders = {}

    # Collect unique venues
    unique_venues = matches_df['venue'].unique()

    # Collect unique team names from both 'team1' and 'team2'
    unique_teams = pd.concat([matches_df['team1'], matches_df['team2']]).unique()

    # Initialize and fit label encoders
    venue_encoder = LabelEncoder().fit(unique_venues)
    team_encoder = LabelEncoder().fit(unique_teams)

    # Store the encoders
    label_encoders['venue_encoder'] = venue_encoder
    label_encoders['team_encoder'] = team_encoder

    return label_encoders



def prepare_batting_data(df: pd.DataFrame, label_encoders: dict, sequence_length=5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares features (X) and target (y) for predicting batting points based on past performance.
    
    Args:
        df (pd.DataFrame): Input DataFrame with player batting performance data.
        label_encoders (dict): Dictionary with fitted label encoders for 'venue' and 'team'.
        sequence_length (int): Number of past matches to consider for the sequence.
        
    Returns:
        tuple:
            X_bat (pd.DataFrame): Features DataFrame containing sequential and contextual features.
            y_bat (pd.DataFrame): Target DataFrame for batting points.
    """
    venue_encoder = label_encoders['venue_encoder']
    team_encoder = label_encoders['team_encoder']
    
    df['venue_encoded'] = venue_encoder.transform(df['venue'])
    df['opponent_encoded'] = team_encoder.transform(df['opponent_team'])

    X_seq = []     
    X_venue = []   
    X_opponent = []  
    y = []         

    grouped = df.groupby('batter')

    for _, group in grouped:
        group = group.sort_values(by='date').reset_index(drop=True)
        
        for i in range(sequence_length, len(group)):
            seq_features = group.loc[i-sequence_length:i-1, ['runs', 'fours', 'sixes', 'balls_faced']].values
            X_seq.append(seq_features)

            X_venue.append(group.loc[i, 'venue_encoded'])
            X_opponent.append(group.loc[i, 'opponent_encoded'])

            y.append(group.loc[i, 'batting_points'])

    X_bat = pd.DataFrame(
        np.array(X_seq).reshape(len(X_seq), -1),
        columns=[f"bat_{i}_{j}" for i in range(sequence_length) for j in range(4)]
    )
    
    # X_bat = pd.concat([
    #     X_seq_flattened,
    #     pd.DataFrame({'venue': X_venue, 'opponent': X_opponent})
    # ], axis=1)
    
    y_bat = pd.DataFrame(y, columns=['batting_points'])

    return X_bat, y_bat


def prepare_bowling_data(df: pd.DataFrame, label_encoders: dict, sequence_length=5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares features (X) and target (y) for predicting bowling points based on past performance.

    Args:
        df (pd.DataFrame): Input DataFrame with player bowling performance data.
        label_encoders (dict): Dictionary with fitted label encoders for 'venue' and 'team'.
        sequence_length (int): Number of past matches to consider for the sequence.

    Returns:
        tuple: 
            X_bowl (pd.DataFrame): Features DataFrame containing sequential and contextual features.
            y_bowl (pd.DataFrame): Target DataFrame for bowling points.
    """
    venue_encoder = label_encoders['venue_encoder']
    team_encoder = label_encoders['team_encoder']

    df['venue_encoded'] = venue_encoder.transform(df['venue'])
    df['opponent_encoded'] = team_encoder.transform(df['opponent_team'])

    X_seq = []       
    X_venue = []     
    X_opponent = []  
    y = []           

    grouped = df.groupby('bowler')

    for _, group in grouped:
        group = group.sort_values(by='date').reset_index(drop=True)

        for i in range(sequence_length, len(group)):
            seq_features = group.loc[i-sequence_length:i-1, [
                'runs_conceded', 'fours_conceded', 'sixes_conceded',
                'wickets_taken', 'bowled_and_lbw', 'balls_bowled',
                'bowling_average', 'economy_rate'
            ]].values
            X_seq.append(seq_features)

            X_venue.append(group.loc[i, 'venue_encoded'])
            X_opponent.append(group.loc[i, 'opponent_encoded'])

            y.append(group.loc[i, 'bowling_points'])

    X_bowl = pd.DataFrame(
        np.array(X_seq).reshape(len(X_seq), -1),
        columns=[f"bowl_{i}_{j}" for i in range(sequence_length) for j in range(8)]
    )
    
    # X_bowl = pd.concat([
    #     X_seq_flattened,
    #     pd.DataFrame({'venue': X_venue, 'opponent': X_opponent})
    # ], axis=1)

    y_bowl = pd.DataFrame(y, columns=['bowling_points'])

    return X_bowl, y_bowl


def prepare_fielding_data(df: pd.DataFrame, label_encoders: dict, sequence_length=5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares features (X) and target (y) for predicting fielding points based on past performance.
    
    Args:
        df (pd.DataFrame): Input DataFrame with player fielding performance data.
        label_encoders (dict): Dictionary with fitted label encoders for 'venue' and 'team'.
        sequence_length (int): Number of past matches to consider for the sequence.
        
    Returns:
        tuple: 
            X_field (pd.DataFrame): Features DataFrame containing sequential and contextual features.
            y_field (pd.DataFrame): Target DataFrame for fielding points.
    """
    venue_encoder = label_encoders['venue_encoder']
    team_encoder = label_encoders['team_encoder']
    
    df['venue_encoded'] = venue_encoder.transform(df['venue'])
    df['opponent_encoded'] = team_encoder.transform(df['opponent_team'])  
    
    X_seq = []     
    X_venue = []   
    X_opponent = []  
    y = []         

    grouped = df.groupby('fielder')  

    for _, group in grouped:
        group = group.sort_values(by='match_id').reset_index(drop=True)
        
        for i in range(sequence_length, len(group)):  
            seq_features = group.loc[i-sequence_length:i-1, ['catches', 'run_outs', 'stumps']].values
            X_seq.append(seq_features)

            X_venue.append(group.loc[i, 'venue_encoded'])
            X_opponent.append(group.loc[i, 'opponent_encoded'])

            y.append(group.loc[i, 'fielding_points'])

    X_field = pd.DataFrame(
        np.array(X_seq).reshape(len(X_seq), -1),
        columns=[f"field_{i}_{j}" for i in range(sequence_length) for j in range(3)]
    )
    
    # X_field = pd.concat([
    #     X_seq_flattened,
    #     pd.DataFrame({'venue': X_venue, 'opponent': X_opponent})
    # ], axis=1)

    y_field = pd.DataFrame(y, columns=['fielding_points'])

    return X_field, y_field


def unique_players(bat_df, bowl_df, field_df):
    """
    Returns a sorted list of unique player names from the batting, bowling, and fielding data.

    Args:
        bat_df (pd.DataFrame): A DataFrame containing batting data, with a 'batter' column representing player names.
        bowl_df (pd.DataFrame): A DataFrame containing bowling data, with a 'bowler' column representing player names.
        field_df (pd.DataFrame): A DataFrame containing fielding data, with a 'fielder' column representing player names.

    Returns:
        list: A sorted list of unique player names found in all three DataFrames.
    
    """
    players_list = list(set(bat_df['batter'].unique()).union(
        set(bowl_df['bowler'].unique()).union(
            set(field_df['fielder'].unique())  
        )
    ))
    players_list.sort()
    return players_list




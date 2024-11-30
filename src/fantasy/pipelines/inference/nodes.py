"""
This is a boilerplate pipeline 'inference'
generated using Kedro 0.19.9
"""

import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import logging
import random
import pulp
from captum.attr import IntegratedGradients


import torch
import torch.nn as nn
from torchcp.regression.predictors import SplitPredictor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


def fetch_recent_batting_performances(bat_df, player_list):
    """
    Fetches the 5 most recent match performances for each player in the player_list.
    Pads with average or zeros if fewer than 5 performances are available.

    Args:
        bat_df (pd.DataFrame): DataFrame containing batting data with columns 'match_id', 'batter', 'runs', etc.
        player_list (list): List of player names whose performances are to be fetched.

    Returns:
        pd.DataFrame: DataFrame containing the 5 most recent performances for each player,
                      padded with average or zero where necessary.
    """
    all_player_performances = []

    for player in player_list:
        player_data = bat_df[bat_df['batter'] == player].sort_values(by='date', ascending=False)

        if len(player_data) < 5:
            avg_performance = player_data[['runs', 'fours', 'sixes', 'balls_faced']].mean()
            if player_data.empty:
                avg_performance = pd.Series([0] * len(avg_performance), index=avg_performance.index)
            player_data = pd.concat([player_data, pd.DataFrame([avg_performance] * (5 - len(player_data)), columns=avg_performance.index)], ignore_index=True)
        
        player_data = player_data.head(5)
        player_data['batter'] = player
        all_player_performances.append(player_data)

    final_df = pd.concat(all_player_performances, ignore_index=True)
    return final_df[['batter', 'runs', 'fours', 'sixes', 'balls_faced']]


def prepare_batting_tensor(df, sequence_length=5):
    """
    Converts player performance data into a tensor for model input.

    Args:
        df (pd.DataFrame): DataFrame containing player performances with columns 'batter', 'runs', 'fours', 'sixes', and 'balls_faced'.
        sequence_length (int): Number of matches (sequences) to consider for each player.

    Returns:
        torch.Tensor: Tensor of shape (num_players, sequence_length, 4), where 4 represents the features per match.
    """
    sequences = []

    grouped = df.groupby('batter')
    for player, group in grouped:
        if len(group) >= sequence_length:
            last_performances = group.tail(sequence_length)
            seq_features = last_performances[['runs', 'fours', 'sixes', 'balls_faced']].values
            sequences.append(seq_features)
        else:
            padded_performance = np.zeros((sequence_length, 4))
            last_performances = group[['runs', 'fours', 'sixes', 'balls_faced']].values
            padded_performance[:len(last_performances)] = last_performances
            sequences.append(padded_performance)

    sequences_array = np.array(sequences)
    sequences_tensor = torch.tensor(sequences_array, dtype=torch.float32)

    return sequences_tensor


def predict_batting_points(model, split_predictor, df, sequence_length=5, significance_level=0.1):
    """
    Predicts fantasy league points and provides conformal prediction intervals.

    Args:
        model (nn.Module): Trained LSTM model used for point prediction.
        split_predictor (SplitPredictor): Calibrated SplitPredictor for conformal prediction.
        df (pd.DataFrame): DataFrame containing player performance data with columns 'batter', 'runs', 'fours', 'sixes', and 'balls_faced'.
        sequence_length (int): Number of matches (sequences) to consider for each player.
        significance_level (float): Significance level for the conformal prediction (default: 0.1 for 90% confidence interval).

    Returns:
        tuple: A tuple containing:
            - A list of predicted fantasy points for each player.
            - A list of lower bounds for each player's prediction.
            - A list of upper bounds for each player's prediction.
    """
    X_tensor = prepare_batting_tensor(df, sequence_length=sequence_length)

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    intervals_array = split_predictor.predict(X_tensor).numpy()
    lower_bounds = intervals_array[:, 0, 0].tolist()
    upper_bounds = intervals_array[:, 0, 1].tolist()

    predicted_points_list = predictions.flatten().tolist()

    return predicted_points_list, lower_bounds, upper_bounds


def predict_batting_points_for_players(model, split_predictor, bat_df, player_list, sequence_length=5):
    """
    Predicts fantasy league points and conformal intervals for a list of players based on their recent performances.

    Args:
        model (nn.Module): Trained LSTM model used to predict fantasy points.
        split_predictor (SplitPredictor): Calibrated SplitPredictor for conformal prediction.
        bat_df (pd.DataFrame): DataFrame containing player performance data with columns like 'batter', 'runs', 'fours', 'sixes', 'balls_faced', and 'date'.
        player_list (list): List of player names whose batting points are to be predicted.
        sequence_length (int): Number of matches (sequences) to consider for each player.

    Returns:
        pd.DataFrame: DataFrame containing each player's name, predicted batting points, and conformal prediction intervals (lower and upper bounds).
    """
    recent_performances_df = fetch_recent_batting_performances(bat_df, player_list)
    predicted_points, lower, upper = predict_batting_points(model, split_predictor, recent_performances_df, sequence_length)

    result_df = pd.DataFrame({
        'player': player_list,
        'predicted_batting_points': predicted_points,
        'bat_lower_bounds': lower,
        'bat_upper_bounds': upper
    })

    return result_df


def fetch_recent_bowling_performances(bowl_df, player_list):
    """
    Fetches the 5 most recent match performances for each bowler in the list.
    If fewer than 5 performances exist, pads with the average of previous performances or zeros if no performance exists.

    Args:
        bowl_df (pd.DataFrame): A DataFrame containing the bowling data with columns like 'match_id', 'bowler', 'date', 'runs_conceded', 
                                'fours_conceded', 'sixes_conceded', 'wickets_taken', etc.
        player_list (list): A list of bowler names whose performances are to be fetched.

    Returns:
        pd.DataFrame: A DataFrame containing the 5 most recent performances of each bowler, padded with averages or zeros where necessary.
    """
    all_bowler_performances = []

    for player in player_list:
        player_data = bowl_df[bowl_df['bowler'] == player].sort_values(by='date', ascending=False)

        if len(player_data) < 5:
            avg_performance = player_data[['runs_conceded', 'fours_conceded', 'sixes_conceded',
                                           'wickets_taken', 'bowled_and_lbw', 'balls_bowled',
                                           'bowling_average', 'economy_rate']].mean()

            if player_data.empty:
                avg_performance = pd.Series([0] * len(avg_performance), index=avg_performance.index)

            player_data = pd.concat([player_data,
                                     pd.DataFrame([avg_performance] * (5 - len(player_data)),
                                                  columns=avg_performance.index)],
                                    ignore_index=True)

        player_data = player_data.head(5)
        player_data['bowler'] = player
        all_bowler_performances.append(player_data)

    final_df = pd.concat(all_bowler_performances, ignore_index=True)
    return final_df[['bowler', 'runs_conceded', 'fours_conceded', 'sixes_conceded',
                     'wickets_taken', 'bowled_and_lbw', 'balls_bowled', 'bowling_average', 'economy_rate']]


def prepare_bowling_tensor(df, sequence_length=5):
    """
    Converts each bowler's 5-match performances into a tensor with shape (num_players, sequence_length, num_features).

    Args:
        df (pd.DataFrame): DataFrame containing bowler performances with columns like 'bowler', 'runs_conceded',
                           'fours_conceded', 'sixes_conceded', 'wickets_taken', etc.
        sequence_length (int): The number of matches (sequences) to consider for each player.

    Returns:
        torch.Tensor: A tensor with shape (num_players, sequence_length, num_features).
    """
    sequences = []
    grouped = df.groupby('bowler')

    for player, group in grouped:
        if len(group) >= sequence_length:
            last_performances = group.tail(sequence_length)
            seq_features = last_performances[['runs_conceded', 'fours_conceded', 'sixes_conceded',
                                              'wickets_taken', 'bowled_and_lbw', 'balls_bowled',
                                              'bowling_average', 'economy_rate']].values
            sequences.append(seq_features)
        else:
            padded_performance = np.zeros((sequence_length, 8))
            last_performances = group[['runs_conceded', 'fours_conceded', 'sixes_conceded',
                                       'wickets_taken', 'bowled_and_lbw', 'balls_bowled',
                                       'bowling_average', 'economy_rate']].values
            padded_performance[:len(last_performances)] = last_performances
            sequences.append(padded_performance)

    sequences_array = np.array(sequences)
    sequences_tensor = torch.tensor(sequences_array, dtype=torch.float32)

    return sequences_tensor


def predict_bowling_points(model, split_predictor, df, sequence_length=5):
    """
    Predicts fantasy league points for each bowler based on their recent performances
    and provides conformal prediction intervals.

    Args:
        model (nn.Module): The trained LSTM model used to predict points.
        split_predictor (SplitPredictor): The calibrated SplitPredictor model used for conformal prediction.
        df (pd.DataFrame): DataFrame containing bowler performances.
        sequence_length (int): The number of matches (sequences) to consider for each player.

    Returns:
        tuple: A tuple containing two lists:
            - A list of predicted bowling points for each player.
            - A list of tuples, each containing the conformal prediction interval (lower bound, upper bound) for each player.
    """
    X_tensor = prepare_bowling_tensor(df, sequence_length=sequence_length)

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    intervals_array = split_predictor.predict(X_tensor).numpy()
    lower_bounds = intervals_array[:, 0, 0].tolist()
    upper_bounds = intervals_array[:, 0, 1].tolist()

    predicted_points_list = predictions.flatten().tolist()

    return predicted_points_list, lower_bounds, upper_bounds


def predict_bowling_points_for_players(model, split_predictor, bowl_df, player_list, sequence_length=5):
    """
    Predicts fantasy league points for a list of players based on their recent bowling performances.

    Args:
        model (nn.Module): The trained LSTM model used to predict points.
        split_predictor (SplitPredictor): The calibrated SplitPredictor model used for conformal prediction.
        bowl_df (pd.DataFrame): DataFrame containing player performances with relevant columns.
        player_list (list): A list of player names whose bowling points are to be predicted.
        sequence_length (int): The number of matches (sequences) to consider for each player.

    Returns:
        pd.DataFrame: A DataFrame containing each player's name, predicted bowling points, and conformal intervals.
    """
    recent_performances_df = fetch_recent_bowling_performances(bowl_df, player_list)

    predicted_points, lower_bounds, upper_bounds = predict_bowling_points(model, split_predictor, recent_performances_df, sequence_length)

    result_df = pd.DataFrame({
        'player': player_list,
        'predicted_bowling_points': predicted_points,
        'bowl_lower_bounds': lower_bounds,
        'bowl_upper_bounds': upper_bounds
    })

    return result_df



def fetch_recent_fielding_performances(field_df, player_list):
    """
    Fetches the 5 most recent fielding performances for each player in the list.
    If there are fewer than 5 performances, pads with the average of previous performances or zeros if no performance exists.
    
    Args:
        field_df (pd.DataFrame): A DataFrame containing fielding data with columns such as 'match_id', 'fielder', 'catches', 
                                 'run_outs', 'stumps', 'fielding_points', and 'date'.
        player_list (list): A list of fielder names whose performances are to be fetched.
    
    Returns:
        pd.DataFrame: A DataFrame containing the 5 most recent performances for each fielder, 
                      with additional columns for 'fielder', 'catches', 'run_outs', and 'stumps'.
                      If fewer than 5 performances exist, rows are padded with the average or zeros.
    """
    # Initialize an empty list to store the performance data for each fielder
    all_fielder_performances = []

    for player in player_list:
        # Filter the DataFrame for the current player's performances
        player_data = field_df[field_df['fielder'] == player].sort_values(by='date', ascending=False)
        
        # If the player has fewer than 5 performances, we will need to pad
        if len(player_data) < 5:
            # Get the average of the available performances
            avg_performance = player_data[['catches', 'run_outs', 'stumps', 'fielding_points']].mean()
            
            # If no data is available, fill with zeros
            if player_data.empty:
                avg_performance = pd.Series([0] * len(avg_performance), index=avg_performance.index)
            
            # Pad with average performance or 0s to make 5 entries
            player_data = pd.concat([player_data, pd.DataFrame([avg_performance] * (5 - len(player_data)), columns=avg_performance.index)], ignore_index=True)
        
        # If there are more than 5 performances, truncate to the 5 most recent
        player_data = player_data.head(5)
        
        # Add the player name and their performances to the list
        player_data['fielder'] = player
        all_fielder_performances.append(player_data)

    # Concatenate the list of performances for all fielders into one DataFrame
    final_df = pd.concat(all_fielder_performances, ignore_index=True)
    return final_df[['fielder', 'catches', 'run_outs', 'stumps']]


def prepare_fielding_tensor(df, sequence_length=5):
    """
    Converts each player's 5-match fielding performances into a tensor with shape (num_players, 5, 4).
    The tensor contains performance features like 'catches', 'run_outs', and 'stumps' for each match.
    
    Args:
        df (pd.DataFrame): DataFrame containing player performances with columns 'fielder', 'catches', 'run_outs', 
                           'stumps', etc.
        sequence_length (int): The number of matches (sequences) to consider for each player (default is 5).
        
    Returns:
        torch.Tensor: A tensor with shape (num_players, 5, 4), where 5 is the number of matches and 4 is the number of features 
                      (catches, run_outs, stumps) per match.
    """
    # Initialize a list to store the sequences
    sequences = []

    # Group by player
    grouped = df.groupby('fielder')

    for player, group in grouped:
        # Ensure the player has at least 'sequence_length' performances
        if len(group) >= sequence_length:
            # Select the last 'sequence_length' matches (5 matches in this case)
            last_performances = group.tail(sequence_length)
            
            # Extract the relevant features: catches, run_outs, stumps
            seq_features = last_performances[['catches', 'run_outs', 'stumps']].values
            
            # Append the sequence to the list
            sequences.append(seq_features)
        else:
            # If player has fewer than 'sequence_length' performances, pad with zeros
            padded_performance = np.zeros((sequence_length, 3))  # 5 matches, 3 features per match
            last_performances = group[['catches', 'run_outs', 'stumps']].values
            padded_performance[:len(last_performances)] = last_performances
            sequences.append(padded_performance)

    # Convert the list of sequences to a numpy array and then to a PyTorch tensor
    sequences_array = np.array(sequences)
    sequences_tensor = torch.tensor(sequences_array, dtype=torch.float32)

    return sequences_tensor


def predict_fielding_points(model, split_predictor, df, sequence_length=5):
    """
    Predicts fantasy league points for each player based on their recent fielding performances using a trained model.
    
    Args:
        model (nn.Module): The trained LSTM model used to predict points.
        split_predictor (object): A separate model or function for predicting confidence intervals for the predictions.
        df (pd.DataFrame): DataFrame containing player performances with columns 'fielder', 'catches', 'run_outs', 
                           'stumps', etc.
        sequence_length (int): The number of matches (sequences) to consider for each player (default is 5).
        
    Returns:
        tuple: A tuple containing:
            - predicted_points_list (list): A list of predicted fantasy points for each player.
            - lower_bounds (list): A list of lower confidence bounds for each prediction.
            - upper_bounds (list): A list of upper confidence bounds for each prediction.
    """
    
    # Step 1: Prepare the DataFrame for prediction (convert to tensors)
    X_tensor = prepare_fielding_tensor(df, sequence_length=sequence_length)

    # Step 2: Use the model to predict points for each player's performance sequence
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients during inference
        predictions = model(X_tensor).numpy()  # Get model predictions (as NumPy array)

    # Step 3: Use the split predictor to get confidence intervals
    intervals_array = split_predictor.predict(X_tensor).numpy()
    lower_bounds = intervals_array[:, 0, 0].tolist()
    upper_bounds = intervals_array[:, 0, 1].tolist()

    # Flatten the predictions for ease of use
    predicted_points_list = predictions.flatten().tolist()

    return predicted_points_list, lower_bounds, upper_bounds


def predict_fielding_points_for_players(model, split_predictor, field_df, player_list, sequence_length=5):
    """
    Predicts fantasy league points for a list of players based on their recent fielding performances.

    Args:
        model (nn.Module): The trained LSTM model used to predict points.
        split_predictor (object): A separate model or function for predicting confidence intervals for the predictions.
        field_df (pd.DataFrame): DataFrame containing player performances with columns 'fielder', 'catches', 
                                 'run_outs', 'stumps', 'date'.
        player_list (list): A list of player names whose fielding points are to be predicted.
        sequence_length (int): The number of matches (sequences) to consider for each player (default is 5).

    Returns:
        pd.DataFrame: A DataFrame containing each player's name, their predicted fielding points, 
                      and the confidence intervals (lower and upper bounds).
    """
    # Step 1: Fetch the 5 most recent performances for each player
    recent_performances_df = fetch_recent_fielding_performances(field_df, player_list)

    # Step 2: Predict fielding points for each player
    predicted_points, lower_bounds, upper_bounds = predict_fielding_points(model, split_predictor, recent_performances_df, sequence_length)

    # Step 3: Create a DataFrame with player names and predicted points
    result_df = pd.DataFrame({
        'player': player_list,
        'predicted_fielding_points': predicted_points,
        'field_lower_bounds': lower_bounds,
        'field_upper_bounds': upper_bounds
    })

    return result_df


def preprocess_input_df(input_df : pd.DataFrame):
    
    players_list = list(input_df['players'])
    logger = logging.getLogger(__name__)
    logger.info(f"players_list : {players_list}")
    return players_list


import logging
import random
import pulp
import pandas as pd

def best_team(bat_prediction: pd.DataFrame, bowl_prediction: pd.DataFrame, field_prediction: pd.DataFrame, budget: float = 100):
    """
    Solves an optimization problem to select the best fantasy cricket team based on predictions of batting, bowling,
    and fielding points, along with their respective upper and lower bounds, with random costs for each player.

    Args:
        bat_prediction (pd.DataFrame): DataFrame containing player names, predicted batting points, and lower and upper bounds.
        bowl_prediction (pd.DataFrame): DataFrame containing player names, predicted bowling points, and lower and upper bounds.
        field_prediction (pd.DataFrame): DataFrame containing player names, predicted fielding points, and lower and upper bounds.
        budget (float): The available budget for selecting the team.

    Returns:
        pd.DataFrame: A DataFrame containing the selected players, their roles, and their respective predicted points and bounds.
    """
    # Set up logging
    logger = logging.getLogger(__name__)

    # Combine all predictions into one DataFrame with lowercase columns
    combined_df = bat_prediction.merge(bowl_prediction, on='player', how='left') \
                                .merge(field_prediction, on='player', how='left')

    combined_df.rename(columns={
        'predicted_batting_points': 'batting_points',
        'predicted_bowling_points': 'bowling_points',
        'predicted_fielding_points': 'fielding_points',
        'bat_lower_bounds': 'bat_lower_bound',
        'bat_upper_bounds': 'bat_upper_bound',
        'bowl_lower_bounds': 'bowl_lower_bound',
        'bowl_upper_bounds': 'bowl_upper_bound',
        'field_lower_bounds': 'field_lower_bound',
        'field_upper_bounds': 'field_upper_bound'
    }, inplace=True)

    if 'cost' not in combined_df.columns:
        combined_df['cost'] = [random.randint(3, 8) for _ in range(len(combined_df))]

    logger.info(f"Combined Prediction DataFrame with Random Costs: \n{combined_df}")

    # Initialize optimization problem
    prob = pulp.LpProblem("Fantasy_Cricket_Team", pulp.LpMaximize)

    N = len(combined_df)

    # Define binary variables for each role and player
    batsman_selected = [pulp.LpVariable(f'batsman_selected_{i}', cat='Binary') for i in range(N)]
    bowler_selected = [pulp.LpVariable(f'bowler_selected_{i}', cat='Binary') for i in range(N)]
    allrounder_selected = [pulp.LpVariable(f'allrounder_selected_{i}', cat='Binary') for i in range(N)]
    
    selected = [batsman_selected[i] + bowler_selected[i] + allrounder_selected[i] for i in range(N)]

    # Ensure a player can be selected for only one role
    for i in range(N):
        prob += selected[i] <= 1

    # Objective function: maximize total points
    prob += pulp.lpSum(
        [
            batsman_selected[i] * (combined_df.loc[i, 'batting_points'] + combined_df.loc[i, 'fielding_points']) +
            bowler_selected[i] * (combined_df.loc[i, 'bowling_points'] + combined_df.loc[i, 'fielding_points']) +
            allrounder_selected[i] * (combined_df.loc[i, 'batting_points'] + combined_df.loc[i, 'bowling_points'] + combined_df.loc[i, 'fielding_points'])
            for i in range(N)
        ]
    )

    # Budget constraint
    prob += pulp.lpSum([selected[i] * combined_df.loc[i, 'cost'] for i in range(N)]) <= budget

    # Total number of players selected must be 11
    prob += pulp.lpSum(selected) == 11

    # Position-specific constraints
    prob += pulp.lpSum(batsman_selected) >= 3
    prob += pulp.lpSum(batsman_selected) <= 6

    prob += pulp.lpSum(bowler_selected) >= 3
    prob += pulp.lpSum(bowler_selected) <= 6

    prob += pulp.lpSum(allrounder_selected) >= 1
    prob += pulp.lpSum(allrounder_selected) <= 4

    # Solve the optimization problem
    prob.solve()
    logger.info(f"Status: {pulp.LpStatus[prob.status]}")

    # Prepare the result
    selected_players = []
    for i in range(N):
        if selected[i].value() == 1:
            role = "batsman" if batsman_selected[i].value() == 1 else \
                   "bowler" if bowler_selected[i].value() == 1 else \
                   "allrounder"
            selected_players.append({
                'player': combined_df.loc[i, 'player'],
                'role': role,
                'batting_points': combined_df.loc[i, 'batting_points'],
                'bowling_points': combined_df.loc[i, 'bowling_points'],
                'fielding_points': combined_df.loc[i, 'fielding_points'],
                'total_points': (
                    combined_df.loc[i, 'batting_points'] +
                    combined_df.loc[i, 'bowling_points'] +
                    combined_df.loc[i, 'fielding_points']
                ),
                'cost': combined_df.loc[i, 'cost'],
                'bat_lower_bound': combined_df.loc[i, 'bat_lower_bound'],
                'bat_upper_bound': combined_df.loc[i, 'bat_upper_bound'],
                'bowl_lower_bound': combined_df.loc[i, 'bowl_lower_bound'],
                'bowl_upper_bound': combined_df.loc[i, 'bowl_upper_bound'],
                'field_lower_bound': combined_df.loc[i, 'field_lower_bound'],
                'field_upper_bound': combined_df.loc[i, 'field_upper_bound']
            })

    result_df = pd.DataFrame(selected_players)
    logger.info(f"Selected Team DataFrame: \n{result_df}")
    return result_df




    
    
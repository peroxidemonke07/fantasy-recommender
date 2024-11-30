"""
This is a boilerplate pipeline 'inference'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_input_df,
            inputs=["instance"],
            outputs="players_list",
            name="get_players_list"
        ),
        node(
            func=predict_batting_points_for_players,
            inputs=["lstm_model_bat","split_predictor_bat","batsman_scorecard","players_list"],
            outputs="bat_prediction",
            name="predict_batting_points_for_players"
        ),
        node(
            func=predict_bowling_points_for_players,
            inputs=["lstm_model_bowl","split_predictor_bowl","bowler_scorecard","players_list"],
            outputs="bowl_prediction",
            name="predict_bowling_points_for_players"
        ),
        node(
            func=predict_fielding_points_for_players,
            inputs=["lstm_model_field","split_predictor_field","fielder_scorecard","players_list"],
            outputs="field_prediction",
            name="predict_fielding_points_for_players"
        ),
        node(
            func=best_team,
            inputs=["bat_prediction","bowl_prediction","field_prediction"],
            outputs="inference_output",
            name="best_team"
        )
    ])

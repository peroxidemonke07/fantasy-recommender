"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=calculate_batting_scorecard,
            inputs=["deliveries","matches"],
            outputs="batsman_scorecard",
            name="calculate_batting_scorecard"
        ),
        node(
            func=calculate_bowling_scorecard,
            inputs=["deliveries","matches"],
            outputs="bowler_scorecard",
            name="calculate_bowling_scorecard"
        ),
        node(
            func=calculate_fielding_scorecard,
            inputs=["deliveries","matches"],
            outputs="fielder_scorecard",
            name="calculate_fielding_scorecard"
        ),
        node(
            func=encode_teams_and_venues,
            inputs="matches",
            outputs="label_encoders",
            name="encode_teams_and_venues"
        ),
        node(
            func=prepare_batting_data,
            inputs=["batsman_scorecard","label_encoders","params:batsman_sequence_length"],
            outputs=["X_bat","y_bat"],
            name="prepare_batting_data"
        ),
        node(
            func=prepare_bowling_data,
            inputs=["bowler_scorecard","label_encoders","params:bowler_sequence_length"],
            outputs=["X_bowl","y_bowl"],
            name="prepare_bowling_data"
        ),
        node(
            func=prepare_fielding_data,
            inputs=["fielder_scorecard","label_encoders","params:fielder_sequence_length"],
            outputs=["X_field","y_field"],
            name="prepare_fielding_data"
        ),
        node(
            func=unique_players,
            inputs=["batsman_scorecard","bowler_scorecard","fielder_scorecard"],
            outputs="all_players_list",
            name="get_unique_players"
        )
    ])

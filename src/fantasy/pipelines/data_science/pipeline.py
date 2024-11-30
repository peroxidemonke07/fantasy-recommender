"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline,node
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_train_test,
            inputs=["X_bat", "y_bat", "params:train_test_split_params_bat"],
            outputs=[
                "X_train_bat",
                "X_test_bat",
                "X_cali_bat",
                "y_train_bat",
                "y_test_bat",
                "y_cali_bat"
            ],
            name="train_test_split_bat"
        ),
        node(
            func=split_train_test,
            inputs=["X_bowl", "y_bowl", "params:train_test_split_params_bowl"],
            outputs=[
                "X_train_bowl",
                "X_test_bowl",
                "X_cali_bowl",
                "y_train_bowl",
                "y_test_bowl",
                "y_cali_bowl"
            ],
            name="train_test_split_bowl"
        ),
        node(
            func=split_train_test,
            inputs=["X_field", "y_field", "params:train_test_split_params_field"],
            outputs=[
                "X_train_field",
                "X_test_field",
                "X_cali_field",
                "y_train_field",
                "y_test_field",
                "y_cali_field"
            ],
            name="train_test_split_field"
        ),
        node(
            func=train_lstm_model,
            inputs=["X_train_bat", "y_train_bat", 
                    "params:seq_context_split_params_bat", 
                    "params:lstm_model_params_bat", 
                    "params:train_params_bat"],
            outputs="lstm_model_bat",
            name="train_lstm_model_bat"
        ),
        node(
            func=train_lstm_model,
            inputs=["X_train_bowl", "y_train_bowl", 
                    "params:seq_context_split_params_bowl", 
                    "params:lstm_model_params_bowl", 
                    "params:train_params_bowl"],
            outputs="lstm_model_bowl",
            name="train_lstm_model_bowl"
        ),
        node(
            func=train_lstm_model,
            inputs=["X_train_field", "y_train_field", 
                    "params:seq_context_split_params_field", 
                    "params:lstm_model_params_field", 
                    "params:train_params_field"],
            outputs="lstm_model_field",
            name="train_lstm_model_field"
        ),
        node(
            func=calibrate_lstm_model,
            inputs=[
                "lstm_model_bat",
                "X_cali_bat",
                "y_cali_bat",
                "X_test_bat",
                "y_test_bat",
                "params:calibrate_params_bat"
            ],
            outputs="split_predictor_bat",
            name="calibrate_lstm_model_bat"
        ),
        node(
            func=calibrate_lstm_model,
            inputs=[
                "lstm_model_bowl",
                "X_cali_bowl",
                "y_cali_bowl",
                "X_test_bowl",
                "y_test_bowl",
                "params:calibrate_params_bowl"
            ],
            outputs="split_predictor_bowl",
            name="calibrate_lstm_model_bowl"
        ),
        node(
            func=calibrate_lstm_model,
            inputs=[
                "lstm_model_field",
                "X_cali_field",
                "y_cali_field",
                "X_test_field",
                "y_test_field",
                "params:calibrate_params_field"
            ],
            outputs="split_predictor_field",
            name="calibrate_lstm_model_field"
        ),


    ])

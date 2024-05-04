from steps import load_data, drop_cols, datetime_col, liter_col, pred_col_Tout, xy_split, train_test_splitter, model_evaluation, model_prediction, model_selection_Temp, model_training, data_concat,best_model
from zenml import pipeline
import pandas as pd
from zenml.client import Client
client = Client()

@pipeline(enable_cache=False)
def pipeline_data_etl():
    data = load_data()
    data = drop_cols(data)
    data = datetime_col(data)
    data = liter_col(data)
    return data

@pipeline(enable_cache=False)
def feature_engineering_Temp(df):
    df_t = pred_col_Tout(df)
    x, y = xy_split(df_t)
    x_train, x_test, y_train, y_test = train_test_splitter(x, y)
    return x_train, x_test, y_train, y_test, x

@pipeline(enable_cache=False)
def model_training_Temp(x_train, x_test, y_train, y_test, model_name):
    model = model_selection_Temp(model_name=model_name)
    trained_model = model_training(model, x_train, y_train)
    predictions = model_prediction(trained_model, x_test)
    rmse = model_evaluation(predictions, y_test)
    return trained_model, rmse

@pipeline(enable_cache=False)
def feature_engineering_Wout(trained_model, x):
    # model_id = client.get_artifact_version('Trained model').load()
    x_id = client.get_artifact_version('x-features').load()
    predictions_T = model_prediction(trained_model, x_id)
    data_w = data_concat(x, predictions_T)
    water_out = client.get_artifact_version('Water out [L]').load()
    x_train, x_test, y_train, y_test = train_test_splitter(data_w, water_out)
    return x_train, x_test, y_train, y_test

@pipeline(enable_cache=False)
def model_training_Wout(x_train, x_test, y_train, y_test, model_name):
    model = model_selection_Temp(model_name=model_name)
    trained_model = model_training(model, x_train, y_train)
    predictions = model_prediction(trained_model, x_test)
    rmse = model_evaluation(predictions, y_test)
    return rmse

# main pipeline
@pipeline(enable_cache=False, name='SWH Cascade ML Pipeline')
def main_pipeline():
    data = pipeline_data_etl()
    x_train, x_test, y_train, y_test, x = feature_engineering_Temp(data)
    model_T_rfr, rmse_rfr = model_training_Temp(x_train, x_test, y_train, y_test, model_name='RFR')
    model_T_abr, rmse_abr = model_training_Temp(x_train, x_test, y_train, y_test, model_name='ABR')
    model = best_model(model_T_abr, model_T_rfr, rmse_abr, rmse_rfr)
    x_train_, x_test_, y_train_, y_test_ = feature_engineering_Wout(model, x)
    rmse_w_rfr = model_training_Wout(x_train_, x_test_, y_train_, y_test_, model_name='RFR')
    rmse_w_abr = model_training_Wout(x_train_, x_test_, y_train_, y_test_, model_name='ABR')


    
import pandas as pd
import numpy as np
from zenml import step, pipeline
from typing_extensions import Annotated
from typing import Tuple, Any
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

#loading data
@step
def load_data()->Annotated[pd.DataFrame, 'loaded raw data']:
    df_raw = pd.read_csv('data/Thermosyphone_mode_60_degree.csv', delimiter=';')
    df_raw.drop(df_raw.index[0], inplace=True)
    return df_raw

@step
def drop_cols(df: pd.DataFrame) -> Annotated[pd.DataFrame, 'selected features']:
    df.drop(columns=['mode', 'effi_thermosy', 'effi_forced', 'C', 'f', 'ha', 'v', 'Kb',
                     'beta [degree]', 'N', 'Tau', 'alpha', 'Ub [W/m2 K]', 'Ue [W/m2 K]', 'Fr',
                     'effi_thermosy', 'effi_forced', 'Ta [K] ', 'Tin [K]', 'Tout [K]', 'Tplate [K]'],
            inplace=True, axis=1)
    return df

@step
def datetime_col(df:pd.DataFrame)->Annotated[pd.DataFrame, 'TIME col treatment']:
    df['TIME'] = pd.to_datetime(df['time'])
    # df = df.set_index(df['TIME'])
    df.drop(columns=['time'], inplace=True)
    return df

@step
def liter_col(df:pd.DataFrame)->Annotated[pd.DataFrame, 'Water Output Calculation']:
    time_interval = 10 * 60 # seconds
    df['l/sec'] = df['m_dot [kg/sec]']
    df['l_out'] = df['l/sec'] * (time_interval)
    df['l_out_cumsum'] = df['l_out'].cumsum()
    return df

@step
def pred_col_Tout(df:pd.DataFrame)->Annotated[pd.DataFrame, 'selected feature for ML-i']:
    # T_out = df.pop('Tout [C]')
    l_out = df.pop('l_out_cumsum')
    df.drop(columns=['l_out', 'l/sec', 'TIME'], inplace=True)
    return df

@step
def xy_split(df:pd.DataFrame)->Tuple[Annotated[pd.DataFrame, 'x-features'], Annotated[pd.Series, 'y-target_T-out']]:
    y_true = df.pop('Tout [C]')
    y_true = pd.Series(y_true)
    return df, y_true

@step
def train_test_splitter(x:pd.DataFrame, y:pd.Series)->Tuple[Annotated[pd.DataFrame, 'x_train'],
                                                            Annotated[pd.DataFrame, 'x_test'],
                                                            Annotated[pd.Series, 'y_train'],
                                                            Annotated[pd.Series, 'y_test']]:
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    return x_train, x_test, y_train, y_test

@step
def model_selection_Temp(model_name:str)->Annotated[RegressorMixin, 'Regression model selection']:
    if model_name.lower() == "rfr":
        model = RandomForestRegressor()
    elif model_name.lower() == 'abr':
        model = AdaBoostRegressor()
    else:
        raise ValueError
    return model

@step
def model_training(model:RegressorMixin, x_train:pd.DataFrame, y_train:pd.Series)->Annotated[RegressorMixin, 'Trained model']:
    model.fit(x_train, y_train)
    return model

@step
def model_prediction(model:RegressorMixin, x_test:pd.DataFrame)->Annotated[pd.Series, 'Predictions']:
    predictions = model.predict(x_test)
    predictions = pd.Series(predictions)
    return predictions

@step
def model_evaluation(y_pred:pd.Series, y_test:pd.Series)->Annotated[float, 'RMSE']:
    accu = mean_squared_error(y_pred=y_pred, y_true=y_test)
    print(f"RMSE: {accu}")
    return accu

@step
def data_concat(x_:pd.DataFrame, prediction:pd.Series)->Annotated[pd.DataFrame, 'Combined Data']:
    df = pd.concat([x_, pd.DataFrame(prediction)], axis=1, ignore_index=True)
    df.drop(index=0, axis=0, inplace=True)
    df.fillna(0, inplace=True)
    return df

@step
def best_model(model1:RegressorMixin, model2:RegressorMixin, rmse1:float, rmse2:float)-> Annotated[RegressorMixin, 'Best model for T-out']:
    if rmse1 > rmse2:
        return model2
    else:
        return model1

# def data_pipeline():
#     df_raw = load_data()
#     df = drop_cols(df_raw)
#     df = datetime_col(df)
#     df = liter_col(df)  # with all features
#     return df

# def data_pipeline_temp():
#     df = data_pipeline()
#     df_t = pred_col_Tout(df)
#     x, y = xy_split(df_t)
#     x_train, x_test, y_train, y_test = train_test_splitter(x, y)
#     return x_train, x_test, y_train, y_test, df


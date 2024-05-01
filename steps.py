import pandas as pd
import numpy as np
from zenml import step, pipeline
from typing_extensions import Annotated
from typing import Tuple, Any
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

#loading data
@step
def load_data()->Annotated[pd.DataFrame, 'raw data']:
    df_raw = pd.read_csv('data/Thermosyphone_mode_60_degree.csv', delimiter=';')
    df_raw.drop(df_raw.index[0], inplace=True)
    return df_raw

@step
def drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['mode', 'effi_thermosy', 'effi_forced', 'C', 'f', 'ha', 'v', 'Kb',
                     'beta [degree]', 'N', 'Tau', 'alpha', 'Ub [W/m2 K]', 'Ue [W/m2 K]', 'Fr',
                     'effi_thermosy', 'effi_forced', 'Ta [K]', 'Tin [K]', 'Tout [K]', 'Tplate [K]'],
            inplace=True)
    return df

@step
def datetime_col(df:pd.DataFrame)->Annotated[pd.DataFrame, 'TIME col']:
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


def pred_col_Tout(df:pd.DataFrame):
    # T_out = df.pop('Tout [C]')
    # l_out = df.pop('l_out_cumsum')
    df.drop(columns=['l_out', 'l/sec', 'l_out_cumsum'], inplace=True)
    return df

def xy_split(df:pd.DataFrame):
    y_true = df.pop('Tout [C]')
    # df.drop(columns='TIME', inplace=True)
    return df, y_true

def train_test_splitter(x:pd.DataFrame, y:pd.DataFrame):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42)
    return x_train, x_test, y_train, y_test

def data_pipeline():
    df_raw = load_data()
    df = drop_cols(df_raw)
    df = datetime_col(df)
    df = liter_col(df)  # with all features
    return df

def data_pipeline_temp():
    df = data_pipeline()
    df_t = pred_col_Tout(df)
    x, y = xy_split(df_t)
    x_train, x_test, y_train, y_test = train_test_splitter(x, y)
    return x_train, x_test, y_train, y_test, df


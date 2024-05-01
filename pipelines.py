from steps import load_data, drop_cols, datetime_col, liter_col
from zenml import pipeline
import pandas as pd

@pipeline(enable_cache=False)
def pipeline_data_etl()->pd.DataFrame:
    data = load_data()
    data = datetime_col(data)
    data = liter_col(data)
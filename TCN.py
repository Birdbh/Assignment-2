import pandas as pd
import matplotlib.pyplot as plt

import os
import datetime as dt

from darts.timeseries import TimeSeries as DTS
from darts.models import TCNModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries

from darts.metrics import mape, mae, mse, rmse
from darts import concatenate

def function(train_load_transformed, allCov, val_load_transformed,series_load_transformed, backtest_start_date, train_val_load_transformed):
    #For more info: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html
    model_en = TCNModel(
        input_chunk_length=72, # Try testing different values eg 120
        output_chunk_length=24,
        n_epochs=5, # 50 epochs will take 15-20 min; default is 100 epochs
        dropout=0.1,
        dilation_base=2, # Try other values. Preferably below the kernal size
        weight_norm=True,
        kernel_size=3, #Try updating this value for eg try 8
        num_filters=4, #Try updating this value for eg try 20
        nr_epochs_val_period=1, 
        random_state=0,
    )
    model_en.fit(
        series=train_load_transformed,
        past_covariates=allCov,
        val_series=val_load_transformed,
        val_past_covariates=allCov,
        verbose=False,    
    )
    #For more info: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html#darts.models.forecasting.tcn_model.TCNModel.backtest
    backtest_start_date = pd.to_datetime('20220601')
    backtest_en = model_en.historical_forecasts(
        series=series_load_transformed,
        past_covariates=allCov,
        start=backtest_start_date,
        forecast_horizon=24,
        stride=1,
        retrain=False,
        verbose=False,
    )
    
    series_load_transformed[backtest_start_date:].plot(label="Actual values")
    backtest_en.plot(label="24-Hour historic forecast (Backtest)")
    plt.legend()

    print("MAE = {:.4f}".format(mae(backtest_en, series_load_transformed)))
    print("MAPE = {:.2f}%".format(mape(backtest_en, series_load_transformed)))
    print("RMSE = {:.4f}".format(rmse(backtest_en, series_load_transformed)))
    print("MSE = {:.4f}".format(mse(backtest_en, series_load_transformed)))

    #  After tunning the hyper parameters test the model
    prediction_transformed = model_en.predict(48, series=train_val_load_transformed, past_covariates=allCov)

    scaler = Scaler()
    prediction=scaler.inverse_transform(prediction_transformed).pd_dataframe()
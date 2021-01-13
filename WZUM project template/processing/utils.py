from typing import Tuple
import pickle
import pandas as pd
from pathlib import Path


pd.options.mode.chained_assignment = None  # default='warn'


def perform_processing(
        temperature: pd.DataFrame,
        target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,
        serial_number_for_prediction: str
) -> Tuple[float, float]:
    df_temp = temperature[temperature['serialNumber'] == serial_number_for_prediction]

    df_temp.rename(columns={'value': 'temp'}, inplace=True)
    target_temperature.rename(columns={'value': 'target_temp'}, inplace=True)
    valve_level.rename(columns={'value': 'valve'}, inplace=True)

    df_combined = pd.concat([df_temp, target_temperature, valve_level])
    df_combined = df_combined.sort_index()

    last_reading = df_combined.tail(1)
    df_combined = df_combined.drop(df_combined.tail(1).index) # drop last n rows
    last_reading = pd.DataFrame({'temp': last_reading.iloc[-1].temp,
                                 'unit': last_reading.iloc[-1].unit,
                                 'serialNumber': last_reading.iloc[-1].serialNumber,
                                 'target_temp': last_reading.iloc[-1].target_temp,
                                 'valve': last_reading.iloc[-1].valve},
                                index=pd.to_datetime(last_reading.index - pd.Timedelta(seconds=1)))
    df_combined = pd.concat([df_combined, last_reading])

    df_combined = df_combined.resample(pd.Timedelta(minutes=5), label='right').mean().fillna(method='ffill')

    df_combined['temp_last'] = df_combined['temp'].shift(1)
    df_combined['temp_2nd_last'] = df_combined['temp'].shift(2)
    df_combined['temp_3rd_last'] = df_combined['temp'].shift(3)
    df_combined['temp_4th_last'] = df_combined['temp'].shift(4)

    df_combined['valve_last'] = df_combined['valve'].shift(1)
    df_combined['valve_2nd_last'] = df_combined['valve'].shift(2)
    df_combined['valve_3rd_last'] = df_combined['valve'].shift(3)
    df_combined['valve_4th_last'] = df_combined['valve'].shift(4)

    df_combined['last_temp_reading'] = df_temp.iloc[-1]['temp']
    df_combined['2ndlast_temp_reading'] = df_temp.iloc[-2]['temp']
    df_combined['last_valve_reading'] = valve_level.iloc[-1]['valve']
    df_combined['2ndlast_valve_reading'] = valve_level.iloc[-2]['valve']

    with open(Path('dane/reg_temp.p'), 'rb') as reg_temp_file:
        reg_temp = pickle.load(reg_temp_file)

    with open(Path('dane/reg_valve.p'), 'rb') as reg_valve_file:
        reg_valve = pickle.load(reg_valve_file)

    with open(Path('dane/scaler.p'), 'rb') as s_file:
        scaler = pickle.load(s_file)

    features = ['temp', 'target_temp', 'valve', 'temp_last', 'temp_2nd_last', 'temp_3rd_last',
                'temp_4th_last', 'valve_last', 'valve_2nd_last', 'valve_3rd_last', 'valve_4th_last',
                'last_temp_reading', '2ndlast_temp_reading', 'last_valve_reading', '2ndlast_valve_reading']
    # features = ['temp', 'target_temp', 'valve', 'temp_last', 'valve_last', 'valve_2nd_last', 'valve_3rd_last', 'valve_4th_last']

    X = df_combined[features].to_numpy()[-2:]
    X = scaler.transform(X)
    y_temp = reg_temp.predict(X)[-1]
    y_valve = reg_valve.predict(X)[-1]

    # return df_temp.temp[-1], valve_level.valve[-1]
    return y_temp, y_valve


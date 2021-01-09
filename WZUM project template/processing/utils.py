from typing import Tuple
import pickle
import pandas as pd


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

    last_reading = pd.DataFrame({'target_temp': target_temperature.iloc[-1].target_temp,
                                 'temp': df_temp.iloc[-1].temp,
                                 'valve': valve_level.iloc[-1].valve},
                                index=pd.to_datetime(df_temp.tail(1).index.ceil('15min')))

    df_combined = pd.concat([df_combined, last_reading])

    df_combined = df_combined.resample(pd.Timedelta(minutes=5)).mean().fillna(method='ffill')

    df_combined['temp_last'] = df_combined['temp'].shift(1)
    df_combined['temp_2nd_last'] = df_combined['temp'].shift(2)
    df_combined['temp_3rd_last'] = df_combined['temp'].shift(3)
    df_combined['temp_4th_last'] = df_combined['temp'].shift(4)

    df_combined['valve_last'] = df_combined['valve'].shift(1)
    df_combined['valve_2nd_last'] = df_combined['valve'].shift(2)
    df_combined['valve_3rd_last'] = df_combined['valve'].shift(3)
    df_combined['valve_4th_last'] = df_combined['valve'].shift(4)

    with open('/home/kamil/Pulpit/PUT/WZUM/reg_temp.p', 'rb') as reg_temp_file:
        reg_temp = pickle.load(reg_temp_file)

    with open('/home/kamil/Pulpit/PUT/WZUM/reg_valve.p', 'rb') as reg_valve_file:
        reg_valve = pickle.load(reg_valve_file)

    with open('/home/kamil/Pulpit/PUT/WZUM/scaler.p', 'rb') as s_file:
        scaler = pickle.load(s_file)

    features = ['temp', 'target_temp', 'valve', 'temp_last', 'temp_2nd_last', 'temp_3rd_last',
                'temp_4th_last', 'valve_last', 'valve_2nd_last', 'valve_3rd_last', 'valve_4th_last']
    # features = ['temp', 'target_temp', 'valve', 'temp_last', 'valve_last', 'valve_2nd_last', 'valve_3rd_last', 'valve_4th_last']

    X = df_combined[features].to_numpy()[-5:]
    X = scaler.transform(X)
    y_temp = reg_temp.predict(X)[-1]
    y_valve = reg_valve.predict(X)[-1]

    return y_temp, y_valve
    # return df_temp.temp[-1] #+ 0.1 * (valve_level.valve[-1]/100) * target_temperature.target_temp[-1]

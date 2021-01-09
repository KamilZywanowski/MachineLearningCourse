import pandas as pd
import pickle
import time

pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_columns = None
pd.set_option('expand_frame_repr', False)


def perform_processing(
        gt,
        temperature: pd.DataFrame,
        target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,
        serial_number_for_prediction: str
) -> float:
    f = open("gt_october_5.csv", "a+")
    f.write(f'{gt.name - pd.DateOffset(minutes=15)}, {gt.temperature},  {gt.valve_level}\n')
    f.close()
    print(f'{gt.name - pd.DateOffset(minutes=15)}, {gt.temperature}, {gt.valve_level}')

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

    df_combined.iloc[-1:].to_csv('train_october_5.csv', mode='a', index=True, header=False)
    print(df_combined.tail(1))
    # time.sleep(1)
    print()
    return 0, 0

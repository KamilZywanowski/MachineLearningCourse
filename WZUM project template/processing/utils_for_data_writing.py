import pandas as pd
import pickle

pd.options.mode.chained_assignment = None  # default='warn'

def perform_processing(
        gt,
        temperature: pd.DataFrame,
        target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,
        serial_number_for_prediction: str
) -> float:
    f = open("gt_october.csv", "a+")
    f.write(f'{gt.name - pd.DateOffset(minutes=15)}, {gt.value}\n')
    print(gt.name, gt.value)
    return 0

    df_temp = temperature[temperature['serialNumber'] == serial_number_for_prediction]

    df_temp.rename(columns={'value': 'temp'}, inplace=True)
    target_temperature.rename(columns={'value': 'target_temp'}, inplace=True)
    valve_level.rename(columns={'value': 'valve'}, inplace=True)

    df_combined = pd.concat([df_temp, target_temperature, valve_level])
    df_combined = df_combined.sort_index()

    last_reading = pd.DataFrame({'target_temp': target_temperature.iloc[-1].target_temp,
                                 'temp': df_temp.iloc[-1].temp,
                                 'valve': valve_level.iloc[-1].valve},
                                index=pd.to_datetime(df_combined.tail(1).index + pd.Timedelta(minutes=5)))
    df_combined = pd.concat([df_combined, last_reading])

    df_combined = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')

    # TODO prepare training data with this script. Run on all data, save ftrs
    df_combined['temp_last'] = df_combined['temp'].shift(1)
    df_combined['temp_2nd_last'] = df_combined['temp'].shift(2)
    df_combined['temp_3rd_last'] = df_combined['temp'].shift(3)
    df_combined['temp_4th_last'] = df_combined['temp'].shift(4)

    df_combined.iloc[-1:].to_csv('train_march.csv', mode='a', index=True, header=False)
    return df_temp.temp[-1]

    features = ['temp', 'temp_last', 'temp_2nd_last', 'temp_3rd_last', 'target_temp', 'valve']
    # print(df_combined.tail(1))
    X = df_combined[features].to_numpy()[-5:]

    with open('/home/kamil/Pulpit/PUT/WZUM/regressor.p', 'rb') as reg_file:
        regressor = pickle.load(reg_file)

    with open('/home/kamil/Pulpit/PUT/WZUM/scaler.p', 'rb') as s_file:
        scaler = pickle.load(s_file)

    # X = scaler.transform(X)
    y = regressor.predict(X)
    # print(X[-1])
    # print(y[-1])
    global zz
    zz += 1
    print(zz)
    return y[-1]
    # return df_temp.temp[-1] #+ 0.1 * (valve_level.valve[-1]/100) * target_temperature.target_temp[-1]

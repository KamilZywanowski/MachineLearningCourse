import pandas as pd
import random
import json
import matplotlib.pyplot as plt
from sklearn import metrics, ensemble


def _plot_one_against_original_data(df_original: pd.DataFrame, df_resampled: pd.DataFrame):...


def resample_comparison():...


def read_temp_mid_sn() -> int:
    with open('data/additional_info.json') as f:
        additional_data = json.load(f)

    devices = additional_data['offices']['office_1']['devices']
    sn_temp_mid = [d['serialNumber'] for d in devices if d['description'] == 'temperature_middle'][0]
    return sn_temp_mid


def project_check_data():
    sn_temp_mid = read_temp_mid_sn()
    df_temp = pd.read_csv('data/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv')
    df_temp.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df_temp.rename(columns={'value': 'temp'}, inplace=True)
    df_temp['time'] = pd.to_datetime((df_temp['time']))
    df_temp.drop(columns=['unit'], inplace=True)
    df_temp = df_temp[df_temp['serialNumber'] == sn_temp_mid]
    df_temp.set_index('time', inplace=True)
    
    df_target_temp = pd.read_csv('data/office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv')
    df_target_temp.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df_target_temp.rename(columns={'value': 'target_temp'}, inplace=True)
    df_target_temp['time'] = pd.to_datetime((df_target_temp['time']))
    df_target_temp.drop(columns=['unit'], inplace=True)
    df_target_temp.set_index('time', inplace=True)

    df_valve_lvl = pd.read_csv('data/office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv')
    df_valve_lvl.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df_valve_lvl.rename(columns={'value': 'valve'}, inplace=True)
    df_valve_lvl['time'] = pd.to_datetime((df_valve_lvl['time']))
    df_valve_lvl.drop(columns=['unit'], inplace=True)
    df_valve_lvl.set_index('time', inplace=True)

    # fig, ax1 = plt.subplots()
    #
    # color = 'tab:red'
    # ax1.plot(df_temp.index, df_temp.temp, color='red')
    # ax1.plot(df_target_temp.index, df_target_temp.target_temp, color='green')
    # ax1.tick_params(axis='y', labelcolor=color)
    #
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # color = 'tab:blue'
    # ax2.plot(df_valve_lvl.index, df_valve_lvl.valve, color='blue')
    # ax2.tick_params(axis='y', labelcolor=color)
    #
    # # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # # plt.show()

    df_combined = pd.concat([df_temp, df_target_temp, df_valve_lvl])
    df_combined.sort_index(inplace=True)

    df_combined = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')

    df_combined['temp_last'] = df_combined['temp'].shift(1)
    df_combined['temp_gt'] = df_combined['temp'].shift(-1)

    mask_test = (df_combined.index > '2020-10-27') & (df_combined.index <= '2020-10-28')
    df_test = df_combined.loc[mask_test]
    # df_test.plot()
    # plt.show()

    mask_train = (df_combined.index < '2020-10-27') # | (df_combined.index > '2020-10-28')
    df_train = df_combined.loc[mask_train]
    # df_train.plot()
    # plt.show()

    X_train = df_train[['temp', 'valve']].to_numpy()[1:-1]
    y_train = df_train['temp_gt'].to_numpy()[1:-1]

    X_test = df_test[['temp', 'valve']].to_numpy()
    y_test = df_test['temp_gt'].to_numpy()[1:-1]
    y_last = df_test['temp_last'].to_numpy()[1:-1]

    reg_rf = ensemble.RandomForestRegressor(random_state=42)
    reg_rf.fit(X_train, y_train)
    y_rf = reg_rf.predict(X_test)
    df_test['temp_predicted'] = y_rf.tolist()

    print(f'mae base: {metrics.mean_absolute_error(y_test, y_last)}')
    print(f'mae rf: {metrics.mean_absolute_error(y_test, y_rf[1:-1])}')
    print(f'mse base: {metrics.mean_squared_error(y_test, y_last)}')
    print(f'mse rf: {metrics.mean_squared_error(y_test, y_rf[1:-1])}')

    # print(df_combined.head(5))
    # print(df_combined.tail(5))
    df_test.plot()
    plt.show()




def main():
    random.seed(42)
    pd.options.display.max_columns = None
    resample_comparison()

    project_check_data()


if __name__ == '__main__':
    main()
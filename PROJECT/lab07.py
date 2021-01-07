import pandas as pd
import random
import json
import matplotlib.pyplot as plt
from sklearn import metrics, ensemble, linear_model, neural_network
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle


def load_files(period):
    sn_temp_mid = read_temp_mid_sn()
    df_temp = pd.read_csv(f'data/office_1_temperature_supply_points_data_{period}.csv')
    df_temp.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df_temp.rename(columns={'value': 'temp'}, inplace=True)
    df_temp['time'] = pd.to_datetime((df_temp['time']))
    df_temp.drop(columns=['unit'], inplace=True)
    df_temp = df_temp[df_temp['serialNumber'] == sn_temp_mid]
    df_temp.set_index('time', inplace=True)

    df_target_temp = pd.read_csv(f'data/office_1_targetTemperature_supply_points_data_{period}.csv')
    df_target_temp.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df_target_temp.rename(columns={'value': 'target_temp'}, inplace=True)
    df_target_temp['time'] = pd.to_datetime((df_target_temp['time']))
    df_target_temp.drop(columns=['unit'], inplace=True)
    df_target_temp.set_index('time', inplace=True)

    df_valve_lvl = pd.read_csv(f'data/office_1_valveLevel_supply_points_data_{period}.csv')
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
    return df_temp, df_target_temp, df_valve_lvl


def read_temp_mid_sn() -> int:
    with open('data/additional_info.json') as f:
        additional_data = json.load(f)

    devices = additional_data['offices']['office_1']['devices']
    sn_temp_mid = [d['serialNumber'] for d in devices if d['description'] == 'temperature_middle'][0]
    return sn_temp_mid


def combine_get_ftrs(period):

    df_temp, df_target_temp, df_valve_lvl = load_files(period)
    df_combined = pd.concat([df_temp, df_target_temp, df_valve_lvl])
    df_combined = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')

    df_combined['temp_gt'] = df_combined['temp'].shift(-1)

    df_combined['temp_last'] = df_combined['temp'].shift(1)
    df_combined['temp_2nd_last'] = df_combined['temp'].shift(2)
    df_combined['temp_3rd_last'] = df_combined['temp'].shift(3)
    df_combined['temp_4th_last'] = df_combined['temp'].shift(4)

    df_combined['valve_last'] = df_combined['valve'].shift(1)

    return df_combined
    
    
def train_eval():
    period_1 = '2020-03-05_2020-03-19'
    period_2 = '2020-10-13_2020-11-01'
    df_combined_1 = combine_get_ftrs(period_1)
    df_combined_2 = combine_get_ftrs(period_2)

    df_combined = df_combined_1.append(df_combined_2)

    # yes test
    mask_test = (df_combined.index >= '2020-10-21') & (df_combined.index < '2020-10-22')

    df_test = df_combined.loc[mask_test].between_time('3:45', '15:45')

    # workdays only, no test(21.10):
    mask_train = (df_combined.index >= '2020-03-05') & (df_combined.index < '2020-03-07') | \
                 (df_combined.index >= '2020-10-13') & (df_combined.index < '2020-10-17') | \
                 (df_combined.index >= '2020-03-09') & (df_combined.index < '2020-03-14') | \
                 (df_combined.index >= '2020-03-16') & (df_combined.index < '2020-03-20') | \
                 (df_combined.index >= '2020-10-19') & (df_combined.index < '2020-10-21') | \
                 (df_combined.index >= '2020-10-22') & (df_combined.index < '2020-10-24') | \
                 (df_combined.index >= '2020-10-26') & (df_combined.index < '2020-10-31')

    df_train = df_combined.loc[mask_train].between_time('3:45', '15:45')
    features = ['temp', 'target_temp', 'valve', 'temp_last', 'temp_2nd_last', 'temp_3rd_last', 'temp_4th_last']

    X_train = df_train[features].to_numpy()
    y_train = df_train['temp_gt'].to_numpy()

    scaler_mm = MinMaxScaler()
    X_train = scaler_mm.fit_transform(X_train)

    X_test = df_test[features].to_numpy()
    X_test = scaler_mm.transform(X_test)
    y_test = df_test['temp_gt'].to_numpy()
    y_last = df_test['temp_last'].to_numpy()

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    reg = ensemble.RandomForestRegressor(random_state=1)
    # linear_model.LinearRegression
    # linear_model.Lasso
    # linear_model.Ridge
    # neural_network.MLPRegressor

    # reg = neural_network.MLPRegressor(random_state=42)
    reg.fit(X_train, y_train)
    y_reg = reg.predict(X_test)
    print(X_test)
    print(y_reg)
    # print(X_test)
    # print(y_reg)
    # print(df_test)
    # print(y_reg)
    print(f'mae last: {metrics.mean_absolute_error(y_test, y_last)}')
    print(f'mae rf: {metrics.mean_absolute_error(y_test, y_reg)}')

    with open('/home/kamil/Pulpit/PUT/WZUM/regressor.p', 'wb') as handle:
        pickle.dump(reg, handle)

    with open('/home/kamil/Pulpit/PUT/WZUM/scaler.p', 'wb') as handle:
        pickle.dump(scaler_mm, handle)
    # df_test.plot()
    # plt.show()


def train_eval_from_prepared_data():
    features = ['temp', 'target_temp', 'valve', 'temp_last', 'temp_2nd_last', 'temp_3rd_last', 'temp_4th_last']

    df_combined = pd.read_csv('/home/kamil/Pulpit/PUT/WZUM/MachineLearningCourse/WZUM project template/train.csv',
                              names=['stamp'] + features, index_col=0, parse_dates=True, header=None).fillna(method='ffill')

    df_gt = pd.read_csv('/home/kamil/Pulpit/PUT/WZUM/MachineLearningCourse/WZUM project template/gt.csv',
                        names=['stamp', 'temp_gt'], index_col=0, parse_dates=True, header=None)

    df_combined = df_combined.drop_duplicates()
    df_gt = df_gt.drop_duplicates()
    df_combined = df_combined.join(df_gt, on='stamp', how='inner')

    # yes test
    mask_test = (df_combined.index >= '2020-10-21') & (df_combined.index < '2020-10-22')

    df_test = df_combined.loc[mask_test].between_time('3:45', '16:00')

    # workdays only, no test(21.10):
    mask_train = (df_combined.index >= '2020-03-05') & (df_combined.index < '2020-03-07') | \
                 (df_combined.index >= '2020-10-13') & (df_combined.index < '2020-10-17') | \
                 (df_combined.index >= '2020-03-09') & (df_combined.index < '2020-03-14') | \
                 (df_combined.index >= '2020-03-16') & (df_combined.index < '2020-03-20') | \
                 (df_combined.index >= '2020-10-19') & (df_combined.index < '2020-10-21') | \
                 (df_combined.index >= '2020-10-22') & (df_combined.index < '2020-10-24') | \
                 (df_combined.index >= '2020-10-26') & (df_combined.index < '2020-10-31')

    df_train = df_combined.loc[mask_train].between_time('3:45', '16:00')

    X_train = df_train[features].to_numpy()
    y_train = df_train['temp_gt'].to_numpy()

    scaler_mm = MinMaxScaler()
    X_train = scaler_mm.fit_transform(X_train)

    X_test = df_test[features].to_numpy()
    X_test = scaler_mm.transform(X_test)

    y_test = df_test['temp_gt'].to_numpy()
    y_last = df_test['temp_last'].to_numpy()

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, y_last.shape)

    reg = ensemble.RandomForestRegressor(random_state=1)
    # linear_model.LinearRegression
    # linear_model.Lasso
    # linear_model.Ridge
    # neural_network.MLPRegressor

    # reg = neural_network.MLPRegressor(random_state=42)
    reg.fit(X_train, y_train)
    y_reg = reg.predict(X_test)

    print(f'mae last: {metrics.mean_absolute_error(y_test, y_last)}')
    print(f'mae rf: {metrics.mean_absolute_error(y_test, y_reg)}')

    with open('/home/kamil/Pulpit/PUT/WZUM/regressor.p', 'wb') as handle:
        pickle.dump(reg, handle)

    with open('/home/kamil/Pulpit/PUT/WZUM/scaler.p', 'wb') as handle:
        pickle.dump(scaler_mm, handle)


def main():
    random.seed(42)
    pd.options.display.max_columns = None

    train_eval_from_prepared_data()


if __name__ == '__main__':
    main()


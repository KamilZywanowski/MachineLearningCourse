import pandas as pd
import random
import json
import matplotlib.pyplot as plt
from sklearn import metrics, ensemble, linear_model, neural_network, model_selection, svm, tree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

pd.set_option('expand_frame_repr', False)

#
# def load_files(period):
#     sn_temp_mid = read_temp_mid_sn()
#     df_temp = pd.read_csv(f'data/office_1_temperature_supply_points_data_{period}.csv')
#     df_temp.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
#     df_temp.rename(columns={'value': 'temp'}, inplace=True)
#     df_temp['time'] = pd.to_datetime((df_temp['time']))
#     df_temp.drop(columns=['unit'], inplace=True)
#     df_temp = df_temp[df_temp['serialNumber'] == sn_temp_mid]
#     df_temp.set_index('time', inplace=True)
#
#     df_target_temp = pd.read_csv(f'data/office_1_targetTemperature_supply_points_data_{period}.csv')
#     df_target_temp.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
#     df_target_temp.rename(columns={'value': 'target_temp'}, inplace=True)
#     df_target_temp['time'] = pd.to_datetime((df_target_temp['time']))
#     df_target_temp.drop(columns=['unit'], inplace=True)
#     df_target_temp.set_index('time', inplace=True)
#
#     df_valve_lvl = pd.read_csv(f'data/office_1_valveLevel_supply_points_data_{period}.csv')
#     df_valve_lvl.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
#     df_valve_lvl.rename(columns={'value': 'valve'}, inplace=True)
#     df_valve_lvl['time'] = pd.to_datetime((df_valve_lvl['time']))
#     df_valve_lvl.drop(columns=['unit'], inplace=True)
#     df_valve_lvl.set_index('time', inplace=True)
#     # fig, ax1 = plt.subplots()
#     #
#     # color = 'tab:red'
#     # ax1.plot(df_temp.index, df_temp.temp, color='red')
#     # ax1.plot(df_target_temp.index, df_target_temp.target_temp, color='green')
#     # ax1.tick_params(axis='y', labelcolor=color)
#     #
#     # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#     # color = 'tab:blue'
#     # ax2.plot(df_valve_lvl.index, df_valve_lvl.valve, color='blue')
#     # ax2.tick_params(axis='y', labelcolor=color)
#     #
#     # # fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     # # plt.show()
#     return df_temp, df_target_temp, df_valve_lvl
#
#
# def read_temp_mid_sn() -> int:
#     with open('data/additional_info.json') as f:
#         additional_data = json.load(f)
#
#     devices = additional_data['offices']['office_1']['devices']
#     sn_temp_mid = [d['serialNumber'] for d in devices if d['description'] == 'temperature_middle'][0]
#     return sn_temp_mid
#
#
# def combine_get_ftrs(period):
#     df_temp, df_target_temp, df_valve_lvl = load_files(period)
#     df_combined = pd.concat([df_temp, df_target_temp, df_valve_lvl])
#     df_combined = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')
#
#     df_combined['temp_gt'] = df_combined['temp'].shift(-1)
#
#     df_combined['temp_last'] = df_combined['temp'].shift(1)
#     df_combined['temp_2nd_last'] = df_combined['temp'].shift(2)
#     df_combined['temp_3rd_last'] = df_combined['temp'].shift(3)
#     df_combined['temp_4th_last'] = df_combined['temp'].shift(4)
#
#     df_combined['valve_last'] = df_combined['valve'].shift(1)
#
#     return df_combined
#
#
# def train_eval():
#     period_1 = '2020-03-05_2020-03-19'
#     period_2 = '2020-10-13_2020-11-01'
#     df_combined_1 = combine_get_ftrs(period_1)
#     df_combined_2 = combine_get_ftrs(period_2)
#
#     df_combined = df_combined_1.append(df_combined_2)
#
#     # yes test
#     mask_test = (df_combined.index >= '2020-10-21') & (df_combined.index < '2020-10-22')
#
#     df_test = df_combined.loc[mask_test].between_time('3:45', '15:45')
#
#     # workdays only, no test(21.10):
#     mask_train = (df_combined.index >= '2020-03-05') & (df_combined.index < '2020-03-07') | \
#                  (df_combined.index >= '2020-10-13') & (df_combined.index < '2020-10-17') | \
#                  (df_combined.index >= '2020-03-09') & (df_combined.index < '2020-03-14') | \
#                  (df_combined.index >= '2020-03-16') & (df_combined.index < '2020-03-20') | \
#                  (df_combined.index >= '2020-10-19') & (df_combined.index < '2020-10-21') | \
#                  (df_combined.index >= '2020-10-22') & (df_combined.index < '2020-10-24') | \
#                  (df_combined.index >= '2020-10-26') & (df_combined.index < '2020-10-31')
#
#     df_train = df_combined.loc[mask_train].between_time('3:45', '15:45')
#     features = ['temp', 'target_temp', 'valve', 'temp_last', 'temp_2nd_last', 'temp_3rd_last', 'temp_4th_last']
#
#     X_train = df_train[features].to_numpy()
#     y_train = df_train['temp_gt'].to_numpy()
#
#     scaler_mm = MinMaxScaler()
#     X_train = scaler_mm.fit_transform(X_train)
#
#     X_test = df_test[features].to_numpy()
#     X_test = scaler_mm.transform(X_test)
#     y_test = df_test['temp_gt'].to_numpy()
#     y_last = df_test['temp_last'].to_numpy()
#
#     print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#
#     reg = ensemble.RandomForestRegressor(random_state=1)
#     # linear_model.LinearRegression
#     # linear_model.Lasso
#     # linear_model.Ridge
#     # neural_network.MLPRegressor
#
#     # reg = neural_network.MLPRegressor(random_state=42)
#     reg.fit(X_train, y_train)
#     y_reg = reg.predict(X_test)
#     print(X_test)
#     print(y_reg)
#     # print(X_test)
#     # print(y_reg)
#     # print(df_test)
#     # print(y_reg)
#     print(f'mae last: {metrics.mean_absolute_error(y_test, y_last)}')
#     print(f'mae rf: {metrics.mean_absolute_error(y_test, y_reg)}')
#
#     with open('/home/kamil/Pulpit/PUT/WZUM/regressor.p', 'wb') as handle:
#         pickle.dump(reg, handle)
#
#     with open('/home/kamil/Pulpit/PUT/WZUM/scaler.p', 'wb') as handle:
#         pickle.dump(scaler_mm, handle)
#     # df_test.plot()
#     # plt.show()


def train_eval_from_prepared_data():
    train_header = ['stamp', 'temp', 'target_temp', 'valve', 'temp_last', 'temp_2nd_last', 'temp_3rd_last',
                    'temp_4th_last', 'valve_last', 'valve_2nd_last', 'valve_3rd_last', 'valve_4th_last',
                    'last_temp_reading', '2ndlast_temp_reading', 'last_valve_reading', '2ndlast_valve_reading']
    gt_header = ['stamp', 'temp_gt', 'valve_gt']

    df_combined = pd.read_csv('/home/kamil/Pulpit/PUT/WZUM/MachineLearningCourse/WZUM project template/labelright/train_5_lr.csv',
                              names=train_header, index_col=0,
                              parse_dates=True, header=None)#.fillna(method='ffill')

    df_gt = pd.read_csv('/home/kamil/Pulpit/PUT/WZUM/MachineLearningCourse/WZUM project template/labelright/gt_5_lr.csv',
                        names=gt_header, index_col=0, parse_dates=True, header=None)

    # print(df_combined.index.difference(df_gt.index))
    # print(df_gt.index.difference(df_combined.index))
    # print(df_combined[df_combined.index.duplicated(keep=False) == True])
    # print(df_gt[df_gt.index.duplicated(keep=False) == True])
    # return 0
    df_combined = df_combined.join(df_gt, on='stamp', how='inner')

    features = ['temp', 'target_temp', 'valve', 'temp_last', 'temp_2nd_last', 'temp_3rd_last',
                'temp_4th_last', 'valve_last', 'valve_2nd_last', 'valve_3rd_last', 'valve_4th_last',
                'last_temp_reading', '2ndlast_temp_reading', 'last_valve_reading', '2ndlast_valve_reading']
    # features = ['temp', 'target_temp', 'valve', 'temp_last', 'valve_last']

    # get rid of the real test set:
    # without_test = np.invert((df_combined.index >= '2020-10-21') & (df_combined.index < '2020-10-21'))
    # without_test = np.invert((df_combined.index >= '2020-03-16') & (df_combined.index < '2020-03-17'))
    # df_combined = df_combined.loc[without_test]

    # get rid of weekends
    witout_weekends = df_combined.index.weekday < 5
    df_combined = df_combined.loc[witout_weekends]

    X = df_combined.between_time('3:45', '15:45')[features].to_numpy()
    y_tgt_vgt_tl_vl = df_combined.between_time('3:45', '15:45')[['temp_gt', 'valve_gt',
                                                                 'last_temp_reading', 'last_valve_reading']].to_numpy()

    # X = df_combined[features].to_numpy()
    # y_gt_last = df_combined[['temp_gt', 'temp_last']].to_numpy()

    X_train, X_test, y_train_gt_last, y_test_gt_last = model_selection.train_test_split(X, y_tgt_vgt_tl_vl,
                                                                                        shuffle=True,
                                                                                        test_size=0.15, random_state=6)

    y_train_temp = y_train_gt_last[:, 0]
    y_train_valve = y_train_gt_last[:, 1]

    y_test_temp = y_test_gt_last[:, 0]
    y_test_valve = y_test_gt_last[:, 1]

    y_last_temp = y_test_gt_last[:, 2]
    y_last_valve = y_test_gt_last[:, 3]

    print(X_train.shape, y_train_temp.shape, X_test.shape, y_test_temp.shape, y_last_temp.shape)


    #
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, y_last.shape)
    #
    # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10, 15, 100]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 5, 10]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    # # #
    # # # # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf}
    # #
    # # param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0],
    # #               'tol': [1e-5, 1e-4, 1e-3, 1e-2],
    # #               'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    # #               'normalize': [True, False],
    # #               'random_state': [0, 1, 7, 42, 66]}
    #
    # # reg = GridSearchCV(estimator=linear_model.Ridge(), param_grid=param_grid, cv=3,
    # #                    verbose=2, n_jobs=-2)
    #


    scaler_mm = StandardScaler()
    X_train = scaler_mm.fit_transform(X_train)
    X_test = scaler_mm.transform(X_test)

    # temperature
    reg_temp = linear_model.LinearRegression() #TODO finetune this
    # reg = linear_model.Lasso()
    # reg = linear_model.Ridge(alpha=0.1, tol=1e-4, random_state=1)
    # reg_temp = neural_network.MLPRegressor(random_state=42)
    # reg_temp = RandomizedSearchCV(estimator=ensemble.RandomForestRegressor(), param_distributions=random_grid,
    #                          n_iter=1000, cv=3, verbose=2, random_state=42, n_jobs=-2)
    # reg_temp = ensemble.RandomForestRegressor(n_estimators=200, random_state=42)

    reg_temp.fit(X_train, y_train_temp)
    # print(reg_temp.best_params_)
    y_reg_temp = reg_temp.predict(X_test)

    print(f'mae temp last: {metrics.mean_absolute_error(y_test_temp, y_last_temp)}')
    print(f'mae temp reg: {metrics.mean_absolute_error(y_test_temp, y_reg_temp)}')

    # valve
    # reg_valve = ensemble.RandomForestRegressor(n_estimators=200, random_state=42)
    # reg_valve = linear_model.LinearRegression()
    # reg_valve = neural_network.MLPRegressor(random_state=42)
    # reg_valve = linear_model.Ridge()
    # reg_valve = RandomizedSearchCV(estimator=ensemble.RandomForestRegressor(), param_distributions=random_grid,
    #                                n_iter=1000, cv=3, verbose=2, random_state=42, n_jobs=-2)
    reg_valve = svm.SVR(kernel='linear')#TODO and this
    # for reg_valve in [svm.SVR(kernel='linear'), svm.SVR(kernel='sigmoid'), svm.SVR(kernel='poly'), svm.SVR(kernel='rbf'), ensemble.RandomForestRegressor(), linear_model.LinearRegression(),
    #                   neural_network.MLPRegressor(), linear_model.Ridge(), linear_model.SGDRegressor(loss="squared_loss"),
    #                   linear_model.SGDRegressor(loss="huber"), linear_model.SGDRegressor(loss="epsilon_insensitive"),
    #                   tree.DecisionTreeRegressor(), tree.ExtraTreeRegressor(),
    #                   linear_model.Lasso(), linear_model.TweedieRegressor(), ensemble.AdaBoostRegressor(),
    #                   ensemble.GradientBoostingRegressor()]:
    #     reg_valve.fit(X_train, y_train_valve)
    #     y_reg_valve = reg_valve.predict(X_test)
    #     print(f'mae valve reg: {metrics.mean_absolute_error(y_test_valve, y_reg_valve)}')

    reg_valve.fit(X_train, y_train_valve)
    # # print(reg_valve.best_params_)
    y_reg_valve = reg_valve.predict(X_test)
    print()
    print(f'mae valve last: {metrics.mean_absolute_error(y_test_valve, y_last_valve)}')
    print(f'mae valve reg: {metrics.mean_absolute_error(y_test_valve, y_reg_valve)}')

    with open('/home/kamil/Pulpit/PUT/WZUM/MachineLearningCourse/WZUM project template/dane/reg_temp.p', 'wb') as handle:
        pickle.dump(reg_temp, handle)
    with open('/home/kamil/Pulpit/PUT/WZUM/MachineLearningCourse/WZUM project template/dane/reg_valve.p', 'wb') as handle:
        pickle.dump(reg_valve, handle)
    with open('/home/kamil/Pulpit/PUT/WZUM/MachineLearningCourse/WZUM project template/dane/scaler.p', 'wb') as handle:
        pickle.dump(scaler_mm, handle)


def main():
    random.seed(42)
    pd.options.display.max_columns = None

    train_eval_from_prepared_data()


if __name__ == '__main__':
    main()

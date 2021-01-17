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


def train_eval_from_prepared_data():
    train_header = ['stamp', 'temp', 'target_temp', 'valve', 'temp_last', 'temp_2nd_last', 'temp_3rd_last',
                    'temp_4th_last', 'valve_last', 'valve_2nd_last', 'valve_3rd_last', 'valve_4th_last',
                    'last_temp_reading', '2ndlast_temp_reading', 'last_valve_reading', '2ndlast_valve_reading']
    gt_header = ['stamp', 'temp_gt', 'valve_gt']

    df_combined = pd.read_csv(
        '/home/kamil/Pulpit/PUT/WZUM/MachineLearningCourse/WZUM project template/labelright/train_15_lr.csv',
        names=train_header, index_col=0,
        parse_dates=True, header=None)

    df_gt = pd.read_csv(
        '/home/kamil/Pulpit/PUT/WZUM/MachineLearningCourse/WZUM project template/labelright/gt_15_lr.csv',
        names=gt_header, index_col=0, parse_dates=True, header=None)

    df_combined = df_combined.join(df_gt, on='stamp', how='inner')

    features = ['temp', 'target_temp', 'valve', 'temp_last', 'temp_2nd_last', 'temp_3rd_last',
                'temp_4th_last', 'valve_last', 'valve_2nd_last', 'valve_3rd_last', 'valve_4th_last',
                'last_temp_reading', '2ndlast_temp_reading', 'last_valve_reading', '2ndlast_valve_reading']
    # features = ['temp', 'target_temp', 'valve', 'temp_last', 'temp_2nd_last', 'valve_last', 'valve_2nd_last', 'valve_3rd_last',
    #             'last_temp_reading', '2ndlast_temp_reading', 'last_valve_reading', '2ndlast_valve_reading']

    # get rid of the real test set:
    # without_test = np.invert((df_combined.index >= '2020-10-21') & (df_combined.index < '2020-10-22'))
    # without_test = np.invert((df_combined.index >= '2020-03-19') & (df_combined.index < '2020-03-20'))
    # df_combined = df_combined.loc[without_test]

    # get rid of weekends
    witout_weekends = df_combined.index.weekday < 5
    df_combined = df_combined.loc[witout_weekends]

    X = df_combined.between_time('3:45', '15:45')[features].to_numpy()
    y_tgt_vgt_tl_vl = df_combined.between_time('3:45', '15:45')[['temp_gt', 'valve_gt',
                                                                 'last_temp_reading', 'last_valve_reading']].to_numpy()

    X_train, X_val, y_train_gt_last, y_val_gt_last = model_selection.train_test_split(X, y_tgt_vgt_tl_vl,
                                                                                        shuffle=True,
                                                                                        test_size=0.1, random_state=6)

    y_train_temp = y_train_gt_last[:, 0]
    y_train_valve = y_train_gt_last[:, 1]

    y_val_temp = y_val_gt_last[:, 0]
    y_val_valve = y_val_gt_last[:, 1]

    y_last_temp = y_val_gt_last[:, 2]
    y_last_valve = y_val_gt_last[:, 3]

    print(X_train.shape, y_train_temp.shape, X_val.shape, y_val_temp.shape, y_last_temp.shape)

    scaler_mm = StandardScaler()
    X_train = scaler_mm.fit_transform(X_train)
    X_val = scaler_mm.transform(X_val)

    # TEMPERATURE
    reg_temp = linear_model.LinearRegression()
    reg_temp.fit(X_train, y_train_temp)
    y_reg_temp = reg_temp.predict(X_val)
    print()
    print(f'mae temp last: {metrics.mean_absolute_error(y_val_temp, y_last_temp)}')
    print(f'mae temp reg: {metrics.mean_absolute_error(y_val_temp, y_reg_temp)}')

    # VALVE
    reg_valve = svm.SVR(kernel='poly', C=0.1, coef0=5, degree=4, epsilon=0.2, gamma='auto')
    reg_valve.fit(X_train, y_train_valve)
    y_reg_valve = reg_valve.predict(X_val)
    print()
    print(f'mae valve last: {metrics.mean_absolute_error(y_val_valve, y_last_valve)}')
    print(f'mae valve reg: {metrics.mean_absolute_error(y_val_valve, y_reg_valve)}')

    with open('/home/kamil/Pulpit/PUT/WZUM/MachineLearningCourse/WZUM project template/dane/reg_temp.p', 'wb') \
            as tfile:
        pickle.dump(reg_temp, tfile)
    with open('/home/kamil/Pulpit/PUT/WZUM/MachineLearningCourse/WZUM project template/dane/reg_valve.p', 'wb') \
            as vfile:
        pickle.dump(reg_valve, vfile)
    with open('/home/kamil/Pulpit/PUT/WZUM/MachineLearningCourse/WZUM project template/dane/scaler.p', 'wb') \
            as sfile:
        pickle.dump(scaler_mm, sfile)


def main():
    random.seed(42)
    pd.options.display.max_columns = None
    train_eval_from_prepared_data()


if __name__ == '__main__':
    main()

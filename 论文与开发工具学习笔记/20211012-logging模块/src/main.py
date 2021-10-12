#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202110082329
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
Stream数据预处理，使用logger对日志进行打点。
'''

import warnings
import pandas as pd
from tqdm import tqdm
import numpy as np

import os
import logging
from datetime import datetime
from utils.logger import get_logger

API_URL = 'https://oapi.dingtalk.com/robot/send?access_token=d1b2a29b2ae62bc709693c02921ed097c621bc33e5963e9e0a5d5adf5eac10c1'

def generate_simulation_data(n_points=10000, min_interval=20):
    '''生成KPI仿真数据'''
    sensor_vals = np.linspace(0, np.pi * 10.75, n_points, endpoint=False)
    sensor_vals = np.cos(sensor_vals) + np.sin(sensor_vals * 5) * 0.2
    sensor_vals += np.cos(sensor_vals * 2) * 2.2
    sensor_vals += np.random.uniform(-2, 1, n_points)

    timestamp = np.array(
        [i * min_interval for i in range(4 * n_points)], dtype=np.int64
    )

    # 生成随机的时间戳
    rand_idx = np.arange(0, len(timestamp))
    np.random.shuffle(rand_idx)

    timestamp = timestamp[rand_idx[:n_points]]
    timestamp = np.sort(timestamp)

    # 组装数据为pandas DataFrame
    kpi_df = pd.DataFrame(
        {'timestamp': timestamp, 'value': sensor_vals,
         'kpi_id': np.ones((len(timestamp), ))}
    )

    return kpi_df


def get_datetime():
    '''获取str时间戳'''
    curr_datetime = str(datetime.now()).split('.')[0]
    curr_date = curr_datetime.split(' ')[0]
    curr_time = curr_datetime.split(' ')[1]
    curr_time = curr_time.split(':')
    curr_time = '-'.join(curr_time)

    curr_datetime = '{} {}'.format(curr_date, curr_time)

    return curr_datetime


if __name__ == '__main__':
    # 日志配置相关信息
    # ******************
    LOGGING_PATH = './logs/'
    LOGGING_FILENAME = '{} kpi_anomaly_detection.log'.format(get_datetime())

    # for log_name in os.listdir(LOGGING_PATH):
    #     os.remove(
    #         os.path.join(LOGGING_PATH, log_name)
    #     )

    logger = get_logger(
        logger_name='kpi_anomaly_detection',
        logger_min_level='DEBUG',
        is_print_std=True,
        is_send_dingtalk=False,
        is_save_to_disk=True,
        log_path=os.path.join(LOGGING_PATH, LOGGING_FILENAME)
    )

    # 日志打点
    # ******************
    kpi_df = generate_simulation_data(n_points=2000, min_interval=10)

    logger.info('KPI anomaly detection start...')
    logger.info('train shape: {}'.format(kpi_df.shape))
    logger.info('Testing..........')

    # print('********************')
    # 打印shape
    # 打印均值方差标准差
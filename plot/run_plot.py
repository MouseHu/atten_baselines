import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from plot.plot_utils import *

if __name__ == "__main__":
    tag = "discount_q"
    env = "walker"
    mine_config = "td3_mem_backup_smallbuffer_alpha_0.5"
    baseline_config = "td3_baseline"
    mine_data = data_read(
        paths=['./data/run-{}_{}_{}_tb-tag-{}.csv'.format(env, mine_config, i, tag) for i in range(3)])
    td3_data = data_read(
        paths=['./data/run-{}_{}_{}_tb-tag-{}.csv'.format(env, baseline_config, i, tag) for i in range(3)])
        # paths=['./data/run-{}_{}_tb-tag-{}.csv'.format(env, baseline_config, tag) for i in range(3)])

    datas = [mine_data, td3_data]
    legends = ['CEC', 'TD3']
    # datas = [mine_data_new_intr, mine_data_exploit_only]
    # legends = ['MetaCURE', 'MetaCURE Without Exploitation Policy']
    plot_all(datas, legends, 1)
    plt.title('{}-{}'.format(env,tag), size=20)
    # plt.plot(mine_data[0], np.ones(mine_data[0].shape) * 4.02, color='olive', linestyle='--', linewidth=2,label='EPI')
    legend()
    plt.show()

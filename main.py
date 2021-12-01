# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from Matrix import Matrix


def get_graph_of_ue(matrix, n_gnb, i_ue, n_time):
    arr = [[matrix[g][i_ue][t] for t in range(n_time)] for g in range(n_gnb)]

    df = pd.DataFrame(arr,
                      index=[str(g+1) for g in range(n_gnb)],
                      columns=[str(t+1) for t in range(n_time)])
    df = df.transpose()

    plt.xticks(rotation=45)
    plt.xticks([t for t in range(0, n_time, 10)])
    plt.plot(df.index, df.values)
    plt.title(f'UE: {i_ue+1}')
    plt.xlabel('Time')
    plt.ylabel('CQI')
    plt.legend(labels=["gNB " + str(g+1) for g in range(n_gnb)])
    plt.show()


def get_graph_of_gnb_ue(matrix, i_gnb, i_ue, n_time):
    arr = [matrix[i_gnb][i_ue][t] for t in range(n_time)]

    df = pd.DataFrame(arr,
                      index=[str(t) for t in range(n_time)],
                      columns=[str(i_gnb+1)])

    plt.xticks(rotation=45)
    plt.xlim([-10, 105])
    plt.xticks([t for t in range(n_time, n_time // 10)])
    plt.ylim([-1, 16])
    plt.yticks([y for y in range(16)])
    plt.plot(df.index, df.values)
    plt.title(f'gNB: {i_gnb+1}, UE: {i_ue+1}')
    plt.xlabel('Time')
    plt.ylabel('CQI')
    plt.savefig(f'gnb{i_gnb},ue{i_ue}')
    plt.show()


def save_graphs(cqi_matrix):
    for g in range(cqi_matrix.n_gnb):
        for u in range(cqi_matrix.n_ue):
            get_graph_of_gnb_ue(cqi_matrix.matrix, g, u, cqi_matrix.n_time)


def get_cqi_connection_probability_arr(connection_matrix_file_name):
    cqi_matrix = Matrix()
    cqi_matrix.read_from("cqi_matrix.txt")
    connection_matrix = Matrix()
    connection_matrix.read_from(connection_matrix_file_name)

    arr = [0 for x in range(16)]
    cnt = [0 for x in range(16)]
    for g in range(cqi_matrix.n_gnb):
        for u in range(cqi_matrix.n_ue):
            for t in range(cqi_matrix.n_time):
                cnt[cqi_matrix.matrix[g][u][t]] += 1
                if connection_matrix.matrix[g][u][t] == 1:
                    arr[cqi_matrix.matrix[g][u][t]] += 1

    arr = [arr[i]/cnt[i] for i in range(16)]
    return arr


def get_graph_of_cqi_connection_probability(fig_name, max_gnb_connection):
    na_arr = get_cqi_connection_probability_arr('connection_matrix_naive_approach_' + str(max_gnb_connection) + '.txt')
    df_na_arr = pd.DataFrame(na_arr, index=[x for x in range(16)], columns=["CQI"])

    cs_arr = get_cqi_connection_probability_arr('connection_matrix_cqi_sorting_' + str(max_gnb_connection) + '.txt')
    df_cs_arr = pd.DataFrame(cs_arr, index=[x for x in range(16)], columns=["CQI"])

    es_arr = get_cqi_connection_probability_arr('connection_matrix_expectation_sorting_' + str(max_gnb_connection) + '.txt')
    df_es_arr = pd.DataFrame(es_arr, index=[x for x in range(16)], columns=["CQI"])

    bf_arr = get_cqi_connection_probability_arr('connection_matrix_brute_force_' + str(max_gnb_connection) + '.txt')
    df_bf_arr = pd.DataFrame(bf_arr, index=[x for x in range(16)], columns=["CQI"])

    plt.xlim([-1, 16])
    plt.ylim([0, 1])
    plt.xticks([x for x in range(16)], rotation=45)
    plt.plot(df_na_arr.index, df_na_arr.values, label='naive approach')
    plt.plot(df_cs_arr.index, df_cs_arr.values, label='cqi sorting')
    plt.plot(df_es_arr.index, df_es_arr.values, label='expectation sorting')
    plt.plot(df_bf_arr.index, df_bf_arr.values, label='brute force')
    plt.legend()
    plt.title('CQI Connection Probabilities')
    plt.xlabel('CQI')
    plt.ylabel('Connection Probabilities')
    plt.savefig(fig_name + '_' + str(max_gnb_connection), dpi=300)
    plt.show()


def get_comp_quality_graph_cqi_quality_5_10_100():
    na_arr_5_10_100 = [159.58141141306646, 174.84288677468876, 176.75415796317677, 177.15541114488758,
                       177.15541114488758, 177.15541114488758, 177.15541114488758, 177.15541114488758,
                       177.15541114488758]  # naive approach 5x10x100
    df_na_5_10_100 = pd.DataFrame(na_arr_5_10_100,
                                  index=[x + 2 for x in range(9)],
                                  columns=['CoMP Quality'])

    cs_arr_5_10_100 = [173.53484442078036, 176.5478515014812, 177.3893853524666, 177.5229076524483, 177.5229076524483,
                       177.5229076524483, 177.5229076524483, 177.5229076524483, 177.5229076524483]  # cqi sorting 5x10x100
    df_cs_5_10_100 = pd.DataFrame(cs_arr_5_10_100,
                                  index=[x + 2 for x in range(9)],
                                  columns=['CoMP Quality'])

    es_arr_5_10_100 = [173.3576917232533, 176.35595005541614, 177.03291931986521, 177.15541114488758,
                       177.15541114488758, 177.15541114488758, 177.15541114488758, 177.15541114488758,
                       177.15541114488758]  # expectation sorting 5x10x100
    df_es_5_10_100 = pd.DataFrame(es_arr_5_10_100,
                                  index=[x + 2 for x in range(9)],
                                  columns=['CoMP Quality'])

    bf_arr_5_10_100 = [22.541, 23.225, 23.324, 23.354, 23.354, 23.354, 23.354, 23.354, 23.354]  # brute force 5x10x100
    df_bf_5_10_100 = pd.DataFrame(bf_arr_5_10_100,
                                  index=[x + 2 for x in range(9)],
                                  columns=['CoMP Quality'])

    # plt.xlim([1, 11])

    plt.ylim([159, 178])
    plt.xticks([x + 2 for x in range(9)])
    plt.plot(df_na_5_10_100.index, df_na_5_10_100.values, label='naive approach')
    plt.plot(df_cs_5_10_100.index, df_cs_5_10_100.values, label='cqi sorting')
    plt.plot(df_es_5_10_100.index, df_es_5_10_100.values, label='expectation sorting')
    plt.plot(df_bf_5_10_100.index, df_bf_5_10_100.values, label='brute force')

    plt.legend()
    plt.title('CoMP Quality - BS: 5, UE: 10')
    plt.xlabel('# of available users of a base station')
    plt.ylabel('CoMP Quality')
    plt.savefig('comp_quality_5_10_100', dpi=300)
    plt.show()


def get_comp_quality_graph_cqi_quality_5_100_100():
    na_arr_5_100_100 = [182.63720590988672, 189.92794196630945, 190.23559084705215, 190.23559084705215,
                        190.23559084705215, 190.23559084705215, 190.23559084705215, 190.23559084705215,
                        190.23559084705215]  # naive approach 5x100x100
    df_na_5_100_100 = pd.DataFrame(na_arr_5_100_100,
                                   index=[(x + 2) * 10 for x in range(9)],
                                   columns=['CoMP Quality'])

    cs_arr_5_100_100 = [189.83509278749054, 190.49936835253027, 190.5004912179543, 190.5004912179543, 190.5004912179543,
                        190.5004912179543, 190.5004912179543, 190.5004912179543,
                        190.5004912179543]  # cqi sorting 5x100x100
    df_cs_5_100_100 = pd.DataFrame(cs_arr_5_100_100,
                                   index=[(x + 2) * 10 for x in range(9)],
                                   columns=['CoMP Quality'])

    es_arr_5_100_100 = [189.57197593968166, 190.23264797312356, 190.23559084705215, 190.23559084705215,
                        190.23559084705215, 190.23559084705215, 190.23559084705215, 190.23559084705215,
                        190.23559084705215]  # expectation sorting 5x100x100
    df_es_5_100_100 = pd.DataFrame(es_arr_5_100_100,
                                   index=[(x + 2) * 10 for x in range(9)],
                                   columns=['CoMP Quality'])

    # plt.xlim([1, 11])

    plt.ylim([182, 191])
    plt.xticks([(x + 2) * 10 for x in range(9)])
    plt.plot(df_na_5_100_100.index, df_na_5_100_100.values, label='naive approach')
    plt.plot(df_cs_5_100_100.index, df_cs_5_100_100.values, label='cqi sorting')
    plt.plot(df_es_5_100_100.index, df_es_5_100_100.values, label='expectation sorting')

    plt.legend()
    plt.title('CoMP Quality - BS: 5, UE: 100')
    plt.xlabel('# of available users of a base station')
    plt.ylabel('CoMP Quality')
    plt.savefig('comp_quality_5_100_100', dpi=300)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get_comp_quality_graph_cqi_quality_5_10_100()
    get_comp_quality_graph_cqi_quality_5_100_100()

    sys.exit()

    for i in range(9):
        get_graph_of_cqi_connection_probability('cqi_connection_probability', i + 2)

    sys.exit()



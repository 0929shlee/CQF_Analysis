# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt
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
    plt.xticks([t for t in range(0, n_time, n_time // 10)])
    plt.ylim([-1, 16])
    plt.yticks([y for y in range(0, 16)])
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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cqi_matrix = Matrix()
    cqi_matrix.read_from("cqi_matrix.txt")
    #save_graphs(cqi_matrix)



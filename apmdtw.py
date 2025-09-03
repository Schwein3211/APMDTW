

import os
import numpy as np
from dtw_penalty import apmdtw

from utils import load_data,load_data_query, load_timeseries_id, write_result_file_ts_similarity


ds_count=128


def euclidean(x, y):
    return np.linalg.norm(x-y) 


def compute_penalty_cost_matrix(file_name, line_num_ref, line_num_query, lambda_=0.5, k_a=0.3, k_d=0.8, epsilon=1e-6, gama=1e-3):
    # file_name_ref = 'data/' + file_name
    file_name_ref = 'Data128_txt/' + file_name
    reference = load_data(file_name_ref, line_num_ref)

    file_name_query = 'QueryData128_txt/' + file_name + '_train'
    warped_query = load_data_query(file_name_query, line_num_query)
    X = reference
    Y = warped_query
    T, L = len(X), len(Y)

    total_d = 0
    count = 0
    d_max = 0
    for i in range(T):
        for j in range(L):
            d= euclidean(X[i], Y[j])
            total_d += d
            count += 1
            d_max = max(d_max, d)

    avg_d = total_d / count

    penalty_map = np.zeros((T - 1, L - 1))
    cost_matrix = np.zeros((T - 1, L - 1))
    for i in range(T - 1):
        v_i = np.array([X[i + 1][0] - X[i][0], X[i + 1][1] - X[i][1]])
        for j in range(L - 1):
            u_j = np.array([Y[j + 1][0] - Y[j][0], Y[j + 1][1] - Y[j][1]])

            d = euclidean(X[i], Y[j])
            d_log = np.where(d <= avg_d, np.log(1 + gama * d), d * gama)

            d_norm = d_log / np.log(1 + gama * d_max ) #


            cos_theta = np.dot(v_i, u_j) / (np.linalg.norm(v_i) * np.linalg.norm(u_j) + epsilon)
            theta = np.arccos(np.clip(cos_theta, -1, 1))
            theta_norm = (theta / np.pi) ** 2

            A = (1.0 + k_a * theta_norm) ** (1.05 - d_norm)
            B = (1.0 + k_d * d_norm) ** (d_norm)

            penalty = lambda_ * (A * B - 1.0)
            penalty_map[i, j] = penalty
            cost_matrix[i, j] = d_norm + penalty
    return penalty_map, cost_matrix

def pkg_apmdtw(file_name, cost_matrix, line_num_ref, line_num_query):
    # file_name_ref = 'data/' + file_name
    file_name_ref = 'Data128_txt/' + file_name
    reference = load_data(file_name_ref, line_num_ref)

    file_name_query = 'QueryData128_txt/' + file_name + '_train'
    warped_query = load_data_query(file_name_query, line_num_query)

    dist, acc_cost_matrix, path = apmdtw(reference, warped_query, penalty=cost_matrix)

    return dist



if __name__ == "__main__":
    os.chdir(os.path.abspath('..'))
    data_dir = os.getcwd() + '/Data128_txt/'
    querydata_dir = os.getcwd() + '/QueryData128_txt/' # todo
    oslist = [f for f in sorted(os.listdir(data_dir)) if os.path.isfile(data_dir+f)]


    for i in range(0, ds_count):
        correct_num_acc=0
        linesnum = len(open(data_dir + oslist[i], "r").readlines())
        query_ts_num = len(open(querydata_dir + oslist[i] + '_train', "r").readlines())
        dist_list = []
        for j in range(1, linesnum+1): #reference time series
            min=np.inf
            argmin=np.inf

            for r in range(1, query_ts_num+1): #query time series
                penalty_map, cost_matrix = compute_penalty_cost_matrix(oslist[i], j, r)
                dist = pkg_apmdtw(oslist[i], cost_matrix, j, r)
                if dist < min:
                    min = dist
                    argmin = r

                print("Method: APMDTW, file:", oslist[i], ": ", i + 1, "/", ds_count, " -- ref: ", j, "/", linesnum , " -- query: ", r, "/", query_ts_num )
            # if load_timeseries_id('data/'+oslist[i],j) == load_timeseries_id('data/'+oslist[i],argmin):
            if load_timeseries_id('Data128_txt/' + oslist[i], j) == load_timeseries_id('QueryData128_txt/' + oslist[i] + '_train', argmin):
                correct_num_acc = correct_num_acc + 1

        write_result_file_ts_similarity('result_APMDTW_ucr.csv', 'APMDTW_metric_distance', oslist[i], correct_num_acc/linesnum)

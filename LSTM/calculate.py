import sys

import os
import pickle
import torch
import math
from tqdm import tqdm
from conf import conf, saveResult, checkResult, getSuffix
from dataLoader import getDataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from utility.tools import dateToInt, dateParser, nextDate, dateToFormalString

def show():
    new_conf = conf
    new_conf['entity_pair'] = 'CAN2GBR'

    ground_truth = [[] for i in range(4)]
    data_loader = getDataLoader('train', new_conf, shuffle=False)

    time_list = []
    y = 2005
    m = 1
    d = 1
    for i, data in enumerate(data_loader):
        if i == 20:
            break

        if i == 0:
            for j in range(4):
                ground_truth[j].extend(data[0, :, j].numpy())

            for k in range(new_conf['sequence_len']):
                time_list.append(dateToFormalString(y, m, d))
                for j in range(7):
                    y, m, d = nextDate(y, m, d)

        else:
            for j in range(4):
                ground_truth[j].append(data[0, new_conf['sequence_len'] - 1, j].item())

            time_list.append(dateToFormalString(y, m, d))
            for j in range(7):
                y, m, d = nextDate(y, m, d)

    #print time_list
    print 'Length of ground truth: {}'.format(len(ground_truth[0]))

    plt.figure(figsize=(20, 20))
    plt.xticks(rotation=30)
    plt.xlabel('Time')
    plt.ylabel('Number of events')
    plt.title('Sequence data sample of country pair: {}'.format(new_conf['entity_pair']))

    plt.plot(range(len(ground_truth[0])), ground_truth[0], linewidth=1.0, label='1: Verbal Cooperation')
    plt.plot(range(len(ground_truth[0])), ground_truth[1], linewidth=1.0, label='2: Material Cooperation')
    plt.plot(range(len(ground_truth[0])), ground_truth[2], linewidth=1.0, label='3: Verbal Conflict')
    plt.plot(range(len(ground_truth[0])), ground_truth[3], linewidth=1.0, label='4: Material Conflict')

    for i in range(len(time_list)):
        if i % 3 != 0:
            time_list[i] = ""

    plt.xticks(range(len(ground_truth[0])), time_list)

    plt.legend()
    plt.show()


def cal_accuracy():

    country_list = ['USA', 'GBR', 'RUS', 'CHN', 'CAN', 'AUS', 'FRA', 'JPN']
    clen = len(country_list)

    for i in range(clen):
        for j in range(clen):
            if i == j:
                continue
            conf['entity_pair'] = country_list[i] + '2' + country_list[j]
            print 'Current pair: {}'.format(conf['entity_pair'])

            suffix = getSuffix(conf)

            ground_truth_filename = 'LSTM/result/ground_truth_' + suffix
            model_result_filename = 'LSTM/result/model_result_' + suffix

            with open(ground_truth_filename) as f:
                ground_truth = pickle.load(f)

            with open(model_result_filename) as f:
                model_result = pickle.load(f)

            glen = len(ground_truth)
            mlen = len(model_result)

            correct = 0
            avg_diff = 0
            avg_ground = 0

            for id in range(15, glen):
                avg_diff += math.fabs(ground_truth[id] - model_result[id])
                avg_ground += ground_truth[id]

                if ground_truth[id] > ground_truth[id - 1]:
                    if model_result[id] > model_result[id - 1]:
                        correct += 1
                elif ground_truth[id] < ground_truth[id - 1]:
                    if model_result[id] < model_result[id - 1]:
                        correct += 1

            print 'Correct: {}, Accuracy: {}, Avg_diff: {}, Avg_ground: {}'.format(correct, float(correct) / glen, avg_diff, avg_ground)



if __name__ == '__main__':

    #show()

    cal_accuracy()

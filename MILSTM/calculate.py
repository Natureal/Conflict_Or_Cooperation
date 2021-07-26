import sys
import os
import pickle
import torch
import math
from tqdm import tqdm
from conf import conf, saveResult, checkResult, getSuffix
from dataLoader import getMultiDataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from utility.tools import dateToInt, dateParser, nextDate, dateToFormalString

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

            ground_truth_filename = 'MILSTM/result/ground_truth_' + suffix
            model_result_filename = 'MILSTM/result/model_result_' + suffix

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

    cal_accuracy()


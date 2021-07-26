import sys
import os
import pickle
import torch
from tqdm import tqdm
from model import EMILSTMTrainer
from conf import conf, saveResult, checkResult, showResult
from dataLoader import getMultiDataLoader
import matplotlib.pyplot as plt

def run():

    if checkResult(conf):
        print 'Alrealy have the result'
        showResult(conf)
        return

    print 'current device: {}'.format(conf['device'])

    train_loader = getMultiDataLoader('train', conf, shuffle = True)
    test_loader = getMultiDataLoader('test', conf, shuffle = False)

    print 'Target pair: {}'.format(conf['entity_pair'])
    print 'Length of training set: {}'.format(len(train_loader))
    print 'Length of test set: {}'.format(len(test_loader))

    trainer = EMILSTMTrainer(conf).to(conf['device'])

    ground_truth = []
    model_result = []

    print '=== Start to train ... ==='
    loss_each_epoch = []
    for epoch in tqdm(range(conf['max_epoch'])):
        # training
        for i, data in enumerate(train_loader):
            # data: [batch_size, num_factors + 1, sequence_len, features]
            # data_train: [batch_size, num_factors + 1, sequence_len - 1, features]
            data_train = data[:, :, 0:conf['sequence_len'] - 1, :].to(conf['device'])
            # data_val: [batch_size, 1, 1, features], used for validation (that is loss calculation)
            data_val = data[:, 0, conf['sequence_len'] - 1, conf['what_we_need']].unsqueeze(1).to(conf['device'])

            loss = trainer.update(data_train, data_val)

        # evaluating
        loss_ave = 0
        count = 0
        for i, data in enumerate(test_loader):
            data_test = data[:, :, 0:conf['sequence_len'] - 1, :].to(conf['device'])
            data_val = data[:, 0, conf['sequence_len'] - 1, conf['what_we_need']].unsqueeze(1).to(conf['device'])

            loss, result = trainer.eval(data_test, data_val)
            loss_ave += loss
            count += 1

            if epoch == conf['max_epoch'] - 1:
                if i == 0:
                    ground_truth.extend(data[0, 0, :, conf['what_we_need']].numpy())
                    model_result.extend(data[0, 0, 0:conf['sequence_len'] - 1, conf['what_we_need']].numpy())
                    model_result.append(result[0][0].item())
                else:
                    ground_truth.append(data[0, 0, conf['sequence_len'] - 1, conf['what_we_need']].item())
                    model_result.append(result[0][0].item())

        loss_ave /= float(count)
        print 'loss ave: {}'.format(loss_ave)
        loss_each_epoch.append(loss_ave)

    saveResult(conf, trainer, loss_each_epoch, ground_truth, model_result)
    #showResult(conf)


def batch_train():
    country_list = ['USA', 'GBR', 'RUS', 'CHN', 'CAN', 'AUS', 'FRA', 'JPN']
    clen = len(country_list)

    for i in range(clen):
        for j in range(clen):
            if i == j:
                continue
            conf['entity_pair'] = country_list[i] + '2' + country_list[j]
            print 'current pair: {}'.format(conf['entity_pair'])
            run()


if __name__ == '__main__':
    #run()
    batch_train()
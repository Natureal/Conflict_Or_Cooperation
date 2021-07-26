import sys
import os
import pickle
from tqdm import tqdm
from model import ELSTMTrainer
from conf import conf, saveResult, checkResult, showResult
from dataLoader import getDataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# ---------------------------------------
# This code is for training process.
# Steps:
# (1) Check whether we already have the result.
# (2) Load training and test dataset.
# (3) Load the trainer, whose model is a LSTM in Pytorch or LSTM by hand.
# (4) Trainer trians max_epoch rounds on the whole training set, and calculates loss each round.
# (5) Show the precition result compared with the ground truth.
# ---------------------------------------
def run():
    if checkResult(conf):
        print 'Alrealy have the result'
        #showResult(conf)
        return

    print 'current device: {}'.format(  conf['device'])
    train_loader = getDataLoader('train', conf, shuffle=True)
    test_loader = getDataLoader('test', conf, shuffle=False)

    print 'Target pair: {}'.format(conf['entity_pair'])
    print 'Length of training set: {}'.format(len(train_loader))
    print 'Length of test set: {}'.format(len(test_loader))

    trainer = ELSTMTrainer(conf).to(conf['device'])

    ground_truth = []
    model_result = []

    print '=== Start to train ... ==='
    loss_each_epoch = []
    for epoch in tqdm(range(conf['max_epoch'])):
        # training
        for i, data in enumerate(train_loader):
            # data: [batch_size, sequence_len, features]
            # data_train: [batch_size, sequence_len - 1, features]
            data_train = data[:, 0:conf['sequence_len']-1, :].to(conf['device'])
            # data_val: [batch_size, 1, features], used for validation (that is loss calculation)
            data_val = data[:, conf['sequence_len']-1, conf['what_we_need']].unsqueeze(1).to(conf['device'])

            loss = trainer.update(data_train, data_val)

        # evaluating
        loss_ave = 0
        count = 0
        for i, data in enumerate(test_loader):
            data_test = data[:, 0:conf['sequence_len']-1, :].to(conf['device'])
            data_val = data[:, conf['sequence_len']-1, conf['what_we_need']].unsqueeze(1).to(conf['device'])

            #print data_test

            loss, result = trainer.eval(data_test, data_val)
            loss_ave += loss
            count += 1

            if epoch == conf['max_epoch'] - 1:
                if i == 0:
                    ground_truth.extend(data[0, :, conf['what_we_need']].numpy())
                    model_result.extend(data[0, 0:conf['sequence_len']-1, conf['what_we_need']].numpy())
                    model_result.append(result[0][0].item())
                else:
                    ground_truth.append(data[0, conf['sequence_len']-1, conf['what_we_need']].item())
                    model_result.append(result[0][0].item())

        loss_ave /= float(count)
        print 'loss ave: {}'.format(loss_ave)
        loss_each_epoch.append(loss_ave)


    saveResult(conf, trainer, loss_each_epoch, ground_truth, model_result)
    #showResult(conf)


def batch_train():
    country_list = ['USA', 'GBR', 'RUS', 'CHN', 'CAN',   'AUS', 'FRA', 'JPN']
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
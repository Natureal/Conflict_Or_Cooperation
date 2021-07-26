import torch
import pickle
import os
import matplotlib.pyplot as plt

conf = {
    'origin_path': '/home/naturain/PycharmProjects/',

    'moving_average': (3, 1), # window: 3, number of iteration: 1

    'model': 'PytorchLSTM',

    'time_unit': 7,
    'sequence_len': 16,

    'entity_pair': 'JPN2CHN',

    'train_batch_size': 1,
    'test_batch_size': 1,

    'train_start_date': 20050101,
    'train_end_date': 20160601,
    'test_start_date': 20160601,
    'test_end_date': 20181201,

    'input_size': 4,
    'hidden_size': 256,

    'num_input': 4,
    'num_output': 1,
    'what_we_need': 2,

    'num_layers': 1,
    'drop_out': 0.0,
    'max_epoch': 50,
    'learning_rate': 0.0002,

    'device': 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

def getSuffix(conf):
    suffix = conf['model'] + '_' + \
             str(conf['moving_average'][0]) + '_' + \
             str(conf['moving_average'][1]) + '_' + \
             str(conf['train_batch_size']) + '_' + \
             str(conf['time_unit']) + '_' + \
             str(conf['sequence_len']) + '_' + \
             conf['entity_pair'] + '_' + \
             str(conf['hidden_size']) + '_' + \
             str(conf['num_layers']) + '_' + \
             str(conf['what_we_need']) + '_' + \
             str(conf['drop_out']) + '_' + \
             str(conf['learning_rate']) + '_' + \
             str(conf['max_epoch']) + '.pkl'
    return suffix

def checkResult(conf):
    suffix = getSuffix(conf)

    loss_filename = 'LSTM/result/loss_' + suffix
    if os.path.isfile(loss_filename):
        return True
    return False


def showResult(conf):
    suffix = getSuffix(conf)

    loss_filename = 'LSTM/result/loss_' + suffix
    ground_truth_filename = 'LSTM/result/ground_truth_' + suffix
    model_result_filename = 'LSTM/result/model_result_' + suffix

    with open(loss_filename) as f:
        loss = pickle.load(f)

    with open(ground_truth_filename) as f:
        ground_truth = pickle.load(f)

    with open(model_result_filename) as f:
        model_result = pickle.load(f)

    plt.figure(figsize=(18, 10))
    plt.xlabel('Week')
    plt.ylabel('Number of material conflicts')
    plt.title('Sequence data of country pair: {}'.format(conf['entity_pair']))
    plt.plot(range(len(ground_truth)), ground_truth, label='ground truth for material conflicts')
    plt.plot(range(len(model_result)), model_result, label='prediction for material conflicts')
    plt.legend()
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.xlabel('epoch(s)')
    plt.ylabel('MSE loss')
    plt.title('Training loss country pair: {}'.format(conf['entity_pair']))
    plt.plot(range(len(loss)), loss)
    plt.legend()
    plt.show()


def saveResult(conf, trainer, loss_each_epoch, ground_truth, model_result):
    suffix = getSuffix(conf)

    trainer_filename = 'LSTM/result/trainer_' + suffix
    torch.save(trainer, trainer_filename)

    loss_filename = 'LSTM/result/loss_' + suffix
    with open(loss_filename, 'w') as f:
        pickle.dump(loss_each_epoch, f)

    ground_truth_filename = 'LSTM/result/ground_truth_' + suffix
    with open(ground_truth_filename, 'w') as f:
        pickle.dump(ground_truth, f)

    model_result_filename = 'LSTM/result/model_result_' + suffix
    with open(model_result_filename, 'w') as f:
        pickle.dump(model_result, f)

    print '--- Pickled everything ---'
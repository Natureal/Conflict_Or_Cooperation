import pickle
import torch
import math
import matplotlib.pyplot as plt

entity_pair = 'GBR2RUS'

what_we_need = 3

conf1 = {
    'origin_path': '/home/naturain/PycharmProjects/',
    'moving_average': (3, 1), # window: 3, number of iteration: 1
    'model': 'LSTM',
    'time_unit': 7,
    'sequence_len': 16,
    'entity_pair': entity_pair,
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
    'what_we_need': what_we_need,
    'num_layers': 1,
    'drop_out': 0.0,
    'max_epoch': 50,
    'learning_rate': 0.0002,
}

conf2 = {
    # unique parameters of MILSTM
    'related': True,
    'standard': 'correlation',
    'topK': 3,
    # ----------------
    'origin_path': '/home/naturain/PycharmProjects/',
    'moving_average': (3, 1), # window: 3, number of iteration: 1
    'model': 'MILSTM',
    'time_unit': 7,
    'sequence_len': 16,
    'entity_pair': entity_pair,
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
    'what_we_need': what_we_need,
    'num_layers': 1,
    'drop_out': 0.0,
    'max_epoch': 50,
    'learning_rate': 0.0002,
}

def get_LSTM_suffix(conf):
    suffix = 'LSTM' + '_' + \
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
    loss_filename = 'LSTM/result/loss_' + suffix
    ground_truth_filename = 'LSTM/result/ground_truth_' + suffix
    model_result_filename = 'LSTM/result/model_result_' + suffix
    return loss_filename, ground_truth_filename, model_result_filename


def get_PytorchLSTM_suffix(conf):
    suffix = 'PytorchLSTM' + '_' + \
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
    loss_filename = 'LSTM/result/loss_' + suffix
    ground_truth_filename = 'LSTM/result/ground_truth_' + suffix
    model_result_filename = 'LSTM/result/model_result_' + suffix
    return loss_filename, ground_truth_filename, model_result_filename

def get_MILSTM_suffix(conf):
    suffix = 'MILSTM' + '_' + \
             str(conf['moving_average'][0]) + '_' + \
             str(conf['moving_average'][1]) + '_' + \
             str(conf['standard']) + '_' + \
             str(conf['related']) + '_' + \
             str(conf['topK']) + '_' + \
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
    loss_filename = 'MILSTM/result/loss_' + suffix
    ground_truth_filename = 'MILSTM/result/ground_truth_' + suffix
    model_result_filename = 'MILSTM/result/model_result_' + suffix
    return loss_filename, ground_truth_filename, model_result_filename


def compare():

    ls1_filename, gt1_filename, mr1_filename = get_PytorchLSTM_suffix(conf1)
    with open(gt1_filename) as f:
        gt1 = pickle.load(f)
    with open(mr1_filename) as f:
        mr1 = pickle.load(f)
    with open(ls1_filename) as f:
        ls1 = pickle.load(f)

#conf2['related'] = True
#conf2['topK'] = 3

	ls2_filename, gt2_filename, mr2_filename = get_MILSTM_suffix(conf2)
    with open(gt2_filename) as f:
        gt2 = pickle.load(f)
    with open(mr2_filename) as f:
        mr2 = pickle.load(f)
    with open(ls2_filename) as f:
        ls2 = pickle.load(f)

    plt.figure(figsize=(20, 10))
    plt.xticks(rotation=50)
    plt.xlabel('Time (week)', fontsize='large', fontweight='bold')
    plt.ylabel('Number of material conflict events', fontweight='bold', fontsize='large')
    plt.title('Prediction of the country pair: {}'.format(entity_pair))

    brk = 14
    plt.plot(range(brk + 1), gt2[:brk + 1], linewidth=2, color="green", linestyle="--" , label='First 15 weeks')
    plt.plot(range(brk, len(gt2)), gt1[brk:], linewidth=1.5, color="green", label='ground truth')
    plt.plot(range(brk, len(gt2)), mr1[brk:], linewidth=1.5, label='LSTM')
    plt.plot(range(brk, len(gt2)), mr2[brk:], linewidth=1.5, label='model2')

    plt.legend()
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.xticks(rotation=50)
    plt.xlabel('Epoch(s)')
    plt.ylabel('RMSE')
    plt.title('Root Mean Square Error (RMSE) of prediction of {}'.format(entity_pair))
    for i in range(len(ls1)):
        ls1[i] = math.sqrt(ls1[i])
    for i in range(len(ls2)):
        ls2[i] = math.sqrt(ls2[i])

    plt.plot(range(len(ls1)), ls1, label = 'LSTM')
    plt.plot(range(len(ls2)), ls2, label = 'MILSTM with PCC')
    plt.legend()
    plt.show()


def batch_compare_loss():

    pair_list = []
    country_list = ['USA', 'GBR', 'RUS', 'CHN', 'CAN', 'AUS', 'FRA', 'JPN']
    clen = len(country_list)
    for i in range(clen):
        for j in range(clen):
            if i == j:
                continue
            pair_list.append(country_list[i] + '2' + country_list[j])

   # improvement = 0
    improvement2 = 0
    count = 0

    for pair in pair_list:
        conf1['entity_pair'] = pair
        #conf2['entity_pair'] = pair
        conf3['entity_pair'] = pair
        ls1_filename, _, _ = get_PytorchLSTM_suffix(conf1)
        with open(ls1_filename) as f:
            ls1 = pickle.load(f)

        #conf2['related'] = True
        #ls2_filename, _, _ = get_MILSTM_suffix(conf2)
        #with open(ls2_filename) as f:
        #    ls2 = pickle.load(f)

        #conf2['related'] = True
        ls3_filename, _, _ = get_ELSTM_suffix(conf3)
        with open(ls3_filename) as f:
            ls3 = pickle.load(f)

        # l1 = (ls1[-1] + ls1[-2] + ls1[-3]) / 3.0;
        # l2 = (ls2[-1] + ls2[-2] + ls2[-3]) / 3.0;
        l1 = ls1[-13]
        #l2 = ls2[-11]
        l3 = ls3[-13]

        if l1 < 60:
            continue
        print 'entity_pair: {}, Loss of LSTM: {}, Loss of ELSTM: {}'.format(pair, l1, l3),
        #tmp = 1.0 * (l1 - l2) / l1
        #improvement += tmp

        tmp2 = 1.0 * (l1 - l3) / l1
        improvement2 += tmp2

        count += 1
        print 'Improvement: {} %'.format(round(tmp2, 3))

    #improvement /= 1.0 * coun
    improvement2 /= 1.0 * count
    print 'improvement2: {}%'.format(round(improvement2, 3))


if __name__ == "__main__":

    compare()
    #batch_compare_loss()

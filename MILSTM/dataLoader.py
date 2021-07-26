import pickle
import copy
import os
import sys
sys.path.append("/home/naturain/PycharmProjects/GDELT_prediction")
import torch
from utility.tools import dateParser, dateEqual, dateToString, dateToInt, nextDate
from torch.utils.data import Dataset, DataLoader

standard_filename = {
    'distance': 'top50CountryDistanceRank.pkl',
    'distance_unrelated': 'top50CountryBigDistanceRank.pkl',

    'correlation': 'top50CountryCorrelationRank.pkl',
    'correlation_unrelated': 'top50CountryNegCorrelationRank.pkl',
    
    'word2vec': 'top50CountryWvRank.pkl',
    'word2vec_unrelated': 'top50CountryNegWvRank.pkl'
}


def getDateList(start_date, end_date, conf):
    # generate a list consisting of event lists happened in each day
    start_y, start_m, start_d = dateParser(start_date)
    end_y, end_m, end_d = dateParser(end_date)

    current_y, current_m, current_d = start_y, start_m, start_d
    count = 0
    date_unit_list = []

    with open(conf['origin_path'] + 'top50CountryPairsPkl/' + conf['entity_pair'] + '.pkl') as f:
        data_dict = pickle.load(f)

    while dateEqual(current_y, current_m, current_d, end_y, end_m, end_d) == False:

        count += 1
        date_int = dateToInt(current_y, current_m, current_d)

        if data_dict.has_key(date_int):
            date_unit_list.append(data_dict[date_int])  # data_unit_list is a [ [], [], ... ]
        else:
            date_unit_list.append([])

        current_y, current_m, current_d = nextDate(current_y, current_m, current_d)

    return date_unit_list


def getTimeUnitList(date_unit_list, conf):
    # aggregate a series of date units into an unit with length equals to conf['time_unit']
    time_unit_list = []
    time_unit = []
    count = 0

    for item in date_unit_list:
        count += 1
        time_unit.extend(item)

        if count == conf['time_unit']:
            time_unit_list.append(time_unit)
            time_unit = []
            count = 0

    return time_unit_list


def check(event):
    # filter some low-mention events
    if event['NumMentions'] < 3:
        return False
    return True


def movingAverage(data_list, win_size):
    # smooth the curve
    result_list = []
    num_channel = len(data_list[0])
    sum = [0 for i in range(num_channel)]

    for i in range(len(data_list)):
        tmp = []
        for j in range(num_channel):
            sum[j] += data_list[i][j]
            if i >= win_size:
                sum[j] -= data_list[i - win_size][j]
                tmp.append(float(sum[j]) / win_size)
            else:
                tmp.append(float(sum[j]) / (i + 1))
        result_list.append(tmp)

    return result_list


def getCodeSequenceUnitList(start_date, end_date, conf):
    date_unit_list = getDateList(start_date, end_date, conf)
    time_unit_list = getTimeUnitList(date_unit_list, conf)
    code_time_unit_list = []

    # aggregate event codes of each time unit

    for idx in range(len(time_unit_list)):
        code_count = {1: 0, 2: 0, 3: 0, 4: 0}
        for event in time_unit_list[idx]:
            if check(event):
                event_code = event['QuadClass']
                code_count[event_code] += 1
        code_time_unit_list.append(code_count.values())

    for i in range(conf['moving_average'][1]):
        code_time_unit_list = movingAverage(code_time_unit_list, conf['moving_average'][0])

    code_sequence_unit_list = []

    for idx in range(len(code_time_unit_list) - conf['sequence_len'] + 1):
        # code_sequence_unit: [sequence_len, 4]
        code_sequence_unit = []
        for bias in range(conf['sequence_len']):
            code_sequence_unit.append(code_time_unit_list[idx + bias])
        code_sequence_unit_list.append(code_sequence_unit)

    return code_sequence_unit_list


def checkAndGet(tag, conf, start_date, end_date):
    prepared_filename = conf['origin_path'] + 'pickledData/' + 'pickled_' + \
                        conf['entity_pair'] + '_' + \
                        tag + '_' + \
                        str(conf['moving_average'][0]) + '_' + \
                        str(conf['moving_average'][1]) + '_' + \
                        str(conf['time_unit']) + '_' + \
                        str(conf['sequence_len']) + '.pkl'

    if os.path.isfile(prepared_filename):
        print 'Already have {}'.format(prepared_filename)
        with open(prepared_filename) as f:
            code_sequence_unit_list = pickle.load(f)
    else:
        code_sequence_unit_list = getCodeSequenceUnitList(start_date, end_date, conf)
        with open(prepared_filename, 'w') as f:
            pickle.dump(code_sequence_unit_list, f)
            print 'Pickled {}'.format(prepared_filename)

    return code_sequence_unit_list


def getMultiDataLoader(tag, conf, shuffle = True):
    start_date = conf[tag + '_start_date']
    end_date = conf[tag + '_end_date']

    code_sequence_unit_list = checkAndGet(tag, conf, start_date, end_date)
    code_sequence_unit_array = []
    code_sequence_unit_array.append(code_sequence_unit_list)

    rank_filename = './pickled/' + standard_filename[conf['standard'] + ('' if conf['related'] else '_unrelated')]
    with open(rank_filename) as f:
        rank = pickle.load(f)

    count = 0
    for i in range(20):
        if count >= conf['topK']:
            break
        count += 1
        new_conf = conf.copy()
        new_conf['entity_pair'] = rank[conf['entity_pair']][i][0]
        print 'Related pair: {}'.format(new_conf['entity_pair'])
        code_sequence_unit_array.append(checkAndGet(tag, new_conf, start_date, end_date))

    data_set = codeMultiSequenceDataset(code_sequence_unit_array)
    return DataLoader(data_set, batch_size = conf[tag + '_batch_size'], num_workers = 1, shuffle = shuffle, drop_last = True)


class codeMultiSequenceDataset(Dataset):
    def __init__(self, data_array):
        self.data_array = data_array
        self.len = len(data_array[0])
        self.dim = len(data_array)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = torch.FloatTensor(self.data_array[0][idx]).unsqueeze(0)
        for i in range(1, self.dim):
            data = torch.cat((data, torch.FloatTensor(self.data_array[i][idx]).unsqueeze(0)), 0)
        return data







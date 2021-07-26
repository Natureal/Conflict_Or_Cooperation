import pandas as pd
import pickle
import os
from tqdm import tqdm
from metadata import country_list
import sys

project_path = '/home/naturain/PycharmProjects/'

# select top 50 countries
clen = len(country_list)

all_country_pair = {}

attributes = ['GLOBALEVENTID', 'SQLDATE', 'Actor1Code', 'Actor2Code', 'Actor1Name', 'Actor2Name',
                'Actor1CountryCode', 'Actor2CountryCode', 'IsRootEvent', 'EventCode', 'EventBaseCode',
                'EventRootCode', 'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles',
                'AvgTone']

def createPairDict():

    for i in range(50):

        for j in range(50):

            if i == j:
                continue

            current_pair = country_list[i] + '2' + country_list[j]

            all_country_pair[current_pair] = 1


def run():

    createPairDict()

    for cid in tqdm(range(50)):

            print "cid: {}".format(cid)

            if cid < 10:

                current_path = project_path + 'gdelt_by_country/' + country_list[cid] + '/'

            else:

                if cid % 5 != 0:

                    continue

                current_path = project_path + 'gdelt_by_country/' + country_list[cid] + '_' \
                                                                  + country_list[cid + 1] + '_' \
                                                                  + country_list[cid + 2] + '_' \
                                                                  + country_list[cid + 3] + '_' \
                                                                  + country_list[cid + 4] + '/'

            for item in tqdm(os.listdir(current_path)):

                country_pairs_dict = {}

                print item

                fname = os.path.join(current_path + item)

                df = pd.read_csv(fname)

                for index, row in df.iterrows():

                    date = row['SQLDATE']

                    if date < 20000101:
                        continue

                    country_pair = str(row['Actor1CountryCode']) + '2' + str(row['Actor2CountryCode'])

                    if all_country_pair.has_key(country_pair) is False:
                        continue

                    if country_pairs_dict.has_key(country_pair) is False:
                        # create a dictionary for each country pair
                        country_pairs_dict[country_pair] = {}

                    # create a dictionary for each day
                    dict_day = {}

                    for attr in attributes:
                        # record each attribute
                        dict_day[attr] = row[attr]

                    if country_pairs_dict[country_pair].has_key(row['SQLDATE']) is False:
                        # country_pairs_dict is now a {{[], [], ..}, {}, {}, ....}, nested dictionary
                        # item is a list
                        country_pairs_dict[country_pair][row['SQLDATE']] = [dict_day]

                    else:

                        country_pairs_dict[country_pair][row['SQLDATE']].append(dict_day)

                for key, value in country_pairs_dict.items():
                    # key: country pair, value: dict
                    pkl_filename = project_path + 'top50CountryPairsPkl/' + key + '.pkl'

                    if os.path.isfile(pkl_filename):

                        with open(pkl_filename) as f:

                            pair_dict = pickle.load(f)

                            for sql_date, event_list in value.items():

                                if pair_dict.has_key(sql_date):

                                    pair_dict[sql_date].extend(event_list)

                                else:

                                    pair_dict[sql_date] = event_list

                        with open(pkl_filename, 'w') as f:

                            pickle.dump(pair_dict, f)

                    else:

                        with open(pkl_filename, 'w') as f:

                            pickle.dump(value, f)


if __name__ == '__main__':
    run()

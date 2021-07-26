


eventRootCode = {"1": "MAKE PUBLIC STATEMENT",
                 "2": "APPEAL",
                 "3": "EXPRESS INTENT TO COOPERATE",
                 "4": "CONSULT",
                 "5": "ENGAGE IN DIPLOMATIC COOPERATION",
                 "6": "ENGAGE IN MATERIAL COOPERATION",
                 "7": "PROVIDE AID",
                 "8": "YIELD",
                 "9": "INVESTIGATE",
                 "10": "DEMAND",
                 "11": "DISAPPROVE", # unrest
                 "12": "REJECT", # unrest
                 "13": "THREATEN", # unrest
                 "14": "PROTEST", # unrest
                 "15": "EXHIBIT FORCE POSTURE", # unrest
                 "16": "REDUCE RELATIONS", # unrest
                 "17": "COERCE", # unrest
                 "18": "ASSAULT", # unrest
                 "19": "FIGHT", # unrest
                 "20": "USE UNCONVENTIONAL MASS VIOLENCE" # unrest
                 }

country_list = ['USA', 'GBR', 'RUS', 'CHN', 'ISR', 'CAN', 'AUS', 'PAK', 'FRA', 'IRN', # 10
                'DEU', 'TUR', 'NGA', 'SYR', 'IND', 'JPN', 'EUR', 'IRQ', 'AFG', 'AFR', # 20
                'EGY', 'PSE', 'PHL', 'SAU', 'ITA', 'KOR', 'IRL', 'MEX', 'UKR', 'MYS', # 30
                'ESP', 'IDN', 'PRK', 'KEN', 'LBN', 'JOR', 'GRC', 'VNM', 'NZL', 'THA', # 40
                'ZAF', 'CHE', 'LBY', 'POL', 'NLD', 'BEL', 'BGD', 'SDN', 'CUB', 'UGA', # 50
                'BRA', 'ZWE', 'LKA', 'GHA', 'ARE', 'YEM', 'SOM', 'ARM', 'AZE', 'MMR', # 60
                'QAT', 'SGP', 'COL', 'SWE', 'WST', 'VEN', 'NPL', 'AUT', 'ETH', 'BGR', # 70
                'NOR', 'ARG', 'KHM', 'CZE', 'DNK', 'HUN', 'KWT', 'SEA', 'ZMB', 'BLR', # 80
                'HRV', 'PER', 'CHL', 'FIN', 'RWA', 'JAM', 'BHR', 'MDV', 'PRT', 'CYP', # 90
                'MAR', 'KAZ', 'TUN', 'ALB', 'TZA', 'DZA', 'TWN', 'HTI', 'AGO', 'SRB', # 100
                'LBR', 'VAT', 'FJI', 'COD', 'KGZ', 'TJK', 'NAM', 'OMN', 'MKD', 'SVK', # 110
                'MLT', 'UZB', 'ECU', 'TCD', 'PAN', 'SSD', 'LTU', 'NMR', 'BHS', 'MOZ', # 120
                'BOL', 'SLE', 'MWI', 'LVA', 'GTM', 'NIC', 'MLI', 'HND', 'EST', 'MCO', # 130
                'SLV', 'TKM', 'GUY', 'TTO', 'CIV', 'CMR', 'BWA', 'GMB', 'NER', 'BDI', # 140
                'MDA', 'SEN', 'LUX', 'ERI', 'LAO', 'BRN', 'CRI', 'MEA', 'COG', 'DOM', # 150
                'SAS', 'PNG', 'GIN', 'BLZ', 'MNG', 'SYC', 'ISL', 'BMU', 'GRD', 'CAF', # 160
                'URY', 'DJI', 'TMP', 'FSM', 'BTN', 'BFA', 'BEN', 'BRB', 'PRY', 'MRT', # 170
                'SWZ', 'MUS', 'MDG', 'TGO', 'LSO', 'PGS', 'WSM', 'CYM', 'GAB', 'LCA', # 180
                'WAF', 'SLB', 'SAF', 'VCT', 'KNA', 'TON', 'ATG', 'GEO', 'GNB', 'VUT', # 190
                'GNQ', 'NRU', 'ASA', 'MHL', 'SUR', 'HKG', 'DMA', 'COM', 'CPV', 'PLW', # 200
                'COK', 'KIR', 'MAC', 'ABW', 'STP', 'LIE', 'CRB', 'CAS', 'EAF', 'TUV', # 210
                'AIA', 'ROM', 'SMR', 'AND', 'SHN', 'SAM', 'LAM', 'SCN', 'WLF', 'NAF', # 220
                'CAU', 'PRI', 'EEU', 'BLK'] # 224
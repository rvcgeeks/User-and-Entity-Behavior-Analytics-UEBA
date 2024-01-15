
import csv, json
from os import path, makedirs
from datetime import datetime

# download r4.2.tar.bz2 from https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247?file=24856766
# and extract in Data directory
CERT_DATASET_PATH = 'Data/r4.2'

with open('Data/config.json', 'r') as fh:
    CONFIG = json.load(fh)

dt_format = '%m/%d/%Y %H:%M:%S'
start_time = CONFIG['monitor.timerange']['start']
if None != start_time:
    start_time = datetime.strptime(start_time, dt_format)
end_time = CONFIG['monitor.timerange']['end']
if None != end_time:
    end_time = datetime.strptime(end_time, dt_format)

for item in CONFIG['monitor.inputdata'].items():
    data_source, user_index = item[0], item[1]['user_index']
    print('processing %s ...' % data_source)
    csv_in_path = path.join(CERT_DATASET_PATH, data_source)
    bins = {}
    all_users = []

    for username,subconfig in CONFIG['monitor'].items():
        makedirs('Data/' + username, exist_ok = True)
        all_users.append(username)
        fh = open('Data/' + username + '/' + data_source, 'w', newline='')
        bins[username] = {
            'fh': fh,
            'csvwriter': csv.writer(fh),
            'fcnt': 0,
            'finished': False
        }
        print('Bin for %s opened...' % username)

    with open(csv_in_path, 'r', newline='') as fh_in:
        reader = csv.reader(fh_in)
        next(reader, None) # skip header
        cnt = 0
        for row in reader:
            username = row[user_index]
            if username in bins.keys():
                if bins[username]['finished'] == False:
                    if data_source == 'psychometric.csv':
                        bins[username]['csvwriter'].writerow(row)
                        bins[username]['fcnt'] += 1
                    else:
                        current_time = datetime.strptime(row[1], dt_format)
                        if start_time <= current_time and end_time >= current_time:
                            bins[username]['csvwriter'].writerow(row)
                            bins[username]['fcnt'] += 1
                        if end_time < current_time:
                            bins[username]['finished'] = True
            all_finished = True
            for uname in all_users:
                if bins[uname]['finished'] == False:
                    all_finished = False
            if all_finished:
                break
            cnt += 1
            print('[%d] %s => %s ...\r' % (cnt, row[1], ''.join(['%s->%s ' % (uname, bins[uname]['fcnt']) for uname in all_users])), end='')
        print()

    for uname, bin0 in bins.items():
        bin0['fh'].close()
        print('Bin for %s closed...' % uname)

print('\nDone!')

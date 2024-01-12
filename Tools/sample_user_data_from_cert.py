
import csv
from sys import argv
from os import path, makedirs

# download r4.2.tar.bz2 from https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247?file=24856766
# and extract alongside this file
CERT_DATASET_PATH = 'r4.2'

USER = argv[1]

OUTDIR = path.join('Data', USER)
makedirs(OUTDIR, exist_ok = True)

for item in [
        ('device.csv', 2, -1), # 2nd entry is column no at which is for user id, 3rd entry is Limit No of rows, specify -1 if all rows
        ('file.csv', 2, -1),
        ('logon.csv', 2, -1),
        ('email.csv', 2, -1),
        ('http.csv', 2, -1),
        ('psychometric.csv', 1, -1)
    ]:
    data_source, user_index, limit = item
    print('processing %s ...' % data_source)
    csv_in_path = path.join(CERT_DATASET_PATH, data_source)
    csv_out_path = path.join(OUTDIR, data_source)
    with open(csv_in_path, 'r', newline='') as fh_in:
        reader = csv.reader(fh_in)
        with open(csv_out_path, 'w', newline='') as fh_out:
            writer = csv.writer(fh_out)
            cnt = 0
            for row in reader:
                if row[user_index] == USER:
                    writer.writerow(row)
                cnt += 1
                print('processed %d rows...\r' % cnt, end='')
                if limit > 0:
                    if cnt >= limit:
                        break
            print('\nDone!')

'''
D:\CoreDSE\Data\sample_user_data_from_cert>py sample_user_data_from_cert.py MOH0273
processing device.csv ...
processed 405381 rows...
Done!
processing file.csv ...
processed 445582 rows...
Done!
processing logon.csv ...
processed 854860 rows...
Done!
processing email.csv ...
processed 2629980 rows...
Done!
processing http.csv ...
processed 28434424 rows...
Done!
processing psychometric.csv ...
processed 1001 rows...
Done!

D:\CoreDSE\Data\sample_user_data_from_cert>
'''

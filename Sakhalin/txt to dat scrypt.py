"""Скрипт переписывает данные из файлов в подневные записи."""
import os
import datetime

Deltadate = datetime.timedelta(days=1)
print('Filename: ')
txtfilename = input()
if not os.path.isdir("DataTXT_all"):
    os.mkdir("DataTXT_all")
with open(txtfilename, 'r') as file:
    for line in file:
        pressure = float(line.strip().split()[2])
        pressure = pressure * 133.32239023154
        pressure = ((pressure - 101020 - 1026 * 9.80665) / (1026 * 9.80665))
        date = datetime.datetime.strptime(line.strip().split()[0], '%d.%m.%y').date()
        filenamedate = datetime.datetime.strftime(date, '%Y.%m.%d')
        with open(f'DataTXT_all/0_Press_meters_{filenamedate}.dat', 'a') as rec:
            rec.write(f'{str(pressure)}\n')


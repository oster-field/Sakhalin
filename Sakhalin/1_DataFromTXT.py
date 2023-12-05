"""Чтение данных с датчика, конвертация .dat в .npy
Данные разбиваются на отдельные фрагменты формата (дата, запись), содержащие фиксированное число точек. (~20 минут).
Каждый файл содержит информацию о дате и времени записи, переписываются все данные с начала до конца, даже если в дне не
целое количество 20-минутных фрагментов."""
import numpy as np
import datetime
from functions import DateStart, DateEnd
from tqdm import tqdm
import pandas as pd
import os

if not os.path.isdir("Data"):
    os.mkdir("Data")
Pressure = np.arange(0)
ReadingsPerFile = int(open('DataTXT/INFO.dat').readlines()[2].strip()[
                      15:17]) * 1200  # Сколько точек будет в файле .npy, 20 мин. (Для 8 Гц 9600 точек это 20 минут)
Deltadate = datetime.timedelta(days=1)
pbar = tqdm(total=len(pd.date_range(DateStart, DateEnd).strftime('%d.%m').tolist()), desc="Progress: ", colour='green')

while DateStart <= DateEnd:
    counter = 0
    filename = 'DataTXT/15_Press_meters_' + DateStart.strftime('%Y.%m.%d') + '.dat'
    num_lines = len(open(filename).readlines())
    with open(filename, 'r') as file:
        for line in file:
            Pressure = np.append(Pressure, float(line.strip().replace(',', '.')))
            if (len(Pressure) == ReadingsPerFile) or (len(Pressure) == num_lines % ReadingsPerFile) and (
                    counter * ReadingsPerFile + num_lines % ReadingsPerFile == num_lines):
                np.save('Data/' + DateStart.strftime('%Y.%m.%d') + ' reading ' + str(counter + 1),
                        Pressure.astype(float))
                Pressure = np.arange(0)
                counter += 1
    pbar.update(1)
    DateStart += Deltadate

with open('Data/isconverted.txt', 'w') as file:
    file.write('Not converted')
with open('Data/istransformed.txt', 'w') as file:
    file.write('Not transformed')
with open('Data/isprocessed.txt', 'w') as file:
    file.write('Not processed')
with open('Data/dba.txt', 'w') as file:
    file.write('Not calculated')
with open('Data/dbl.txt', 'w') as file:
    file.write('Not calculated')


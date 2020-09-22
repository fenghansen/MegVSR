import os
import numpy as np

if __name__ == '__main__':
    with open('RRDB_6.txt', 'r') as f:
        files1 = [line[:-1] for line in f.readlines()]
    with open('slowfusion.txt', 'r') as f:
        files2 = [line[:-1] for line in f.readlines() if len(line)<=20]
    for i in range(min(len(files1), len(files2))):
        if files1[i] != files2[i]:
            print(files1[i], files2[i])
            del files2[i]
            print('error!!') 
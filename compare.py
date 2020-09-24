import os
import numpy as np

if __name__ == '__main__':
    root_dir = r'F:\datasets\MegVSR\city\original'
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        new_name = '0'+name[6:]
        new_path = os.path.join(root_dir, new_name)
        print(name, new_name)
        os.rename(path, new_path)
    # with open('RRDB_6.txt', 'r') as f:
    #     files1 = [line[:-1] for line in f.readlines()]
    # with open('slowfusion.txt', 'r') as f:
    #     files2 = [line[:-1] for line in f.readlines() if len(line)<=20]
    # for i in range(min(len(files1), len(files2))):
    #     if files1[i] != files2[i]:
    #         print(files1[i], files2[i])
    #         del files2[i]
    #         print('error!!') 
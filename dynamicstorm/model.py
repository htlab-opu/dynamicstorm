import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk

import glob
import sys


class Expt():
    """実験"""

    def __init__(self, target_dir):
        self.location = target_dir
        self.file_list = []
        self.get_file_list()
        self.filtered_file_list = []
        self.time_averaged_data = ''

    def get_file_list(self):
        """渡されたディレクトリ内に存在する csv ファイルをリストに格納する"""
        self.file_list = glob.glob(self.location + '/*.csv')
        if len(self.file_list) == 0:
            self.file_list = glob.glob(self.location + '/*/*.csv')
        if len(self.file_list) == 0:
            self.file_list = glob.glob(self.location + '/*/.*.csv')
        if len(self.file_list) == 0:
            print('No applicable data in directory.\nExit')
            sys.exit()

    def show_invalid_vector_example(self):
        """含まれる瞬時データの内100個が持つ誤ベクトル数を表示する"""
        print('invalid vector example')
        for i, file in enumerate(self.file_list):
            status = InstantData(file).get_data('Status')
            total_invalid_vector = np.sum(status == 1) + np.sum(status == 17)
            print('-', total_invalid_vector)
            if i > 100:
                break

    def filter_invalid_vector(self):
        pass


class InstantData():
    "””瞬時データ””"

    def __init__(self, file):
        self.file = file
        header_row = self.get_header_row()  # ヘッダ行数を取得
        self.dataframe = pd.read_csv(self.file, header_row)  # ファイルからデータを読み出し

    def get_header_row(self):
        """データのヘッダ行数を取得する """
        file = open(self.file, 'r')
        for i, line in enumerate(file):
            if line[0] == 'x':
                file.close()
                return i
        file.close()

    def get_data(self, label):
        return self.dataframe[label]

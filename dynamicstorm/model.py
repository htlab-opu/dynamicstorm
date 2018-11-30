import pandas as pd
import glob
import sys

from . import filter as dyfil
from . import statistics as dyst


class ExptSet:
    """実験"""

    def __init__(self, target_dir):
        self.file_list = []
        self.time_averaged_data = ''
        self.filter = dyfil.Filter()
        self.statistics = dyst.Statistics()
        if not target_dir == '':
            """
            インスタンス生成時にディレクトリが
            指定された場合は指定されたディレクトリ内のファイルを瞬時データとして保持する
            指定されなかった場合は空とする
            """
            self.get_file_list(target_dir)

    def get_file_list(self, location):
        """渡されたディレクトリ内に存在する csv ファイルをリストに格納する"""
        self.file_list = glob.glob(location + '/*.csv')
        if len(self.file_list) == 0:
            self.file_list = glob.glob(location + '/*/*.csv')
        if len(self.file_list) == 0:
            self.file_list = glob.glob(location + '/*/.*.csv')
        if len(self.file_list) == 0:
            print('No applicable data in directory.\nExit')
            sys.exit()

    def filter(self, filter):
        self.file_list = self.filter.filter_incorrect_vector(self.file_list, filter)

    def time_averaging(self):
        self.statistics.time_averaging(self.file_list)

    def join(self, expt_instance_list):
        """
        複数の実験データを結合してまとめて取り扱えるようにする
        時間平均，空間平均済みのデータがある場合は結合後のデータで上書きする
        """
        for expt_instance in expt_instance_list:
            self.file_list.append(expt_instance.filelist)
        # TODO: 平均済みのデータがあるものとないものを結合する場合の考慮 → 削除でよさげ

class InstantData:
    """瞬時データ"""

    def __init__(self, file):
        self.file = file
        header_row = self.get_header_row(file)  # ヘッダ行数を取得

    @staticmethod
    def get_header_row(file):
        """データのヘッダ行数を取得する"""
        file = open(file, 'r')
        for i, line in enumerate(file):
            if line[0] == 'x':
                file.close()
                return i
        file.close()

    def get_data(self, label):
        """指定したラベルのデータ列を取り出す"""
        return self.df[label]

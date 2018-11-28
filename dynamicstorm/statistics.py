import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from . import model as dymod
from . import filter as dyfil


class Statistics:
    """統計処理"""

    def __init__(self):
        self.time_averaged_data = ''
        self.time_spatial_averaged_data = ''

    def time_averaging(self, file_list):
        """瞬時データを時間平均する"""
        for file in tqdm(file_list, desc='time averaging'):
            df = dymod.InstantData(file)
            Utmp = df.get_data('U[m/s]')
            Vtmp = df.get_data('V[m/s]')
            n = df.get_data('Status')
            try:
                N = N + n
                U = U + Utmp
                V = V + Vtmp
                uu = uu + Utmp ** 2 * n
                vv = vv + Vtmp ** 2 * n
                uv = uv + Utmp * Vtmp * n
                uuu = uuu + Utmp ** 3 * n
                vvv = vvv + Vtmp ** 3 * n
                uuv = uuv + Utmp ** 2 * Vtmp * n
                uvv = uvv + Utmp * Vtmp ** 2 * n
            except:
                N = n
                U = Utmp
                V = Vtmp
                uu = Utmp ** 2 * n
                vv = Vtmp ** 2 * n
                uv = Utmp * Vtmp * n
                uuu = Utmp ** 3 * n
                vvv = Vtmp ** 3 * n
                uuv = Utmp ** 2 * Vtmp * n
                uvv = Utmp * Vtmp ** 2 * n
        np.seterr(all='ignore')  # N = 0 の場合，割った値が nan になるようにする
        U = U / N
        V = V / N
        uu = uu / N
        vv = vv / N
        uv = uv / N
        uuu = uuu / N
        vvv = vvv / N
        uuv = uuv / N
        uvv = uvv / N

        U[np.isnan(U)] = 0  # NaN になった部分をを 0 にする
        V[np.isnan(V)] = 0
        uu[np.isnan(uu)] = 0
        vv[np.isnan(vv)] = 0
        uv[np.isnan(uv)] = 0
        uuu[np.isnan(uuu)] = 0
        vvv[np.isnan(vvv)] = 0
        uuv[np.isnan(uuv)] = 0
        uvv[np.isnan(uvv)] = 0

        cuu = uu
        cvv = vv
        cuv = uv
        uu = np.sqrt(uu - U ** 2)
        vv = np.sqrt(vv - V ** 2)
        uv = uv - U * V
        uuu = uuu - 3 * U * cuu + 2 * U ** 3
        vvv = vvv - 3 * U * cvv + 2 * V ** 3
        uuv = uuv - V * cuu - 2 * U * cuv + 2 * U ** 2 * V
        uvv = uvv - U * cvv - 2 * V * cuv + 2 * U * V ** 2

        # DynamicStudio の形式を真似て保存
        header = self.get_header_row(file_list[0])
        df = pd.read_csv(file_list[0], header=header)
        self.time_averaged_data = pd.DataFrame({'x (mm)[mm]': df['x (mm)[mm]'],
                                                'y (mm)[mm]': df['y (mm)[mm]'],
                                                'U[m/s]': U,
                                                'V[m/s]': V,
                                                'Std dev (U)[m/s]': uu,
                                                'Std dev (V)[m/s]': vv,
                                                'uuu': uuu,
                                                'vvv': vvv,
                                                'uuv': uuv,
                                                'uvv': uvv,
                                                'Covar (U': uv,
                                                'N': N
                                                })

    def save_time_averaged_data(self, filename):
        self.time_averaged_data.to_csv(filename)

    def read_time_averaged_data(self, file):
        """以前保存した解析済みのデータを読み出す"""
        header = self.get_header_row(file)
        self.time_averaged_data = pd.read_csv(file, header=header)

    def spatial_averaging(self, gridshape, range):
        """
        時間平均済みのデータを空間平均する
        range: [xmin, xmax, ymin, ymax]
        """
        xmin, xmax, ymin, ymax = range
        self.time_spatial_averaged_data

    def reshape_data(self, gridshape):
        """
        1 次元配列を 2 次元配列に変換する
        gridshape: [ygrid,xgrid]
        """
        pass

    @staticmethod
    def get_header_row(file):
        """ファイルのヘッダ行数を取得する"""
        file = open(file, 'r')
        for i, line in enumerate(file):
            if line[0] == 'x':
                file.close()
                return i
        file.close()

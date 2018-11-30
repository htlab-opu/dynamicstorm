import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from . import model as dymod
from . import filter as dyfil


class Statistics:
    """統計処理"""

    def __init__(self):
        self.time_averaged_data_frame = ''
        self.time_spatial_averaged_data_frame = ''

    def time_averaging(self, file_list):
        """瞬時データを時間平均する"""
        # 並列計算
        print('Time averaging...')
        total_core = mp.cpu_count()
        pool = mp.Pool(total_core)
        args = [(file_list, total_core, i) for i in range(total_core)]
        callback = pool.map(parallel_task, args)

        # 並列計算の結果を統合
        for [U_each, V_each,
             uu_each, vv_each, uv_each,
             uuu_each, vvv_each, uuv_each, uvv_each,
             N_each] in callback:
            try:
                U = U + U_each
                V = V + V_each
                uu = uu + uu_each
                vv = vv + vv_each
                uv = uv + uv_each
                uuu = uuu + uuu_each
                vvv = vvv + vvv_each
                uuv = uuv + uuv_each
                uvv = uvv + uvv_each
                N = N + N_each
            except:
                N = N_each
                U = U_each
                V = V_each
                uu = uu_each
                vv = vv_each
                uv = uv_each
                uuu = uuu_each
                vvv = vvv_each
                uuv = uuv_each
                uvv = uvv_each

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
        header = dymod.InstantData.get_header_row(file_list[0])
        df = pd.read_csv(file_list[0], header=header)
        self.time_averaged_data_frame = pd.DataFrame({'x (mm)[mm]': df['x (mm)[mm]'],
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
        print('Finish!\nTime averaging completed.')

    def save_time_averaged_data(self, filename):
        """時間平均済みデータを保存する"""
        self.time_averaged_data_frame.to_csv(filename)

    def read_time_averaged_data(self, file):
        """以前保存した解析済みのデータを読み出す"""
        header = self.get_header_row(file)
        self.time_averaged_data_frame = pd.read_csv(file, header=header)
        # TODO: 保存済みの時間平均データを読み出す処理の実装

    def spatial_averaging(self, grid_shape, xy_range):
        """
        時間平均済みのデータを空間平均する
        xy_range: [x_min, x_max, y_min, y_max]
        """
        x_min, x_max, y_min, y_max = range
        self.time_spatial_averaged_data_frame
        # TODO: 空間平均処理の実装

    def reshape_data(self, grid_shape):
        """
        1 次元配列を 2 次元配列に変換する
        grid_shape: [y_grid,x_grid]
        """
        pass

    def save_spatial_averaged_data(self, filename):
        """空間平均済みデータを保存する"""
        self.time_spatial_averaged_data_frame.to_csv(filename)

    def read_time_averaged_data(self, filename):
        """以前保存した空間平均済みのデータを読み出す"""
        pass
        # TODO: 保存済みの空間平均データを読み出す処理の実装


"""
    @staticmethod
    def get_header_row(file):
#ファイルのヘッダ行数を取得する
        file = open(file, 'r')
        for i, line in enumerate(file):
            if line[0] == 'x':
                file.close()
                return i
        file.close()
"""


def parallel_task(args):
    """並列計算タスク"""
    file_list, total_core, current_core = args
    file_count = len(file_list)
    start = int(file_count * current_core / total_core)
    end = int(file_count * (current_core + 1) / total_core) - 1
    header = dymod.InstantData.get_header_row(file_list[0])
    if current_core == 0:
        text = 'time averaging task ' + '1/' + str(total_core)
        for i in tqdm(range(start, end), desc=text):
            df = pd.read_csv(file_list[i], header=header)
            U_tmp = df['U[m/s]'].values
            V_tmp = df['V[m/s]'].values
            n = ((df['Status'] == 0) * 1).values
            try:
                N = N + n
                U = U + U_tmp
                V = V + V_tmp
                uu = uu + U_tmp ** 2 * n
                vv = vv + V_tmp ** 2 * n
                uv = uv + U_tmp * V_tmp * n
                uuu = uuu + U_tmp ** 3 * n
                vvv = vvv + V_tmp ** 3 * n
                uuv = uuv + U_tmp ** 2 * V_tmp * n
                uvv = uvv + U_tmp * V_tmp ** 2 * n
            except:
                N = n
                U = U_tmp
                V = V_tmp
                uu = U_tmp ** 2 * n
                vv = V_tmp ** 2 * n
                uv = U_tmp * V_tmp * n
                uuu = U_tmp ** 3 * n
                vvv = V_tmp ** 3 * n
                uuv = U_tmp ** 2 * V_tmp * n
                uvv = U_tmp * V_tmp ** 2 * n
        print('Wait till other {} parallel tasks will have finished.'.format(total_core))
    else:
        for i in range(start, end):
            df = pd.read_csv(file_list[i], header=header)
            U_tmp = df['U[m/s]'].values
            V_tmp = df['V[m/s]'].values
            n = ((df['Status'] == 0) * 1).values
            try:
                N = N + n
                U = U + U_tmp
                V = V + V_tmp
                uu = uu + U_tmp ** 2 * n
                vv = vv + V_tmp ** 2 * n
                uv = uv + U_tmp * V_tmp * n
                uuu = uuu + U_tmp ** 3 * n
                vvv = vvv + V_tmp ** 3 * n
                uuv = uuv + U_tmp ** 2 * V_tmp * n
                uvv = uvv + U_tmp * V_tmp ** 2 * n
            except:
                N = n
                U = U_tmp
                V = V_tmp
                uu = U_tmp ** 2 * n
                vv = V_tmp ** 2 * n
                uv = U_tmp * V_tmp * n
                uuu = U_tmp ** 3 * n
                vvv = V_tmp ** 3 * n
                uuv = U_tmp ** 2 * V_tmp * n
                uvv = U_tmp * V_tmp ** 2 * n
    return U, V, uu, vv, uv, uuu, vvv, uuv, uvv, N

import pandas as pd
import glob
import sys
import numpy as np
from scipy import interpolate
import cv2
import multiprocessing as mp
from tqdm import tqdm

from . import filtering as dyfil
from .datalabel import label_dict


class ExptSet:
    """瞬時データの集まり"""

    def __init__(self, target_dir=''):
        self.instant_data_list = []
        if not target_dir == '':
            """
            インスタンス生成時にディレクトリが
            指定された場合は指定されたディレクトリ内のファイルを瞬時データとして保持する
            指定されなかった場合は空とする
            """
            self.get_file_list(target_dir)

    def get_file_list(self, location):
        """渡されたディレクトリ内に存在する csv ファイルをリストに格納する"""
        self.instant_data_list = glob.glob(location + '/*.csv')
        if len(self.instant_data_list) == 0:
            self.instant_data_list = glob.glob(location + '/*/*.csv')
        if len(self.instant_data_list) == 0:
            self.instant_data_list = glob.glob(location + '/*/.*.csv')
        if len(self.instant_data_list) == 0:
            print('No applicable data in {}.'.format(location))
            sys.exit()

    def incorrect_vector_filter(self, filter_value):
        self.instant_data_list = dyfil.Filter().filter_incorrect_vector(
            self.instant_data_list, filter_value)

    def get_incorrect_vector_example(self, example_number):
        return dyfil.Filter().get_incorrect_vector_example(
            self.instant_data_list, example_number
        )

    def show_incorrect_vector_example(self, example_number):
        dyfil.Filter().show_incorrect_vector_example(
            self.instant_data_list, example_number
        )

    def join(self, instant_data_list_list):
        """
        複数の実験データを結合してまとめて取り扱えるようにする
        """
        if type(instant_data_list_list[0]) is list:  # リストを一つだけ渡された場合
            for instant_data_list in instant_data_list_list:
                self.instant_data_list.extend(instant_data_list)
        else:  # リストを複数渡された場合
            self.instant_data_list.extend(instant_data_list_list)


class Statistics:
    """時間平均データ"""

    def __init__(self, instant_data_list=None, source_file=None):
        self.time_averaged_data_frame = None
        self.array_2d = None
        if instant_data_list is not None:
            self.time_averaging(instant_data_list)
        elif source_file is not None:
            self.read(source_file)

    def time_averaging(self, file_list):
        """瞬時データを時間平均する"""
        # 並列計算
        print('Time averaging...')
        total_core = mp.cpu_count()
        pool = mp.Pool(total_core)
        args = [(file_list, total_core, i) for i in range(total_core)]
        callback = pool.map(time_averaging_parallel_task, args)

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

        np.seterr(all='ignore')
        U = U / N
        V = V / N
        uu = uu / N
        vv = vv / N
        uv = uv / N
        uuu = uuu / N
        vvv = vvv / N
        uuv = uuv / N
        uvv = uvv / N

        cuu = uu
        cvv = vv
        cuv = uv
        cuuu = uuu
        cvvv = vvv
        cuuv = uuv
        cuvv = uvv

        uu = np.sqrt(cuu - U ** 2)
        vv = np.sqrt(cvv - V ** 2)
        uv = cuv - U * V
        uuu = cuuu - 3 * U * cuu + 2 * U ** 3
        vvv = cvvv - 3 * V * cvv + 2 * V ** 3
        uuv = cuuv - V * cuu - 2 * U * cuv + 2 * U ** 2 * V
        uvv = cuvv - U * cvv - 2 * V * cuv + 2 * U * V ** 2

        U[np.isnan(U)] = 0
        V[np.isnan(V)] = 0
        uu[np.isnan(uu)] = 0
        vv[np.isnan(vv)] = 0
        uv[np.isnan(uv)] = 0
        uuu[np.isnan(uuu)] = 0
        vvv[np.isnan(vvv)] = 0
        uuv[np.isnan(uuv)] = 0
        uvv[np.isnan(uvv)] = 0

        U[np.isinf(U)] = 0
        V[np.isinf(V)] = 0
        uu[np.isinf(uu)] = 0
        vv[np.isinf(vv)] = 0
        uv[np.isinf(uv)] = 0
        uuu[np.isinf(uuu)] = 0
        vvv[np.isinf(vvv)] = 0
        uuv[np.isinf(uuv)] = 0
        uvv[np.isinf(uvv)] = 0

        # DynamicStudio の形式を真似て保存
        header = InstantData.get_header_row(file_list[0])
        df = pd.read_csv(file_list[0], header=header)
        self.time_averaged_data_frame = pd.DataFrame(
            {label_dict['x']['label']: df[label_dict['x']['label']],
             label_dict['y']['label']: df[label_dict['y']['label']],
             label_dict['U']['label']: U,
             label_dict['V']['label']: V,
             label_dict['u']['label']: uu,
             label_dict['v']['label']: vv,
             label_dict['uuu']['label']: uuu,
             label_dict['vvv']['label']: vvv,
             label_dict['uuv']['label']: uuv,
             label_dict['uvv']['label']: uvv,
             label_dict['uv']['label']: uv,
             label_dict['N']['label']: N
             })
        print('Finish!\nTime averaging completed.')

    def save(self, file_name):
        """時間平均済みデータを保存する"""
        self.time_averaged_data_frame.to_csv(file_name, index=False)

    def read(self, file_name):
        """以前保存した解析済みのデータを読み出す"""
        header = self.get_header_row(file_name)
        self.time_averaged_data_frame = pd.read_csv(file_name, header=header)

    def join(self, time_averaged_data_frame_list):
        """複数実験セットのデータを統合する"""
        np.seterr(all='ignore')
        if self.time_averaged_data_frame is None:  # オブジェクトが空の場合
            if type(time_averaged_data_frame_list) is list:  # 複数のデータフレームを渡された場合
                self.time_averaged_data_frame = time_averaged_data_frame_list[0].copy()  # 最初のループの処理
                for i, time_averaged_data_frame in enumerate(time_averaged_data_frame_list):
                    if i == 0: continue  # 最初のループ分の処理をスキップ
                    df1 = self.time_averaged_data_frame.copy()
                    N1 = df1[label_dict['N']['label']].copy()
                    df2 = time_averaged_data_frame
                    N2 = df2[label_dict['N']['label']].copy()
                    for text in ['U', 'V', 'uuu', 'vvv', 'uuv', 'uvv', 'uv']:
                        a1 = df1[label_dict[text]['label']].copy()
                        a2 = df2[label_dict[text]['label']].copy()
                        data = (a1 * N1 + a2 * N2) / (N1 + N2)
                        data[np.isnan(data)] = 0
                        data[np.isinf(data)] = 0
                        self.time_averaged_data_frame[label_dict[text]['label']] = data.copy()
                    for text in ['u', 'v']:
                        a1 = df1[label_dict[text]['label']].copy()
                        a2 = df2[label_dict[text]['label']].copy()
                        data = np.sqrt((a1 ** 2 * N1 + a2 ** 2 * N2) / (N1 + N2))
                        data[np.isnan(data)] = 0
                        data[np.isinf(data)] = 0
                        self.time_averaged_data_frame[label_dict[text]['label']] = data.copy()
                    self.time_averaged_data_frame[label_dict['N']['label']] = N1 + N2

            else:  # データフレームを一つだけ渡された場合
                time_averaged_data_frame = time_averaged_data_frame_list.copy()
                self.time_averaged_data_frame = time_averaged_data_frame.copy()

        else:
            if type(time_averaged_data_frame_list) is list:
                for time_averaged_data_frame in time_averaged_data_frame_list:
                    df1 = self.time_averaged_data_frame.copy()
                    N1 = df1[label_dict['N']['label']].copy()
                    df2 = time_averaged_data_frame.copy()
                    N2 = df2[label_dict['N']['label']].copy()
                    for text in ['U', 'V', 'uuu', 'vvv', 'uuv', 'uvv', 'uv']:
                        a1 = df1[label_dict[text]['label']].copy()
                        a2 = df2[label_dict[text]['label']].copy()
                        data = (a1 * N1 + a2 * N2) / (N1 + N2)
                        data[np.isnan(data)] = 0
                        data[np.isinf(data)] = 0
                        self.time_averaged_data_frame[label_dict[text]['label']] = data.copy()
                    for text in ['u', 'v']:
                        a1 = df1[label_dict[text]['label']].copy()
                        a2 = df2[label_dict[text]['label']].copy()
                        data = np.sqrt((a1 ** 2 * N1 + a2 ** 2 * N2) / (N1 + N2))
                        data[np.isnan(data)] = 0
                        data[np.isinf(data)] = 0
                        self.time_averaged_data_frame[label_dict[text]['label']] = data.copy()
                    self.time_averaged_data_frame[label_dict['N']['label']] = N1 + N2
            else:
                df1 = self.time_averaged_data_frame
                N1 = df1[label_dict['N']['label']].copy()
                df2 = time_averaged_data_frame_list
                N2 = df2[label_dict['N']['label']].copy()
                for text in ['U', 'V', 'uuu', 'vvv', 'uuv', 'uvv', 'uv']:
                    a1 = df1[label_dict[text]['label']].copy()
                    a2 = df2[label_dict[text]['label']].copy()
                    data = (a1 * N1 + a2 * N2) / (N1 + N2)
                    data[np.isnan(data)] = 0
                    data[np.isinf(data)] = 0
                    self.time_averaged_data_frame[label_dict[text]['label']] = data.copy()
                for text in ['u', 'v']:
                    a1 = df1[label_dict[text]['label']].copy()
                    a2 = df2[label_dict[text]['label']].copy()
                    data = np.sqrt((a1 ** 2 * N1 + a2 ** 2 * N2) / (N1 + N2))
                    data[np.isnan(data)] = 0
                    data[np.isinf(data)] = 0
                    self.time_averaged_data_frame[label_dict[text]['label']] = data.copy()
                self.time_averaged_data_frame[label_dict['N']['label']] = N1 + N2

    @staticmethod
    def get_header_row(file_name):
        # ファイルのヘッダ行数を取得する
        file = open(file_name, 'r')
        for i, line in enumerate(file):
            if line[0] == 'x':
                file.close()
                return i
        file.close()


def time_averaging_parallel_task(args):
    """並列計算タスク"""
    file_list, total_core, current_core = args
    file_count = len(file_list)
    start = int(file_count * current_core / total_core)
    end = int(file_count * (current_core + 1) / total_core) - 1
    header = InstantData.get_header_row(file_list[0])
    text = 'time averaging task ' + \
           str(current_core + 1) + '/' + str(total_core)

    # 全て 0 の配列を用意
    df = pd.read_csv(file_list[0], header=header)
    n = ((df[label_dict['Status']['label']] == 0) * 1).values
    N = n * 0
    U = N
    V = N
    uu = N
    vv = N
    uv = N
    uuu = N
    vvv = N
    uuv = N
    uvv = N

    for i in tqdm(range(start, end), desc=text):
        df = pd.read_csv(file_list[i], header=header)
        U_tmp = df[label_dict['U']['label']].values
        V_tmp = df[label_dict['V']['label']].values
        n = ((df[label_dict['Status']['label']] == 0) * 1).values
        N = N + n
        U = U + U_tmp * n
        V = V + V_tmp * n
        uu = uu + U_tmp ** 2 * n
        vv = vv + V_tmp ** 2 * n
        uv = uv + U_tmp * V_tmp * n
        uuu = uuu + U_tmp ** 3 * n
        vvv = vvv + V_tmp ** 3 * n
        uuv = uuv + U_tmp ** 2 * V_tmp * n
        uvv = uvv + U_tmp * V_tmp ** 2 * n
    return U, V, uu, vv, uv, uuu, vvv, uuv, uvv, N


class Array2d:
    """時間平均データの必要な部分だけ取り出した 2 次元配列データ"""

    def __init__(self, data_frame=None, grid_shape=[74, 101], crop_range=['', '', '', ''], size=[80, 80]):
        self.array_2d_dict = None
        if data_frame is not None:
            self.crop_array_2d(data_frame, grid_shape, crop_range, size)

    def crop_array_2d(self, time_averaged_data_frame, grid_shape=[74, 101], crop_range=['', '', '', ''], size=[80, 80]):
        size = (size[0], size[1])
        df = time_averaged_data_frame
        x_min_index, x_max_index, y_min_index, y_max_index = get_crop_index(df,
                                                                            grid_shape,
                                                                            crop_range)
        U = cv2.resize(df[label_dict['U']['label']].values.reshape(grid_shape)[y_min_index:y_max_index + 1,
                       x_min_index:x_max_index + 1], size, interpolation=cv2.INTER_CUBIC)
        V = cv2.resize(df[label_dict['V']['label']].values.reshape(grid_shape)[y_min_index:y_max_index + 1,
                       x_min_index:x_max_index + 1], size, interpolation=cv2.INTER_CUBIC)
        u = cv2.resize(df[label_dict['u']['label']].values.reshape(grid_shape)[y_min_index:y_max_index + 1,
                       x_min_index:x_max_index + 1], size, interpolation=cv2.INTER_CUBIC)
        v = cv2.resize(df[label_dict['v']['label']].values.reshape(grid_shape)[y_min_index:y_max_index + 1,
                       x_min_index:x_max_index + 1], size, interpolation=cv2.INTER_CUBIC)
        uv = cv2.resize(df[label_dict['uv']['label']].values.reshape(grid_shape)[y_min_index:y_max_index + 1,
                        x_min_index:x_max_index + 1], size, interpolation=cv2.INTER_CUBIC)
        uuu = cv2.resize(df[label_dict['uuu']['label']].values.reshape(grid_shape)[y_min_index:y_max_index + 1,
                         x_min_index:x_max_index + 1], size, interpolation=cv2.INTER_CUBIC)
        vvv = cv2.resize(df[label_dict['vvv']['label']].values.reshape(grid_shape)[y_min_index:y_max_index + 1,
                         x_min_index:x_max_index + 1], size, interpolation=cv2.INTER_CUBIC)
        uuv = cv2.resize(df[label_dict['uuv']['label']].values.reshape(grid_shape)[y_min_index:y_max_index + 1,
                         x_min_index:x_max_index + 1], size, interpolation=cv2.INTER_CUBIC)
        uvv = cv2.resize(df[label_dict['uvv']['label']].values.reshape(grid_shape)[y_min_index:y_max_index + 1,
                         x_min_index:x_max_index + 1], size, interpolation=cv2.INTER_CUBIC)

        x = np.linspace(0, 1, size[0])
        y = np.linspace(0, 1, size[1])
        x, y = np.meshgrid(x, y)
        N = cv2.resize(df[label_dict['N']['label']].values.reshape(grid_shape)[y_min_index:y_max_index + 1,
                       x_min_index:x_max_index + 1], size, interpolation=cv2.INTER_NEAREST)
        self.array_2d_dict = {
            'x': x,
            'y': y,
            'U': U,
            'V': V,
            'u': u,
            'v': v,
            'uv': uv,
            'uuu': uuu,
            'vvv': vvv,
            'uuv': uuv,
            'uvv': uvv,
            'N': N,
        }

    def join(self, array_2d_dict_list):
        np.seterr(all='ignore')
        if self.array_2d_dict is None:  # オブジェクトが空の場合
            if type(array_2d_dict_list) is list:  # 複数のデータフレームを渡された場合
                self.array_2d_dict = array_2d_dict_list[0]  # 最初のループの処理
                for i, array_2d_dict in enumerate(array_2d_dict_list):
                    if i == 0: continue  # 最初のループ分の処理をスキップ
                    array_2d_dict1 = self.array_2d_dict
                    N1 = array_2d_dict1['N']
                    array_2d_dict2 = array_2d_dict
                    N2 = array_2d_dict2['N']
                    for text in ['U', 'V', 'uuu', 'vvv', 'uuv', 'uvv', 'uv']:
                        a1 = array_2d_dict1[text]
                        a2 = array_2d_dict2[text]
                        data = (a1 * N1 + a2 * N2) / (N1 + N2)
                        data[np.isnan(data)] = 0
                        data[np.isinf(data)] = 0
                        self.array_2d_dict[text] = data
                    for text in ['u', 'v']:
                        a1 = array_2d_dict1[text]
                        a2 = array_2d_dict2[text]
                        data = np.sqrt((a1 ** 2 * N1 + a2 ** 2 * N2) / (N1 + N2))
                        data[np.isnan(data)] = 0
                        data[np.isinf(data)] = 0
                        self.array_2d_dict[text] = data
                    self.array_2d_dict['N'] = N1 + N2

            else:  # データフレームを一つだけ渡された場合
                self.array_2d_dict = array_2d_dict_list

        else:  # オブジェクトが既に存在する場合
            if type(array_2d_dict_list) is list:  # 複数のデータを渡された場合
                for array_2d_dict in array_2d_dict_list:
                    array_2d_dict1 = self.array_2d_dict
                    N1 = array_2d_dict1['N']
                    array_2d_dict2 = array_2d_dict
                    N2 = array_2d_dict2['N']
                    for text in ['U', 'V', 'uuu', 'vvv', 'uuv', 'uvv', 'uv']:
                        a1 = array_2d_dict1[text]
                        a2 = array_2d_dict2[text]
                        data = (a1 * N1 + a2 * N2) / (N1 + N2)
                        data[np.isnan(data)] = 0
                        data[np.isinf(data)] = 0
                        self.array_2d_dict[text] = data
                    for text in ['u', 'v']:
                        a1 = array_2d_dict1[text]
                        a2 = array_2d_dict2[text]
                        data = np.sqrt((a1 ** 2 * N1 + a2 ** 2 * N2) / (N1 + N2))
                        data[np.isnan(data)] = 0
                        data[np.isinf(data)] = 0
                        self.array_2d_dict[text] = data
                    self.array_2d_dict['N'] = N1 + N2
            else:
                array_2d_dict1 = self.array_2d_dict
                N1 = array_2d_dict1['N']
                array_2d_dict2 = array_2d_dict_list
                N2 = array_2d_dict2['N']
                for text in ['U', 'V', 'uuu', 'vvv', 'uuv', 'uvv', 'uv']:
                    a1 = array_2d_dict1[text]
                    a2 = array_2d_dict2[text]
                    data = (a1 * N1 + a2 * N2) / (N1 + N2)
                    data[np.isnan(data)] = 0
                    data[np.isinf(data)] = 0
                    self.array_2d_dict[text] = data
                for text in ['u', 'v']:
                    a1 = array_2d_dict1[text]
                    a2 = array_2d_dict2[text]
                    data = np.sqrt((a1 ** 2 * N1 + a2 ** 2 * N2) / (N1 + N2))
                    data[np.isnan(data)] = 0
                    data[np.isinf(data)] = 0
                    self.array_2d_dict[text] = data
                self.array_2d_dict['N'] = N1 + N2


class SpaceAverage:
    """時間平均かつ空間平均データ"""

    def __init__(self, data_frame=None, source_file=None, array_2d_dict=None,
                 grid_shape=[74, 101], crop_range=['', '', '', ''], size=80):
        self.space_averaged_data_frame = None
        self.grid_shape = grid_shape
        self.crop_range = []
        self.set_range(crop_range)
        if source_file is not None:
            self.read(source_file)
        elif data_frame is not None:
            self.space_averaging(data_frame, size)
        elif array_2d_dict is not None:
            self.space_averaging_from_array_2d(array_2d_dict, size)

    def save(self, file_name):
        """空間平均済みデータを保存する"""
        self.space_averaged_data_frame.to_csv(file_name, index=False)

    def read(self, file_name):
        """以前保存した解析済みのデータを読み出す"""
        header = self.get_header_row(file_name)
        self.space_averaged_data_frame = pd.read_csv(file_name, header=header)

    def set_range(self, crop_range):
        x_min_mm, x_max_mm, y_min_mm, y_max_mm = crop_range
        if x_min_mm == '': x_min_mm = 0
        if x_max_mm == '': x_max_mm = float('inf')
        if y_min_mm == '': y_min_mm = 0
        if y_max_mm == '': y_max_mm = float('inf')

        x_min_mm = float(x_min_mm)
        x_max_mm = float(x_max_mm)
        y_min_mm = float(y_min_mm)
        y_max_mm = float(y_max_mm)
        self.crop_range = x_min_mm, x_max_mm, y_min_mm, y_max_mm

    def space_averaging(self, time_averaged_data_frame, size):
        df = time_averaged_data_frame
        y = df[label_dict['y']['label']].values.reshape(self.grid_shape)[:, 0]

        x_min_index, x_max_index, y_min_index, y_max_index = get_crop_index(time_averaged_data_frame,
                                                                            self.grid_shape,
                                                                            self.crop_range)
        y = y - y[y_min_index]  # 原点を合わせる
        y = y[y_min_index:y_max_index + 1]  # 必要な範囲だけ取り出す

        # range 内のデータを取り出す
        U_tmp = df[label_dict['U']['label']].values.reshape(self.grid_shape)[y_min_index:y_max_index + 1,
                x_min_index:x_max_index + 1]
        V_tmp = df[label_dict['V']['label']].values.reshape(self.grid_shape)[y_min_index:y_max_index + 1,
                x_min_index:x_max_index + 1]
        u_tmp = df[label_dict['u']['label']].values.reshape(self.grid_shape)[y_min_index:y_max_index + 1,
                x_min_index:x_max_index + 1]
        v_tmp = df[label_dict['v']['label']].values.reshape(self.grid_shape)[y_min_index:y_max_index + 1,
                x_min_index:x_max_index + 1]
        uv_tmp = df[label_dict['uv']['label']].values.reshape(self.grid_shape)[y_min_index:y_max_index + 1,
                 x_min_index:x_max_index + 1]
        uuu_tmp = df[label_dict['uuu']['label']].values.reshape(self.grid_shape)[y_min_index:y_max_index + 1,
                  x_min_index:x_max_index + 1]
        vvv_tmp = df[label_dict['vvv']['label']].values.reshape(self.grid_shape)[y_min_index:y_max_index + 1,
                  x_min_index:x_max_index + 1]
        uuv_tmp = df[label_dict['uuv']['label']].values.reshape(self.grid_shape)[y_min_index:y_max_index + 1,
                  x_min_index:x_max_index + 1]
        uvv_tmp = df[label_dict['uvv']['label']].values.reshape(self.grid_shape)[y_min_index:y_max_index + 1,
                  x_min_index:x_max_index + 1]
        n_tmp = df[label_dict['N']['label']].values.reshape(self.grid_shape)[y_min_index:y_max_index + 1,
                x_min_index:x_max_index + 1]

        # それぞれ空間平均する
        U_tmp = np.nansum(U_tmp * n_tmp, axis=1)
        V_tmp = np.nansum(V_tmp * n_tmp, axis=1)
        u_tmp = np.nansum(u_tmp ** 2 * n_tmp, axis=1)
        v_tmp = np.nansum(v_tmp ** 2 * n_tmp, axis=1)
        uv_tmp = np.nansum(uv_tmp * n_tmp, axis=1)
        uuu_tmp = np.nansum(uuu_tmp * n_tmp, axis=1)
        vvv_tmp = np.nansum(vvv_tmp * n_tmp, axis=1)
        uuv_tmp = np.nansum(uuv_tmp * n_tmp, axis=1)
        uvv_tmp = np.nansum(uvv_tmp * n_tmp, axis=1)
        N = np.sum(n_tmp, axis=1)

        np.seterr(all='ignore')
        U = U_tmp / N
        V = V_tmp / N
        u = np.sqrt(u_tmp / N)
        v = np.sqrt(v_tmp / N)
        uv = uv_tmp / N
        uuu = uuu_tmp / N
        vvv = vvv_tmp / N
        uuv = uuv_tmp / N
        uvv = uvv_tmp / N

        # 0 で割った部分の対処
        U[np.isnan(U)] = 0
        V[np.isnan(V)] = 0
        u[np.isnan(u)] = 0
        v[np.isnan(v)] = 0
        uv[np.isnan(uv)] = 0
        uuu[np.isnan(uuu)] = 0
        vvv[np.isnan(vvv)] = 0
        uuv[np.isnan(uuv)] = 0
        uvv[np.isnan(uvv)] = 0
        U[np.isinf(U)] = 0
        V[np.isinf(V)] = 0
        u[np.isinf(u)] = 0
        v[np.isinf(v)] = 0
        uv[np.isinf(uv)] = 0
        uuu[np.isinf(uuu)] = 0
        vvv[np.isinf(vvv)] = 0
        uuv[np.isinf(uuv)] = 0
        uvv[np.isinf(uvv)] = 0

        # サイズを標準化
        y_size = np.linspace(y[0], y[-1], size)
        fU = interpolate.interp1d(y, U)
        U = fU(y_size)
        fV = interpolate.interp1d(y, V)
        V = fV(y_size)
        fu = interpolate.interp1d(y, u)
        u = fu(y_size)
        fv = interpolate.interp1d(y, v)
        v = fv(y_size)
        fuv = interpolate.interp1d(y, uv)
        uv = fuv(y_size)
        fuuu = interpolate.interp1d(y, uuu)
        uuu = fuuu(y_size)
        fvvv = interpolate.interp1d(y, vvv)
        vvv = fvvv(y_size)
        fuuv = interpolate.interp1d(y, uuv)
        uuv = fuuv(y_size)
        fuvv = interpolate.interp1d(y, uvv)
        uvv = fuvv(y_size)
        fN = interpolate.interp1d(y, N)
        N = fN(y_size)

        self.space_averaged_data_frame = pd.DataFrame({
            'y': y_size,
            'U': U,
            'V': V,
            'u': u,
            'v': v,
            'uv': uv,
            'uuu': uuu,
            'vvv': vvv,
            'uuv': uuv,
            'uvv': uvv,
            'N': N
        })

    def space_averaging_from_array_2d(self, array_2d_dict, size):
        array_2d_dict = array_2d_dict

        y = array_2d_dict['y'][:,0]
        U_tmp = array_2d_dict['U']
        V_tmp = array_2d_dict['V']
        u_tmp = array_2d_dict['u']
        v_tmp = array_2d_dict['v']
        uv_tmp = array_2d_dict['uv']
        uuu_tmp = array_2d_dict['uuu']
        vvv_tmp = array_2d_dict['vvv']
        uuv_tmp = array_2d_dict['uuv']
        uvv_tmp = array_2d_dict['uvv']
        n_tmp = array_2d_dict['N']

        # それぞれ空間平均する
        U_tmp = np.nansum(U_tmp * n_tmp, axis=1)
        V_tmp = np.nansum(V_tmp * n_tmp, axis=1)
        u_tmp = np.nansum(u_tmp ** 2 * n_tmp, axis=1)
        v_tmp = np.nansum(v_tmp ** 2 * n_tmp, axis=1)
        uv_tmp = np.nansum(uv_tmp * n_tmp, axis=1)
        uuu_tmp = np.nansum(uuu_tmp * n_tmp, axis=1)
        vvv_tmp = np.nansum(vvv_tmp * n_tmp, axis=1)
        uuv_tmp = np.nansum(uuv_tmp * n_tmp, axis=1)
        uvv_tmp = np.nansum(uvv_tmp * n_tmp, axis=1)
        N = np.sum(n_tmp, axis=1)

        np.seterr(all='ignore')
        U = U_tmp / N
        V = V_tmp / N
        u = np.sqrt(u_tmp / N)
        v = np.sqrt(v_tmp / N)
        uv = uv_tmp / N
        uuu = uuu_tmp / N
        vvv = vvv_tmp / N
        uuv = uuv_tmp / N
        uvv = uvv_tmp / N

        # 0 で割った部分の対処
        U[np.isnan(U)] = 0
        V[np.isnan(V)] = 0
        u[np.isnan(u)] = 0
        v[np.isnan(v)] = 0
        uv[np.isnan(uv)] = 0
        uuu[np.isnan(uuu)] = 0
        vvv[np.isnan(vvv)] = 0
        uuv[np.isnan(uuv)] = 0
        uvv[np.isnan(uvv)] = 0
        U[np.isinf(U)] = 0
        V[np.isinf(V)] = 0
        u[np.isinf(u)] = 0
        v[np.isinf(v)] = 0
        uv[np.isinf(uv)] = 0
        uuu[np.isinf(uuu)] = 0
        vvv[np.isinf(vvv)] = 0
        uuv[np.isinf(uuv)] = 0
        uvv[np.isinf(uvv)] = 0

        # サイズを標準化
        y_size = np.linspace(y[0], y[-1], size)
        fU = interpolate.interp1d(y, U)
        U = fU(y_size)
        fV = interpolate.interp1d(y, V)
        V = fV(y_size)
        fu = interpolate.interp1d(y, u)
        u = fu(y_size)
        fv = interpolate.interp1d(y, v)
        v = fv(y_size)
        fuv = interpolate.interp1d(y, uv)
        uv = fuv(y_size)
        fuuu = interpolate.interp1d(y, uuu)
        uuu = fuuu(y_size)
        fvvv = interpolate.interp1d(y, vvv)
        vvv = fvvv(y_size)
        fuuv = interpolate.interp1d(y, uuv)
        uuv = fuuv(y_size)
        fuvv = interpolate.interp1d(y, uvv)
        uvv = fuvv(y_size)
        fN = interpolate.interp1d(y, N)
        N = fN(y_size)

        self.space_averaged_data_frame = pd.DataFrame({
            'y': y_size,
            'U': U,
            'V': V,
            'u': u,
            'v': v,
            'uv': uv,
            'uuu': uuu,
            'vvv': vvv,
            'uuv': uuv,
            'uvv': uvv,
            'N': N
        })

    @staticmethod
    def get_header_row(file):
        """データのヘッダ行数を取得する"""
        file = open(file, 'r')
        for i, line in enumerate(file):
            if line[0] == 'y':
                file.close()
                return i
        file.close()

    def join(self, space_averaged_data_frame_list):
        """複数実験セットのデータを統合する"""
        np.seterr(all='ignore')
        if self.space_averaged_data_frame is None:  # 空のオブジェクトの場合
            if type(space_averaged_data_frame_list) is list:  # 複数 set のデータ
                self.space_averaged_data_frame = space_averaged_data_frame_list[0].copy()
                for i, space_averaged_data_frame in enumerate(space_averaged_data_frame_list):
                    if i == 0: continue
                    df1 = self.space_averaged_data_frame.copy()
                    N1 = df1['N']
                    df2 = space_averaged_data_frame.copy()
                    N2 = df2['N']
                    for text in ['U', 'V', 'uuu', 'vvv', 'uuv', 'uvv', 'uv']:
                        a1 = df1[text]
                        a2 = df2[text]
                        data = (a1 * N1 + a2 * N2) / (N1 + N2)
                        data[np.isnan(data)] = 0
                        data[np.isinf(data)] = 0
                        self.space_averaged_data_frame[text] = data.copy()
                    for text in ['u', 'v']:
                        a1 = df1[text]
                        a2 = df2[text]
                        data = np.sqrt((a1 ** 2 * N1 + a2 ** 2 * N2) / (N1 + N2))
                        data[np.isnan(data)] = 0
                        data[np.isinf(data)] = 0
                        self.space_averaged_data_frame[text] = data.copy()
                self.space_averaged_data_frame['N'] = N1 + N2
            else:  # 1 set のデータのみ
                space_averaged_data_frame = space_averaged_data_frame_list.copy()
                self.space_averaged_data_frame = space_averaged_data_frame.copy()

        else:
            if type(space_averaged_data_frame_list) is list:  # 複数 set のデータ
                for space_averaged_data_frame in space_averaged_data_frame_list:
                    df1 = self.space_averaged_data_frame.copy()
                    N1 = df1['N']
                    df2 = space_averaged_data_frame.copy()
                    N2 = df2['N']
                    for text in ['U', 'V', 'uuu', 'vvv', 'uuv', 'uvv', 'uv']:
                        a1 = df1[text]
                        a2 = df2[text]
                        data = (a1 * N1 + a2 * N2) / (N1 + N2)
                        data[np.isnan(data)] = 0
                        data[np.isinf(data)] = 0
                        self.space_averaged_data_frame[text] = data.copy()
                    for text in ['u', 'v']:
                        a1 = df1[text]
                        a2 = df2[text]
                        data = np.sqrt((a1 ** 2 * N1 + a2 ** 2 * N2) / (N1 + N2))
                        data[np.isnan(data)] = 0
                        data[np.isinf(data)] = 0
                        self.space_averaged_data_frame[text] = data.copy()
                self.space_averaged_data_frame['N'] = N1 + N2
            else:  # 1 set のデータのみ
                space_averaged_data_frame = space_averaged_data_frame_list.copy()
                df1 = self.space_averaged_data_frame.copy()
                N1 = df1['N']
                df2 = space_averaged_data_frame.copy()
                N2 = df2['N']
                for text in ['U', 'V', 'uuu', 'vvv', 'uuv', 'uvv', 'uv']:
                    a1 = df1[text]
                    a2 = df2[text]
                    data = (a1 * N1 + a2 * N2) / (N1 + N2)
                    data[np.isnan(data)] = 0
                    data[np.isinf(data)] = 0
                    self.space_averaged_data_frame[text] = data.copy()
                for text in ['u', 'v']:
                    a1 = df1[text]
                    a2 = df2[text]
                    data = np.sqrt((a1 ** 2 * N1 + a2 ** 2 * N2) / (N1 + N2))
                    data[np.isnan(data)] = 0
                    data[np.isinf(data)] = 0
                    self.space_averaged_data_frame[text] = data.copy()
                self.space_averaged_data_frame['N'] = N1 + N2


class InstantData:
    """瞬時データ"""

    def __init__(self, file):
        self.file = file
        header_row = self.get_header_row(file)  # ヘッダ行数を取得
        self.df = pd.read_csv(self.file, header=header_row)  # ファイルからデータを読み出し

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


def get_crop_index(time_averaged_data_frame, grid_shape=[74, 101], crop_range=['', '', '', '']):
    """クロップ後のデータのインデックスを取得"""
    df = time_averaged_data_frame
    x = df[label_dict['x']['label']].values.reshape(grid_shape)[0, :]
    y = df[label_dict['y']['label']].values.reshape(grid_shape)[:, 0]

    # range に対応するデータフレームの範囲を探索
    x_min_mm, x_max_mm, y_min_mm, y_max_mm = crop_range
    x_min_index = 0
    x_max_index = len(x) - 1
    y_min_index = 0
    y_max_index = len(y) - 1
    for i in range(len(x) - 1):
        if x[i] < x_min_mm and x_min_mm <= x[i + 1]:
            x_min_index = i + 1
        elif x[i] <= x_max_mm and x_max_mm < x[i + 1]:
            x_max_index = i
    for j in range(len(y) - 1):
        if y[j] < y_min_mm and y_min_mm <= y[j + 1]:
            y_min_index = j + 1
        elif y[j] <= y_max_mm and y_max_mm < y[j + 1]:
            y_max_index = j

    return x_min_index, x_max_index, y_min_index, y_max_index

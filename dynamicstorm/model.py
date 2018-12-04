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

    def filter(self, filter_value):
        self.file_list = dyfil.Filter().filter_incorrect_vector(self.file_list, filter_value)

    def time_averaging(self):
        self.statistics.time_averaging(self.file_list)

    def join(self, expt_instance_list):
        """
        複数の実験データを結合してまとめて取り扱えるようにする
        全てに時間平均，空間平均済みのデータがある場合は結合後のデータで上書きする
        一つでも平均済みのデータがない物があった場合は平均データを空にする
        """
        for expt_instance in expt_instance_list:
            self.file_list.append(expt_instance.filelist)
        # TODO: 平均済みのデータがあるものとないものを結合する場合の考慮 → 削除でよさげ
class Statistics:
    """時間平均データ"""

    def __init__(self, instant_data_list=None, source_dir=None):
        self.time_averaged_data_frame = ''
        if instant_data_list is not None:
            self.time_averaging(instant_data_list)
        elif source_dir is not None:
            self.read(source_dir)

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
        uu = np.sqrt(uu - U ** 2)
        vv = np.sqrt(vv - V ** 2)
        uv = uv - U * V
        uuu = uuu - 3 * U * cuu + 2 * U ** 3
        vvv = vvv - 3 * U * cvv + 2 * V ** 3
        uuv = uuv - V * cuu - 2 * U * cuv + 2 * U ** 2 * V
        uvv = uvv - U * cvv - 2 * V * cuv + 2 * U * V ** 2

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

    def join(self, statistics):
        pass

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
    text = 'time averaging task ' + str(current_core + 1) + '/' + str(total_core)

    # 全て 0 の配列を用意
    df = pd.read_csv(file_list[0], header=header)
    n = ((df['Status'] == 0) * 1).values
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
        U_tmp = df['U[m/s]'].values
        V_tmp = df['V[m/s]'].values
        n = ((df['Status'] == 0) * 1).values
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
    return U, V, uu, vv, uv, uuu, vvv, uuv, uvv, N




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

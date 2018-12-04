# coding: utf-8

# モジュールインポート
import dynamicstorm as ds
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np


def main():
    # 瞬時データの集まりのオブジェクトを作成
    # 瞬時データの csv ファイルが複数入ったディレクトリを指定
    expt = ds.model.ExptSet('expt_instant_data_sample')

    # クロップ範囲を設定（マスキングがかかっていない範囲[mm]）
    # [x_min, x_max, y_min, y_max], クロップしない値は '' で埋める
    crop_range = [14.50, 68.29, 2.08, 51.85]

    # 誤ベクトル数の例を表示（フィルタリング前）
    expt.incorrect_vector_example(100)

    # 指定した数以上の誤ベクトルを持つデータを除外
    # （multiprocessing を利用しているので，進捗状況は外部ターミナルで表示）
    expt.incorrect_vector_filter(3800)

    # 誤ベクトル数の例を表示（フィルタリング後）
    expt.incorrect_vector_example(100)

    # 時間平均
    # （multiprocessing を利用しているので，進捗状況は外部ターミナルで表示）
    stat = ds.model.Statistics(expt.instant_data_list)

    # 時間平均データフレーム
    stat.time_averaged_data_frame

    # 時間平均されたデータを csv 形式で保存
    # stat.save('time_averaged_data.csv')

    # 表示するデータとデータラベルの対応を表示（datalabel.py）
    print('value:', 'label')
    print('---------------')
    for valiable in ds.label_dict:
        print('{}:'.format(valiable), ds.label_dict[valiable]['label'])

    # クロップ範囲を取得
    x_min_index, x_max_index, y_min_index, y_max_index = ds.get_crop_index(
        stat.time_averaged_data_frame, grid_shape=[74, 101], crop_range=crop_range)

    # 原点を合わせる
    xlabel = ds.label_dict['x']['label']
    ylabel = ds.label_dict['y']['label']
    x = stat.time_averaged_data_frame[xlabel] - \
        stat.time_averaged_data_frame['x (mm)[mm]'][x_min_index]
    y = stat.time_averaged_data_frame[ylabel] - \
        stat.time_averaged_data_frame['y (mm)[mm]'][y_min_index]

    x = x.values.reshape(74, 101)[0, :]
    y = y.values.reshape(74, 101)[:, 0]

    # コンタープロット用に x 軸，y 軸のグリッドを作成
    X, Y = np.meshgrid(x, y)

    # 主流方向が ← なので，左右反転，主流方向速度の符号を反転させる
    C = np.fliplr(
        -stat.time_averaged_data_frame['U[m/s]'].values.reshape(74, 101))
    X = np.fliplr(X)

    # 無次元化
    C = C / np.max(C)
    X = X / 50
    Y = Y / 50

    # コンター図をプロット
    plt.title('title')
    plt.pcolor(X, Y, C, cmap='jet')
    bar = plt.colorbar()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.clim([-1, 1])
    plt.axhline(0.5, color='gray')  # y = 0.5 に補助線
    plt.xlabel('$x/H$')
    plt.ylabel('$y/H$')
    bar.set_label('$U/U_{max}$')
    plt.show()

    # 空間平均

    # 時間平均データのデータフレームから空間平均データを生成
    space = ds.model.SpaceAverage(data_frame=stat.time_averaged_data_frame, grid_shape=[
        74, 101], crop_range=crop_range)
    # 保存済みの csv から空間平均データを生成
    # space = ds.model.SpaceAverage(source_file='space.csv',grid_shape=[74,101], crop_range=crop_range)

    # 空間平均データフレーム
    space.space_averaged_data_frame

    # 空間平均されたデータを csv 形式で保存
    # space.save('space.csv')

    # y軸，data軸の値を設定
    y = space.space_averaged_data_frame['y']
    data = -space.space_averaged_data_frame['U']  # 主流方向が ← なので符号を反転

    # 無次元化
    y = y / 50
    data = data / np.max(data)

    # 散布図をプロット
    plt.title('title')
    plt.scatter(data, y)
    plt.xlim(0, 3)
    plt.ylim(0, 1)
    plt.xlabel('$\overline{U}/U_{max}$')
    plt.ylabel('$y/H$')
    plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(0.1))
    plt.grid(which='nimor')
    plt.axhline(0.5, color='gray')  # y = 0.5 に補助線
    plt.show()


if __name__ == '__main__':
    main()

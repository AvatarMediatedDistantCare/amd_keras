# amd_keras

### 1. データセット
- data/csv/kinect/xxx.csv: 100次元(25関節x4次元数) x timestep
- data/csv/mocap/xxx.csv: 100次元(25関節x4次元数) x timestep
- data/csv/kinect_noised/xxx.csv: 100次元(25関節x4次元数) x timestep （ki_xxx に対してノイズを加えたもの）
- data/data_path.csv: データセットを作成するデータを指定するcsvファイル(必要に応じて編集)

- data/npy/: データセット保存先

### 3. データセットの作成
```sh
cd data/
python create_dataset.py 20 #  20: num of context（当該フレームを含めて前後10frameずつみてる）
```

TODO: (timesteps, 100) のデータを (timesteps, 20, 100) にしているようだが、本来は (timesteps, 21, 100) にするべき 

### 4. モデルの作成
```sh
cd amd_keras
python train.py model.hdf5 20
```

### 5. フィルタのかけ方
1. `node movingAverage_oneEuro.js (input file name) (window size)` ... 推定エラー補正後のモーションの移動平均フィルタとoneEuroフィルタを適用するために使用

『この段階の成果物』
1. (input file nameと同じディレクトリ) / OOO_apply_normalize.csv ... ノーマライズされたモーションデータ
1. (input file nameと同じディレクトリ) / OOO_apply_one_euro_filter.csv ... one_euroフィルタが適用されたモーションデータ
1. (input file nameと同じディレクトリ) / OOO_apply_moving_average_(window size).csv ... (window size)分の移動平均フィルタが適用されたモーションデータ

### 6. 描画プログラム
1. input_predicted_correct.js ... input, predicted, correctを比較するコード
(比較したいもののためにコード内の `filename` 適宜変えてください)

### 7. 定量的評価プログラムの使い方
1. 以下のコードのfilename[0]~[2]の箇所を任意の値に変更
  - https://github.com/banaoh/amd_kinect_quaternion/blob/master/js/quantitative_evaluation.js#L153-L157
1. 以下のコメントアウトを消して、 `js/quantitative_evaluation.js` を読み込ませる
  - https://github.com/banaoh/amd_kinect_quaternion/blob/master/index.html#L23

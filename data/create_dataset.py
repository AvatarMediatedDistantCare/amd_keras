import sys
import os
import pandas as pd
import numpy as np
import math

N_INPUT = 100 # Number of MFCC features
N_OUTPUT = 100 # Number of gesture features
DATA_FILE = ''
N_CONTEXT = 20

def create_vectors(kinect_filename, mocap_filename):
    # Step 1:
    # kinectの生データ読み込みー配列化, org[frame][quaternion = 100（25関節 x quaternion）]
    # mocapの生データを読み込み -> 配列化, org[frame][quaternion = 100（25関節 x quaternion）]

    print(kinect_filename)
    f = open(kinect_filename, 'r')
    kinect_data = f.readlines()
    for idx, line in enumerate(kinect_data):
        line = line.rstrip()
        kinect_data[idx]= [float(x) if(x != '' and math.isnan(float(x)) != True) else 0.0 for x in line.split(',')]
    input_vectors = np.array(kinect_data)

    print(mocap_filename)
    f = open(mocap_filename, 'r')
    mocap_data = f.readlines()

    # mocap[frame][quaternioni]の形にする.空白が入っている場合は0.0を代入
    for idx, line in enumerate(mocap_data):
        line = line.rstrip()
        mocap_data[idx] = [float(x) if(x != '' and math.isnan(float(x)) != True) else 0.0  for x in line.split(',')]
    output_vectors = np.array(mocap_data)

    # Step 2: 短い方にあわせる
    diff = len(input_vectors) - len(output_vectors)
    if diff > 0:
        input_vectors = input_vectors[0:len(input_vectors) - diff]
    elif diff < 0:
        output_vectors = output_vectors[0:len(output_vectors) + diff]

    print('前後N_contextを見る前のshape')
    print(input_vectors.shape)
    print(output_vectors.shape)


    # Step 3: N_CONTEXTずつとりだして格納して、１ずつSRIDEしていく
    input_with_context = np.array([])
    output_with_context = np.array([])

    progress = 0
    for i in range( len(input_vectors) - N_CONTEXT ):
        if (i/len(input_vectors - N_CONTEXT))*100 > progress:
            print("progress: " + str(progress) + "%")
            progress += 1
        if i == 0:
            input_with_context = input_vectors[i:i + N_CONTEXT].reshape(1, N_CONTEXT, N_INPUT)
            output_with_context = output_vectors[i + int(N_CONTEXT / 2)].reshape(1, N_OUTPUT)
        else:
            input_with_context = np.append(input_with_context,
                                           input_vectors[i:i + N_CONTEXT].reshape(1, N_CONTEXT, N_INPUT), axis=0)
            output_with_context = np.append(output_with_context, output_vectors[i + N_CONTEXT].reshape(1, N_OUTPUT), axis=0)

    print('前後N_contextを見たあとのshape')
    print(input_with_context.shape)
    print(output_with_context.shape)

    return input_with_context, output_with_context
    # return input_vectors, output_vectors

# name = ('train' or 'dev') and 'test' 
# kinectとmocapのそれぞれ対応するデータを元にnpyを作成
def create(name):
    DATA_FILE = pd.read_csv('data_path.csv')
    X = np.array([])
    Y = np.array([])

    for i in range(len(DATA_FILE)):
        input_vectors, output_vectors = create_vectors(DATA_FILE['kinect_filename'][i], DATA_FILE['mocap_filename'][i])
        if len(X) == 0:
            X = input_vectors
            Y = output_vectors
        else:
            X = np.concatenate((X, input_vectors), axis=0)
            Y = np.concatenate((Y, output_vectors), axis=0)

        if i == 38: # i:0-38 まで通常のデータセット、以降はオーグメンテーションデータ
            # テストデータを取り除いておく（ランダム抽出で、9:1に分ける）
            num_train = int(len(X) * 0.9)
            num_test = len(X) - num_train
            num_all = num_train + num_test
            id_all = np.random.choice(num_all, num_all, replace=False)
            id_test = id_all[0:num_test]
            id_train = id_all[num_test:num_all]
            X_train = X[id_train]
            Y_train = Y[id_train]
            X_test = X[id_test]
            Y_test = Y[id_test]
            X = X_train
            Y = Y_train

            np.save("npy/X_test.npy", X_test)
            np.save("npy/Y_test.npy", Y_test)

    np.save("npy/X_train.npy", X)
    np.save("npy/Y_train.npy", Y)

if __name__ == "__main__":
    N_CONTEXT = int(sys.argv[1])
    create('dev')
    # create('test')

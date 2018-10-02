import sys
import os
import pandas as pd
import numpy as np
import math

N_INPUT = 100 # Number of MFCC features
N_OUTPUT = 100 # Number of gesture features
DATA_FILE = ''
N_CONTEXT = 10

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
            print("progress: " + progress + "%")
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


    x_file_name = 'npy/X.npy'
    y_file_name = 'npy/Y.npy'
    np.save(x_file_name, X)
    np.save(y_file_name, Y)

if __name__ == "__main__":
    N_CONTEXT = int(sys.argv[1])
    create('dev')
    # create('test')

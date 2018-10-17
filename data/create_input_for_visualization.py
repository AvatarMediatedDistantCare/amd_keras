import sys
import pandas as pd
import numpy as np
import math

N_INPUT = 100 # Number of MFCC features
N_OUTPUT = 100 # Number of gesture features
DATA_FILE = ''
N_CONTEXT = 20

def create_vectors(kinect_filename):
    # Step 1:
    # kinectの生データ読み込みー配列化, org[frame][quaternion = 100（25関節 x quaternion）]

    print(kinect_filename)
    f = open(kinect_filename, 'r')
    kinect_data = f.readlines()
    for idx, line in enumerate(kinect_data):
        line = kinect_data[idx].rstrip()
        kinect_data[idx]= [float(x) if(x != '' and math.isnan(float(x)) != True) else 0.0 for x in line.split(',')]
    input_vectors = np.array(kinect_data)

    print('前後N_contextを見る前のshape')
    print(input_vectors.shape)

    # Step 3: N_CONTEXTずつとりだして格納して、１ずつSRIDEしていく
    input_with_context = np.array([])

    progress = 0
    for i in range( len(input_vectors) - N_CONTEXT ):
        if (i/len(input_vectors - N_CONTEXT))*100 > progress:
            print("progress: " + str(progress) + "%")
            progress += 1
        if i == 0:
            input_with_context = input_vectors[i:i + N_CONTEXT].reshape(1, N_CONTEXT, N_INPUT)
        else:
            input_with_context = np.append(input_with_context,
                                           input_vectors[i:i + N_CONTEXT].reshape(1, N_CONTEXT, N_INPUT), axis=0)

    print('前後N_contextを見たあとのshape')
    print(input_with_context.shape)

    return input_with_context


def create(kinect_filename, npy_filename):
    X = np.array([])

    input_vectors = create_vectors(kinect_filename)
    np.save(npy_filename, X)

if __name__ == "__main__":
    create(sys.argv[1], sys.argv[2])


import numpy as np

def create_csv(filename, prediction):

    with open(filename, 'w') as fo:
        prediction = np.squeeze(prediction)
        print("output vector shape: " + str(prediction.shape))

        print(prediction)
        np.savetxt(filename, prediction, delimiter=',')
        # for row in prediction:
        #     row = np.delete(row, np.s_[0:3])
        #     row[0:3] = 0
        #     fo.write(label_line + '\n')
        print("csv generated")



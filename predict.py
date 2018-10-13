import sys
from keras.models import load_model
from helper import create_csv
import numpy as np

def predict(model_name, input_file, output_file):
    model = load_model(model_name)
    X = np.load(input_file)

    predicted = np.array(model.predict(X))
    print(predicted.shape)
    create_csv(output_file, predicted)

if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2], sys.argv[3])

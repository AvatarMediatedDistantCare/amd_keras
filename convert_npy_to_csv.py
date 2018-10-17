import sys
from helper import create_csv
import numpy as np

def convert(npy_file, csv_file):
    npy = np.load(npy_file)
    create_csv(csv_file, npy)

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])

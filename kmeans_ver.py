import os
import math
import pandas as pd
import numpy as np

type_map = {
    'time': 0,
    'environment': 1,
    'activity': 2,
    'character': 3,
}

class kmeans_assign:
    quo_time = 25
    quo_env = 25
    quo_act = 25
    quo_char = 25

    def __init__(self, file_path: str):
        self.data = pd.read_excel(file_path)
        self.data = self.data.dropna()
        self.data = self.data.drop_duplicates()
        self.data = self.data.to_numpy()
        print(self.data)

    def calcu_diff(self, data1: np.ndarray, data2: np.ndarray, type) -> float:
        diff = 0
        if type == 'time':
            diff_time = 0
            for i in range(1, 4):
                diff_time += (abs(data1[i] - data2[i]))**2
            diff += sqrt(diff_time)
        elif type == 'environment':
        elif type == 'activity':
        elif type == 'character':
        else:
            return -1
        


if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), "kmeans_data.xlsx")
    kmeans_assign(file_path)

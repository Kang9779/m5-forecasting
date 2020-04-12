import pandas as pd
import numpy as np



def my_objective_func(preds,labels):
    pass

def get_weight(series):
    '''得到每个sample 的weight
    '''
    series = series.values
    index = 0
    for index in series:
        if series[index]>0:
            break
    series = pd.Series(series[index:])
    shift_s = series.shift(1)
    diff = (series - shift_s)**2
    weight = 1 / (np.sum(diff) + 1.0)
    return weight

if __name__ == '__main__':
    series = pd.Series([0,0,0,0,1,2,5,6,3,10])
    print(get_weight(series))


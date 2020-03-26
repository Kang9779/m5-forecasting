import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

def tmpFunc(df,key=None):
    df['c_shift'] = df['c1'].transform(lambda x:x.shift(1))
    print(key)
    return df

def applyParallel(dfGrouped, func,key):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group,key) for name, group in dfGrouped)
    return pd.concat(retLst)

if __name__ == '__main__':
    df = pd.DataFrame(columns=['id','c1','c2'],
        data = [
            ['id1',2,10],
            ['id1',3,0],
            ['id1',2,10],
            ['id1',3,0],
            ['id2',2,9],
            ['id2',5,20],
            ['id2',4,9],
            ['id2',8,20]
        ]
    )
    print ('parallel version: ')
    print (applyParallel(df.groupby(df.id), tmpFunc,"groupbyId"))
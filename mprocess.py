import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool, cpu_count

def tmpFunc(df,key=None):
    df['c_shift'] = df['c1'].transform(lambda x:x.shift(1))
    print(key)
    return df
##方法1
# def applyParallel(dfGrouped, func,key=None):
#     retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group,key) for name, group in dfGrouped)
#     return pd.concat(retLst)

##方法2
def applyParallel(dfGrouped, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)

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
    print (applyParallel(df.groupby(df.id), tmpFunc))
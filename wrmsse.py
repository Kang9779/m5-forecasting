import numpy as np
import pandas as pd
import os
import gc
from scipy.sparse import csr_matrix
from sklearn import preprocessing, metrics
import dask.dataframe as dd

sales_train_val = pd.read_csv('./data/sales_train_validation.csv')
product = sales_train_val[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()

NUM_ITEMS = sales_train_val.shape[0]  # 30490

weight_mat = np.c_[np.identity(NUM_ITEMS).astype(np.int8), #item :level 12
                   np.ones([NUM_ITEMS,1]).astype(np.int8), # level 1
                   pd.get_dummies(product.state_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.store_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.cat_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.dept_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.state_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.state_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.store_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.store_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.item_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.state_id.astype(str) + product.item_id.astype(str),drop_first=False).astype('int8').values
                   ].T

weight_mat_csr = csr_matrix(weight_mat)
del weight_mat; 
gc.collect()

def weight_calc(data,product):

    # calculate the denominator of RMSSE, and calculate the weight base on sales amount
    sales_train_val = pd.read_csv('./data/sales_train_validation.csv')
    d_name = ['d_' + str(i+1) for i in range(1913)]
    sales_train_val = weight_mat_csr * sales_train_val[d_name].values
    
    # calculate the start position(first non-zero demand observed date) for each item
    df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))
    
    start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1
    
    # denominator of RMSSE
    weight1 = np.sum((np.diff(sales_train_val,axis=1)**2),axis=1)/(1913-start_no)
    
    # calculate the sales amount for each item/level
    df_tmp = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
    df_tmp['amount'] = df_tmp['demand'] * df_tmp['sell_price']
    df_tmp = df_tmp.groupby(['id'])['amount'].apply(np.sum).values
    
    weight2 = weight_mat_csr * df_tmp 

    weight2 = weight2/np.sum(weight2)
    
    del sales_train_val
    gc.collect()
    
    return weight1, weight2


weight1, weight2 = weight_calc(df_train,product)

def wrmsse(preds, data):
    # actual obserbed values
    y_true = data.get_label().values
    # number of columns
    num_col = len(y_true)//NUM_ITEMS
    
    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) )
    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
    
    x_name = ['pred_' + str(i) for i in range(num_col)]
    x_name2 = ["act_" + str(i) for i in range(num_col)]
          
    train = np.array(weight_mat_csr*np.c_[reshaped_preds, reshaped_true])
    
    score = np.sum(np.sqrt(np.mean(np.square(
                            train[:,:num_col] - train[:,num_col:])
                        ,axis=1) / weight1) * weight2)
    
    return 'wrmsse', score, False

def wrmsse_simple(preds, data):
    # actual obserbed values
    y_true = data.get_label().values
    # number of columns
    num_col = len(y_true)//NUM_ITEMS
    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) 
    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
          
    train = np.c_[reshaped_preds, reshaped_true]
    
    weight2_2 = weight2[:NUM_ITEMS]
    weight2_2 = weight2_2/np.sum(weight2_2)
    
    score = np.sum(np.sqrt(np.mean(np.square(
                            train[:,:num_col] - train[:,num_col:])
                        ,axis=1) /  weight1[:NUM_ITEMS])*weight2_2)
    
    return 'wrmsse', score, False


if __name__ == '__main__':

    pass



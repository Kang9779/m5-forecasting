
import pandas as pd
import numpy as np


# add releae features
# df_data = pd.DataFrame()
# df_data['release'] = df_data.groupby(['store_id','item_id'])['wm_yr_wk'].transform('min')
# df_data['release'] = df_data['release'] - df_data['release'].min()
# df_data['item_nunique'] = df_data.groupby(['store_id','sell_price'])['item_id'].transform('nunique')
# df_data['sell_price_nunique'] = df_data.groupby(['store_id','item_id'])['sell_price'].transform('nunique')


# # add price_sell features
# df_data['price_momentum'] = df_data['sell_price']/df_data.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))
# df_data['price_momentum_month'] = df_data['sell_price']/df_data.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
# df_data['price_momentum_year'] = df_data['sell_price']/df_data.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')


# def get_train_val_sets(df_data=None,folds=3):
#     '''train set and validation set split generator
#     '''
#     if df_data is None:
#         raise ValueError("df_data should not be NoneType")
#     df_tr = []
#     df_val = []
#     for i in range(0,folds):
#         t_split = 1914-(i+1)*28
#         print(t_split, 1914-i*28)
#         df_tr = df_data[df_data['d']<t_split]
#         df_val = df_data[(df_data['d'] >= t_split)&(df_data[df_data['d'] < 1914-i*28])]
#         yield i,df_tr,df_val

# def train_model(model, df_data, df_test, tr_features, folds=3,label='demand'):
#     '''train model
#     params:
#         model:lgb
#         df_tr:[]
#         df_val:[]
#         df_test:pd.DataFrame
#     return:
#         test_pred: test sets prediction
#         model:model
#     '''
#     test_pred = np.zeros((df_test.shape[0],folds))
#     data_iterator = get_train_val_sets(df_data,folds)
#     score = 0
#     for i, x_tr, x_val in data_iterator:
#         '''
#             model training
#         '''
#         pass

#     return test_pred, score,model


# if __name__ == '__main__':
#     dt = get_train_val_sets()
#     for i,x_tr,x_val in dt:
#         print(i,x_tr,x_val)
#     # get_train_val_sets()
#     # test_pred, model = train_model(model, df_tr, df_val, df_test, tr_features)
#     # print(test_pred)


##add fe1
# df_data = pd.DataFrame()

# for f in ['store_id','state_id','dept_id','cat_id']:
#     df_data['%s_sell_price_ratio'%f] = df_data.groupby([f])['sell_price'].transform('sum')
#     df_data['%s_sell_price_ratio'%f] = df_data['sell_price']/df_data['%s_sell_price_ratio'%f]


# ###add fe2
# df_tr = df_data[df_data['d']<=1913]
# df_temp = df_data[['id','state_id','store_id','item_id','cat_id','demand','sell_price']]
# df_temp['sales'] = df_temp['demand'] * df_temp['sell_price'] + 1
# df_temp.drop(columns=['demand','sell_price'],inplace=True)
# for f in ['store_id','state_id','item_id','cat_id']:
#     df_temp['%s_sales_ratio'%f] = df_temp.groupby(f)['sales'].transform('sum')
#     df_temp['%s_sales_ratio'%f] = df_temp['sales']/df_temp['%s_sales_ratio'%f]
#     df_data = feature_mean(df_data,df_temp,fe=['id'],value='%s_sales_ratio'%f)
#     df_data = feature_median(df_data,df_temp,fe=['id'],value='%s_sales_ratio'%f)

df_sub = pd.read_csv('./sub/df_submission_0.51496.csv')
for i in range(28):
    df_sub['F%s'%(i+1)] *= 1.02

df_sub.sort_values(by=['id'],inplace=True)
df_sub.to_csv('./sub/df_submission.csv',index=None)



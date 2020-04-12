import pandas as pd
import numpy as np


sub_lgb = pd.read_csv('./sub/df_submission_0.54431.csv')
sub_cat = pd.read_csv('./sub/df_submission_0.54136.csv')

sub_lgb.sort_values(by=['id'],inplace=True)
sub_cat.sort_values(by=['id'],inplace=True)

lgb_score = 1.0
cat_score = 1.0
_sum = lgb_score + cat_score
w1 = (_sum - lgb_score) / _sum
w2 = (_sum - cat_score) / _sum

for i in range(28):
    sub_lgb['F%s'%(i+1)] = sub_lgb['F%s'%(i+1)] * w1 + sub_cat['F%s'%(i+1)] * w2

sub_lgb.to_csv('./sub/df_sub.csv',index=None)

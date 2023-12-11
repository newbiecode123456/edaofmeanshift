import numpy as np
import pandas as pd
import random
np.random.seed(99)
random.seed(42)
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import logging
from IPython.display import display
import pdb
from functools import reduce
import copy
import time
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
import copy
def EDA_find_remove_columns_with_infinity(df, remove=False):
    """
    This function finds all columns in a dataframe that have inifinite values (np.inf or -np.inf)
    It returns a list of column names. If the list is empty, it means no columns were found.
    If remove flag is set, then it returns a smaller dataframe with inf columns removed.
    """
    nums = df.select_dtypes(include='number').columns.tolist()
    dfx = df[nums]
    sum_rows = np.isinf(dfx).values.sum()
    add_cols =  list(dfx.columns.to_series()[np.isinf(dfx).any()])
    if sum_rows > 0:
        print('    there are %d rows and %d columns with infinity in them...' %(sum_rows,len(add_cols)))
        if remove:
            nocols = [x for x in df.columns if x not in add_cols]
            print("    Shape of dataset before %s and after %s removing columns with infinity" %(df.shape,(df[nocols].shape,)))
            return df[nocols]
        else:
            return add_cols
    else:
        return add_cols
def classify_columns(df_preds, verbose=0):
    train = copy.deepcopy(df_preds)
    max_nlp_char_size = 30
    max_cols_to_print = 30
    print('Classifying variables in data set...')
    cat_limit = 35
    float_limit = 15
    def add(a,b):
        return a+b
    sum_all_cols = dict()
    orig_cols_total = train.shape[1]
    #Types of columns
    cols_delete = []
    cols_delete = [col for col in list(train) if (len(train[col].value_counts()) == 1
                                       ) | (train[col].isnull().sum()/len(train) >= 0.90)]
    inf_cols = EDA_find_remove_columns_with_infinity(train)
    mixed_cols = [x for x in list(train) if len(train[x].dropna().apply(type).value_counts()) > 1]
    if len(mixed_cols) > 0:
        print('    Removing %s column(s) due to mixed data type detected...' %mixed_cols)
    cols_delete += mixed_cols
    cols_delete += inf_cols
    train = train[left_subtract(list(train),cols_delete)]
    var_df = pd.Series(dict(train.dtypes)).reset_index(drop=False).rename(
                        columns={0:'type_of_column'})
    sum_all_cols['cols_delete'] = cols_delete

    var_df['bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['bool','object']
                        and len(train[x['index']].value_counts()) == 2 else 0, axis=1)
    string_bool_vars = list(var_df[(var_df['bool'] ==1)]['index'])
    sum_all_cols['string_bool_vars'] = string_bool_vars
    var_df['num_bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                            np.uint16, np.uint32, np.uint64,
                            'int8','int16','int32','int64',
                            'float16','float32','float64'] and len(
                        train[x['index']].value_counts()) == 2 else 0, axis=1)
    num_bool_vars = list(var_df[(var_df['num_bool'] ==1)]['index'])
    sum_all_cols['num_bool_vars'] = num_bool_vars
    discrete_or_nlp = var_df.apply(lambda x: 1 if x['type_of_column'] in ['object']  and x[
        'index'] not in string_bool_vars+cols_delete else 0,axis=1)
    var_df['nlp_strings'] = 0
    var_df['discrete_strings'] = 0
    var_df['cat'] = 0
    var_df['id_col'] = 0
    discrete_or_nlp_vars = var_df.loc[discrete_or_nlp==1]['index'].values.tolist()
    copy_discrete_or_nlp_vars = copy.deepcopy(discrete_or_nlp_vars)
    if len(discrete_or_nlp_vars) > 0:
        for col in copy_discrete_or_nlp_vars:
            train[[col]] = train[[col]].fillna('  ')
            if train[col].map(lambda x: len(x) if type(x)==str else 0).max(
                ) >= 50 and len(train[col].value_counts()
                        ) >= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'nlp_strings'] = 1
            elif train[col].map(lambda x: len(x) if type(x)==str else 0).mean(
                ) >= max_nlp_char_size and train[col].map(lambda x: len(x) if type(x)==str else 0).max(
                ) < 50 and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'discrete_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'discrete_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) == len(train) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                var_df.loc[var_df['index']==col,'cat'] = 1
    nlp_vars = list(var_df[(var_df['nlp_strings'] ==1)]['index'])
    sum_all_cols['nlp_vars'] = nlp_vars
    discrete_string_vars = list(var_df[(var_df['discrete_strings'] ==1) ]['index'])
    sum_all_cols['discrete_string_vars'] = discrete_string_vars
    var_df['dcat'] = var_df.apply(lambda x: 1 if str(x['type_of_column'])=='category' else 0,
                            axis=1)
    factor_vars = list(var_df[(var_df['dcat'] ==1)]['index'])
    sum_all_cols['factor_vars'] = factor_vars
    date_or_id = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                         np.uint16, np.uint32, np.uint64,
                         'int8','int16',
                        'int32','int64']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    var_df['int'] = 0
    var_df['date_time'] = 0
    var_df['date_time'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['<M8[ns]','datetime64[ns]']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    if len(var_df.loc[date_or_id==1]) != 0:
        for col in var_df.loc[date_or_id==1]['index'].values.tolist():
            if len(train[col].value_counts()) == len(train):
                if train[col].min() < 1900 or train[col].max() > 2050:
                    var_df.loc[var_df['index']==col,'id_col'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                if train[col].min() < 1900 or train[col].max() > 2050:
                    if col not in num_bool_vars:
                        var_df.loc[var_df['index']==col,'int'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        if col not in num_bool_vars:
                            var_df.loc[var_df['index']==col,'int'] = 1
    else:
        pass
    int_vars = list(var_df[(var_df['int'] ==1)]['index'])
    date_vars = list(var_df[(var_df['date_time'] == 1)]['index'])
    id_vars = list(var_df[(var_df['id_col'] == 1)]['index'])
    sum_all_cols['int_vars'] = int_vars
    copy_date_vars = copy.deepcopy(date_vars)
    for date_var in copy_date_vars:
        try:
            pd.to_datetime(train[date_var],infer_datetime_format=True)
        except:
            cols_delete.append(date_var)
            date_vars.remove(date_var)
    sum_all_cols['date_vars'] = date_vars
    sum_all_cols['id_vars'] = id_vars
    sum_all_cols['cols_delete'] = cols_delete
    var_df['numeric'] = 0
    float_or_cat = var_df.apply(lambda x: 1 if x['type_of_column'] in ['float16',
                            'float32','float64'] else 0,
                                        axis=1)
    if len(var_df.loc[float_or_cat == 1]) > 0:
        for col in var_df.loc[float_or_cat == 1]['index'].values.tolist():
            if len(train[col].value_counts()) > 2 and len(train[col].value_counts()
                ) <= float_limit and len(train[col].value_counts()) <= len(train):
                var_df.loc[var_df['index']==col,'cat'] = 1
            else:
                if col not in (num_bool_vars + factor_vars):
                    var_df.loc[var_df['index']==col,'numeric'] = 1
    cat_vars = list(var_df[(var_df['cat'] ==1)]['index'])
    continuous_vars = list(var_df[(var_df['numeric'] ==1)]['index'])

    cat_vars_copy = copy.deepcopy(factor_vars)
    for cat in cat_vars_copy:
        if df_preds[cat].dtype==float:
            continuous_vars.append(cat)
            factor_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'dcat'] = 0
            var_df.loc[var_df['index']==cat,'numeric'] = 1
        elif len(df_preds[cat].value_counts()) == df_preds.shape[0]:
            id_vars.append(cat)
            factor_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'dcat'] = 0
            var_df.loc[var_df['index']==cat,'id_col'] = 1

    sum_all_cols['factor_vars'] = factor_vars
    cat_vars_copy = copy.deepcopy(cat_vars)
    for cat in cat_vars_copy:
        if df_preds[cat].dtype==float:
            continuous_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'numeric'] = 1
        elif len(df_preds[cat].value_counts()) == df_preds.shape[0]:
            id_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'id_col'] = 1
    sum_all_cols['cat_vars'] = cat_vars
    sum_all_cols['continuous_vars'] = continuous_vars
    sum_all_cols['id_vars'] = id_vars
    var_dict_sum = dict(zip(var_df.values[:,0], var_df.values[:,2:].sum(1)))
    for col, sumval in var_dict_sum.items():
        if sumval == 0:
            print('%s of type=%s is not classified' %(col,train[col].dtype))
        elif sumval > 1:
            print('%s of type=%s is classified into more then one type' %(col,train[col].dtype))
        else:
            pass
    copy_discretes = copy.deepcopy(discrete_string_vars)
    for each_discrete in copy_discretes:
        if train[each_discrete].nunique() >= 1000:
            nlp_vars.append(each_discrete)
            discrete_string_vars.remove(each_discrete)
        elif train[each_discrete].nunique() > 100 and train[each_discrete].nunique() < 1000:
            pass
        else:
            cat_vars.append(each_discrete)
            discrete_string_vars.remove(each_discrete)
    sum_all_cols['discrete_string_vars'] =  discrete_string_vars
    sum_all_cols['cat_vars'] = cat_vars
    sum_all_cols['nlp_vars'] = nlp_vars
    if verbose == 0:
        print("    Number of Numeric Columns = ", len(continuous_vars))
        print("    Number of Integer-Categorical Columns = ", len(int_vars))
        print("    Number of String-Categorical Columns = ", len(cat_vars))
        print("    Number of Factor-Categorical Columns = ", len(factor_vars))
        print("    Number of String-Boolean Columns = ", len(string_bool_vars))
        print("    Number of Numeric-Boolean Columns = ", len(num_bool_vars))
        print("    Number of Discrete String Columns = ", len(discrete_string_vars))
        print("    Number of NLP String Columns = ", len(nlp_vars))
        print("    Number of Date Time Columns = ", len(date_vars))
        print("    Number of ID Columns = ", len(id_vars))
        print("    Number of Columns to Delete = ", len(cols_delete))
    if verbose >= 2:
        print('  Printing upto %d columns max in each category:' %max_cols_to_print)
        print("    Numeric Columns : %s" %continuous_vars[:max_cols_to_print])
        print("    Integer-Categorical Columns: %s" %int_vars[:max_cols_to_print])
        print("    String-Categorical Columns: %s" %cat_vars[:max_cols_to_print])
        print("    Factor-Categorical Columns: %s" %factor_vars[:max_cols_to_print])
        print("    String-Boolean Columns: %s" %string_bool_vars[:max_cols_to_print])
        print("    Numeric-Boolean Columns: %s" %num_bool_vars[:max_cols_to_print])
        print("    Discrete String Columns: %s" %discrete_string_vars[:max_cols_to_print])
        print("    NLP text Columns: %s" %nlp_vars[:max_cols_to_print])
        print("    Date Time Columns: %s" %date_vars[:max_cols_to_print])
        print("    ID Columns: %s" %id_vars[:max_cols_to_print])
        print("    Columns that will not be considered in modeling: %s" %cols_delete[:max_cols_to_print])

    len_sum_all_cols = reduce(add,[len(v) for v in sum_all_cols.values()])
    if len_sum_all_cols == orig_cols_total:
        print('    %d Predictors classified...' %orig_cols_total)
        #print('This does not include the Target column(s)')
    else:
        print('No of columns classified %d does not match %d total cols. Continuing...' %(
                   len_sum_all_cols, orig_cols_total))
        ls = sum_all_cols.values()
        flat_list = [item for sublist in ls for item in sublist]
        if len(left_subtract(list(train),flat_list)) == 0:
            print(' Missing columns = None')
        else:
            print(' Missing columns = %s' %left_subtract(list(train),flat_list))
    return sum_all_cols


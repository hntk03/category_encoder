# ---
# jupyter:
#   jupytext_format_version: '1.2'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.3
#   toc:
#     base_numbering: 1
#     nav_menu: {}
#     number_sections: true
#     sideBar: true
#     skip_h1_title: false
#     title_cell: Table of Contents
#     title_sidebar: Contents
#     toc_cell: false
#     toc_position: {}
#     toc_section_display: true
#     toc_window_display: false
# ---

import gc
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed




# +
class CategoryEncoder:
    
        
    def __columns_type(self):
        
        feats = self.__feats
        category_columns = self.__category_columns
        numeric_columns = self.__numeric_columns
        dtypes = self.__dtypes
        type_object = type(object)
        
        for col in feats:
            if dtypes[col] == type_object:
                category_columns.append(col)
            else:
                numeric_columns.append(col)
                
        for col in category_columns:
            feats.remove(col)
        
        self.__category_columns = category_columns
        self.__numeric_columns = numeric_columns
        self.__feats = feats
    
    def __init__(self, train_df, test_df, id_column, target_column):
        
        self.__train_df = train_df
        self.__test_df = test_df
        self.__category_columns = []
        self.__numeric_columns = []
        self.__columns = train_df.columns.values
        self.__train_shape = train_df.shape
        self.__test_shape = test_df.shape
        self.__dtypes = train_df.dtypes
        self.__id_column = id_column
        self.__target_column = target_column
        columns = list(np.copy(self.__columns))
        columns.remove(id_column)
        columns.remove(target_column)
        self.__feats = columns
        self.__columns_type()
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        self.__logger = logger
        
    def __drop_columns(self, category_columns):
        
        train_df = self.__train_df
        test_df = self.__test_df
        
        for col in category_columns:
            train_df.drop(col, axis=1, inplace=True)
            test_df.drop(col, axis=1, inplace=True)
            
        gc.collect()
        
        self.__train_df = train_df
        self.__test_df = test_df
    
    def get_columns_type(self):
        
        return self.__numeric_columns, self.__category_columns
    
    def __fill_nan(self, category_columns):
        
        train_df = self.__train_df
        test_df = self.__test_df
        
        for col in category_columns:
            train_df[col].fillna('Nan', inplace=True)
            test_df[col].fillna('Nan', inplace=True)
            
        self.__train_df = train_df 
        self.__test_df = test_df 
    
    def __encoder(self, col):
        
        feats = []
        train_shape = self.__train_shape
        test_shape = self.__test_shape
        
        traincol_series = self.__train_df[col] 
        testcol_series = self.__test_df[col] 
        
        _train_ce = np.zeros(train_shape[0], dtype=int)
        _test_ce = np.zeros(test_shape[0], dtype=int)
        
        _train_lce = np.zeros(train_shape[0], dtype=int)
        _test_lce = np.zeros(test_shape[0], dtype=int)
        
        
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        
        le = LabelEncoder()

        # Label Encoding
        _train = traincol_series.values
        _test = testcol_series.values
        _all = np.r_[_train, _test]

        le.fit(_all)
        _train_le = le.transform(_train)
        _test_le = le.transform(_test)

        col_LE = col + '_LE'
        train_df[col_LE] = _train_le
        test_df[col_LE] = _test_le
        feats.append(col_LE)

        # Count Encoding and Label Count Encoding
        value_counts = train_df[col_LE].value_counts()
        _index = value_counts.index.values
        rank = pd.Series(np.arange(_index.shape[0], 0, -1), index=_index)

        #train
        for iter, value in enumerate(_train_le):
            _train_ce[iter] = value_counts[value]
            _train_lce[iter] = rank[value]

        _set_index  = set(_index)
        for iter, value in enumerate(_test_le):
            if set([value]) <= _set_index:
                _test_ce[iter] = value_counts[value]
                _test_lce[iter] = rank[value]
            else:
                _test_ce[iter] = 1
                _test_lce[iter] = 1


        col_CE = col + '_CE'
        train_df[col_CE] = _train_ce
        test_df[col_CE] = _test_ce
        feats.append(col_CE)

        col_LCE = col + '_LCE'
        train_df[col_LCE] = _train_lce
        test_df[col_LCE] = _test_lce
        feats.append(col_LCE)
        
        return train_df, test_df, feats

    
    def encoding(self, mycategory_columns=None, n_jobs=-1):
        
        logger = self.__logger
        
        feats = self.__feats
        train_shape = self.__train_shape
        test_shape = self.__test_shape
        
        if mycategory_columns == None:
            category_columns = self.__category_columns
        else:
            category_columns = mycategory_columns
            
        self.__fill_nan(category_columns) 
        
        train_df = self.__train_df
        test_df = self.__test_df
            
            
        _train_ce = np.zeros(train_shape[0], dtype=int)
        _test_ce = np.zeros(test_shape[0], dtype=int)
        
        _train_lce = np.zeros(train_shape[0], dtype=int)
        _test_lce = np.zeros(test_shape[0], dtype=int)
        
        encoder = self.__encoder 
        result = (Parallel(n_jobs=n_jobs, verbose=3)([delayed(encoder)( x) for x in category_columns]))
                    
        for iter, r in enumerate(result):
            
            train_df = pd.concat([train_df, r[0]], axis=1)
            test_df = pd.concat([train_df, r[1]], axis=1)
            feats.extend(r[2])
            
            
        self.__train_df = train_df
        self.__test_df = test_df
        self.__feats = feats
                
        self.__drop_columns(category_columns)
        
        
        
    def get_df(self):
        
        train_df = self.__train_df
        test_df = self.__test_df
        feats = self.__feats
        
        return train_df, test_df, feats
        
            
                
                
            
            
            
            
            
            
         
# -


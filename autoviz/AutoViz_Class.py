## Thank to GoogleLLC
## For Education Purpose only
# import 
import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import io
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import re
import pdb
import pprint
import matplotlib
matplotlib.style.use('seaborn')
from itertools import cycle, combinations
from collections import defaultdict
import copy
import time
import sys
import random
import xlrd
import statsmodels
from io import BytesIO
import base64
from functools import reduce
import traceback
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from autoviz.AutoViz_Holo import AutoViz_Holo
from autoviz.AutoViz_Utils import save_image_data, save_html_data, analyze_problem_type, draw_pivot_tables, draw_scatters
from autoviz.AutoViz_Utils import draw_pair_scatters, plot_fast_average_num_by_cat, draw_barplots, draw_heatmap
from autoviz.AutoViz_Utils import draw_distplot, draw_violinplot, draw_date_vars, catscatter, draw_catscatterplots
from autoviz.AutoViz_Utils import list_difference, search_for_word_in_list, analyze_ID_columns, start_classifying_vars
from autoviz.AutoViz_Utils import analyze_columns_in_dataset, find_remove_duplicates, load_file_dataframe, classify_print_vars
from autoviz.AutoViz_Utils import marthas_columns, EDA_find_remove_columns_with_infinity, return_dictionary_list
from autoviz.AutoViz_Utils import remove_variables_using_fast_correlation, count_freq_in_list, find_corr_vars, left_subtract
from autoviz.AutoViz_Utils import convert_train_test_cat_col_to_numeric, return_factorized_dict, convert_a_column_to_numeric
from autoviz.AutoViz_Utils import convert_all_object_columns_to_numeric, find_top_features_xgb, convert_a_mixed_object_column_to_numeric
from autoviz.AutoViz_NLP import draw_word_clouds

# func
class AutoViz_Class():
    def __init__(self):
        self.overall = {
        'name': 'overall',
        'plots': [],
        'heading': [],
        'subheading':[],  #"\n".join(subheading)
        'desc': [],  #"\n".join(subheading)
        'table1_title': "",
        'table1': [],
        'table2_title': "",
        'table2': []
            }  
        self.scatter_plot = {
        'name': 'scatter',
        'heading': 'Scatter Plot of each Continuous Variable against Target Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        } 
        self.pair_scatter = {
        'name': 'pair-scatter',
        'heading': 'Pairwise Scatter Plot of each Continuous Variable against other Continuous Variables',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': []  #"\n".join(desc)
        } 
        self.dist_plot = {
        'name': 'distribution',
        'heading': 'Distribution Plot of Target Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': []  #"\n".join(desc)
        } 
        self.pivot_plot = {
        'name': 'pivot',
        'heading': 'Pivot Plots of all Continuous Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        } 
        self.violin_plot = {
        'name': 'violin',
        'heading': 'Violin Plots of all Continuous Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  
        self.heat_map = {
        'name': 'heatmap',
        'heading': 'Heatmap of all Continuous Variables for target Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }   
        self.bar_plot = {
        'name': 'bar',
        'heading': 'Bar Plots of Average of each Continuous Variable by Target Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  
        self.date_plot = {
        'name': 'time-series',
        'heading': 'Time Series Plots of Two Continuous Variables against a Date/Time Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  
        self.wordcloud = {
        'name': 'wordcloud',
        'heading': 'Word Cloud Plots of NLP or String vars',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  
        self.catscatter_plot = {
        'name': 'catscatter',
        'heading': 'Cat-Scatter  Plots of categorical vars',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  


    def add_plots(self,plotname,X):
        if X is None:
            pass
        else:
            getattr(self, plotname)["plots"].append(X)

    def add_subheading(self,plotname,X):
        if X is None:
            pass
        else:
            getattr(self,plotname)["subheading"].append(X)

    def AutoViz(self, filename, sep=',', depVar='', dfte=None, header=0, verbose=1,
                            lowess=False,chart_format='svg',max_rows_analyzed=150000,
                                max_cols_analyzed=30, save_plot_dir=None):
        if isinstance(depVar, list):
            print('Since AutoViz cannot visualize multi-label targets, choosing first item in targets: %s' %depVar[0])
            depVar = depVar[0]
        
        if chart_format.lower() in ['bokeh','server','bokeh_server','bokeh-server', 'html']:
            dft = AutoViz_Holo(filename, sep, depVar, dfte, header, verbose,
                        lowess,chart_format,max_rows_analyzed,
                            max_cols_analyzed, save_plot_dir)
        else:
            dft = self.AutoViz_Main(filename, sep, depVar, dfte, header, verbose,
                        lowess,chart_format,max_rows_analyzed,
                            max_cols_analyzed, save_plot_dir)
        return dft
    
    def AutoViz_Main(self, filename, sep=',', depVar='', dfte=None, header=0, verbose=0,
                            lowess=False,chart_format='svg',max_rows_analyzed=150000,
                                max_cols_analyzed=30, save_plot_dir=None):
        corr_limit = 0.8
        
        if not depVar is None:
            if isinstance(depVar, list):
                target_dir = depVar[0]
            elif isinstance(depVar, str):
                if depVar == '':
                    target_dir = 'AutoViz'
                else:
                    target_dir = copy.deepcopy(depVar)
        else:
            target_dir = 'AutoViz'
        if save_plot_dir is None:
            mk_dir = os.path.join(".","AutoViz_Plots")
        else:
            mk_dir = copy.deepcopy(save_plot_dir)
        if verbose == 2 and not os.path.isdir(mk_dir):
            os.mkdir(mk_dir)
        mk_dir = os.path.join(mk_dir,target_dir)
        if verbose == 2 and not os.path.isdir(mk_dir):
            os.mkdir(mk_dir)
        start_time = time.time()
        try:
            dft, depVar,IDcols,bool_vars,cats,continuous_vars,discrete_string_vars,date_vars,classes,problem_type,selected_cols = classify_print_vars(
                                                filename,sep,max_rows_analyzed, max_cols_analyzed,
                                                depVar,dfte,header,verbose)
        except:
            print('Not able to read or load file. Please check your inputs and try again...')
            return None
        if verbose >= 1:
            print('To fix data quality issues automatically, import FixDQ from autoviz...')
            data_cleaning_suggestions(dft, target=depVar)

        if depVar == None or depVar == '':
            if len(continuous_vars) > 1:
                try:
                    svg_data = draw_pair_scatters(dft,continuous_vars,problem_type,verbose,chart_format,
                                                    depVar,classes,lowess, mk_dir)
                    self.add_plots('pair_scatter',svg_data)
                except Exception as e:
                    print(e)
                    print('Could not draw Pair Scatter Plots')
            try:
                svg_data = draw_distplot(dft, bool_vars+cats, continuous_vars,verbose,chart_format,problem_type,
                                    depVar,classes, mk_dir)
                self.add_plots('dist_plot',svg_data)
            except:
                print('Could not draw Distribution Plot')
            try:
                if len(continuous_vars) > 0:
                    svg_data = draw_violinplot(dft,depVar,continuous_vars,verbose,chart_format,problem_type, mk_dir)
                    self.add_plots('violin_plot',svg_data)
                else:
                    svg_data = draw_pivot_tables(dft, problem_type, verbose,
                        chart_format,depVar,classes, mk_dir)
                    self.add_plots('pivot_plot',svg_data)
            except:
                print('Could not draw Distribution Plots')
            try:
                numeric_cols = dft.select_dtypes(include='number').columns.tolist()
                numeric_cols = list_difference(numeric_cols, date_vars)
                svg_data = draw_heatmap(dft, numeric_cols, verbose,chart_format, date_vars, depVar,
                                    problem_type,classes, mk_dir)
                self.add_plots('heat_map',svg_data)
            except:
                print('Could not draw Heat Map')
            if date_vars != [] and len(continuous_vars) > 0:
                try:
                    svg_data = draw_date_vars(dft,depVar,date_vars,
                                              continuous_vars,verbose,chart_format,problem_type, mk_dir)
                    self.add_plots('date_plot',svg_data)
                except:
                    print('Could not draw Date Vars')
            if len(continuous_vars) > 0 and len(cats) > 0:
                try:
                    svg_data = draw_barplots(dft,cats,continuous_vars, problem_type,
                                    verbose,chart_format,depVar,classes, mk_dir)
                    self.add_plots('bar_plot',svg_data)
                except:
                    print('Could not draw Bar Plots')
            else:
                if len(cats) > 1:
                    try:
                        svg_data = draw_catscatterplots(dft,cats, problem_type, verbose, 
                                    chart_format, mk_dir=None)
                        self.add_plots('catscatter_plot',svg_data)
                    except:
                        print ('Could not draw catscatter plots...')
        else:
            if problem_type=='Regression':
                if len(continuous_vars) > 0:
                    try:
                        svg_data = draw_scatters(dft,
                                        continuous_vars,verbose,chart_format,problem_type,depVar,classes,lowess, mk_dir)
                        self.add_plots('scatter_plot',svg_data)
                    except Exception as e:
                        print("Exception Drawing Scatter Plots")
                        print(e)
                        traceback.print_exc()
                        print('Could not draw Scatter Plots')
                if len(continuous_vars) > 1:
                    try:
                        svg_data = draw_pair_scatters(dft,continuous_vars,problem_type,verbose,chart_format,
                                                        depVar,classes,lowess, mk_dir)
                        self.add_plots('pair_scatter',svg_data)
                    except:
                        print('Could not draw Pair Scatter Plots')
                try:
                    if type(depVar) == str:
                        othernums = [x for x in continuous_vars if x not in [depVar]]
                    else:
                        othernums = [x for x in continuous_vars if x not in depVar]
                    if len(othernums) >= 1:
                        svg_data = draw_distplot(dft, bool_vars+cats, continuous_vars,verbose,chart_format,
                                            problem_type, depVar, classes, mk_dir)
                        self.add_plots('dist_plot',svg_data)
                except:
                    print('Could not draw some Distribution Plots')
                try:
                    if len(continuous_vars) > 0:
                        svg_data = draw_violinplot(dft,depVar,continuous_vars,verbose,chart_format,problem_type, mk_dir)
                        self.add_plots('violin_plot',svg_data)
                except:
                    print('Could not draw Violin Plots')
                try:
                    numeric_cols = [x for x in dft.select_dtypes(include='number').columns.tolist() if x not in [depVar]]
                    numeric_cols = list_difference(numeric_cols, date_vars)
                    svg_data = draw_heatmap(dft,
                                        numeric_cols, verbose,chart_format, date_vars, depVar,
                                            problem_type, classes, mk_dir)
                    self.add_plots('heat_map',svg_data)
                except:
                    print('Could not draw some Heat Maps')
                if date_vars != [] and len(continuous_vars) > 0:
                    try:
                        svg_data = draw_date_vars(
                            dft,depVar,date_vars,continuous_vars,verbose,chart_format,problem_type, mk_dir)
                        self.add_plots('date_plot',svg_data)
                    except:
                        print('Could not draw some Time Series plots')
                if len(cats) > 0 and len(continuous_vars) == 0:
                    try:
                        svg_data = draw_pivot_tables(dft, problem_type, verbose,
                            chart_format,depVar,classes, mk_dir)
                        self.add_plots('pivot_plot',svg_data)
                    except:
                        print('Could not draw some Pivot Charts against Dependent Variable')
                if len(continuous_vars) > 0 and len(cats) > 0:
                    try:
                        svg_data = draw_barplots(dft, find_remove_duplicates(cats+bool_vars),continuous_vars,
                                                    problem_type, verbose,chart_format,depVar,classes, mk_dir)
                        self.add_plots('bar_plot',svg_data)
                        #self.add_plots('bar_plot',None)
                    except:
                        print('Could not draw some Bar Charts')
                else:
                    if len(cats) > 1:
                        try:
                            svg_data = draw_catscatterplots(dft,cats, problem_type, verbose, 
                                        chart_format, mk_dir=None)
                            self.add_plots('catscatter_plot',svg_data)
                        except:
                            print ('Could not draw catscatter plots...')
            else :
                if len(continuous_vars) > 0:
                    try:
                        svg_data = draw_scatters(dft,continuous_vars,
                                                 verbose,chart_format,problem_type,depVar, classes,lowess, mk_dir)
                        self.add_plots('scatter_plot',svg_data)
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        print('Could not draw some Scatter Plots')
                if len(continuous_vars) > 1:
                    try:
                        svg_data = draw_pair_scatters(dft,continuous_vars,
                                                      problem_type,verbose,chart_format,depVar,classes,lowess, mk_dir)
                        self.add_plots('pair_scatter',svg_data)
                    except:
                        print('Could not draw some Pair Scatter Plots')
                try:
                    if type(depVar) == str:
                        othernums = [x for x in continuous_vars if x not in [depVar]]
                    else:
                        othernums = [x for x in continuous_vars if x not in depVar]

                    if len(othernums) >= 1:
                        svg_data = draw_distplot(dft, bool_vars+cats, continuous_vars,verbose,chart_format,
                                                problem_type,depVar,classes, mk_dir)
                        self.add_plots('dist_plot',svg_data)
                    else:
                        print('No continuous var in data set: drawing categorical distribution plots')
                except:
                    print('Could not draw some Distribution Plots')
                try:
                    if len(continuous_vars) > 0:
                        svg_data = draw_violinplot(dft,depVar,continuous_vars,verbose,chart_format,problem_type, mk_dir)
                        self.add_plots('violin_plot',svg_data)
                except:
                    print('Could not draw some Violin Plots')
                try:
                    numeric_cols = [x for x in dft.select_dtypes(include='number').columns.tolist() if x not in [depVar]]
                    numeric_cols = list_difference(numeric_cols, date_vars)
                    svg_data = draw_heatmap(dft, numeric_cols,
                                            verbose,chart_format, date_vars, depVar,problem_type,
                                            classes, mk_dir)
                    self.add_plots('heat_map',svg_data)
                except:
                    print('Could not draw some Heat Maps')
                if date_vars != [] and len(continuous_vars) > 0:
                    try:
                        svg_data = draw_date_vars(dft,depVar,date_vars,
                                                  continuous_vars,verbose,chart_format,problem_type, mk_dir)
                        self.add_plots('date_plot',svg_data)
                    except:
                        print('Could not draw some Time Series plots')
                if len(cats) > 0 and len(continuous_vars) == 0:
                    try:
                        svg_data = draw_pivot_tables(dft, problem_type, verbose,
                                        chart_format,depVar,classes, mk_dir)
                        self.add_plots('pivot_plot',svg_data)
                    except:
                        print('Could not draw some Pivot Charts against Dependent Variable')
                if len(continuous_vars) > 0 and len(cats) > 0:
                    try:
                        svg_data = draw_barplots(dft,find_remove_duplicates(cats+bool_vars),continuous_vars,problem_type,
                                        verbose,chart_format, depVar, classes, mk_dir)
                        self.add_plots('bar_plot',svg_data)
                        pass
                    except:
                        if verbose <= 1:
                            print('Could not draw some Bar Charts')
                        pass
                else:
                    if len(cats) > 1:
                        try:
                            svg_data = draw_catscatterplots(dft,cats, problem_type, verbose, 
                                        chart_format, mk_dir=None)
                            self.add_plots('catscatter_plot',svg_data)
                        except:
                            print ('Could not draw catscatter plots...')
        if len(discrete_string_vars) > 0:
            plotname = 'wordcloud'
            import nltk
            nltk.download('popular')
            for each_string_var in discrete_string_vars:
                try:
                    svg_data = draw_word_clouds(dft, each_string_var, chart_format, plotname, 
                                    depVar, problem_type, classes, mk_dir, verbose=0)
                    self.add_plots(plotname,svg_data)
                except:
                    print('Could not draw wordcloud plot for %s' %each_string_var)
        if verbose <= 1:
            print('All Plots done')
        else:
            print('All Plots are saved in %s' %mk_dir)
        print('Time to run AutoViz = %0.0f seconds ' %(time.time()-start_time))
        if verbose <= 1:
            print ('\n Done Visualizing!!')
        return dft
from pandas_dq import Fix_DQ
class FixDQ(Fix_DQ):
    def __init__(self, quantile=0.87, cat_fill_value = 'missing', 
                num_fill_value = 9999, rare_threshold = 0.01, 
                correlation_threshold = 0.9):
        super().__init__()  
        self.quantile = quantile
        self.cat_fill_value = cat_fill_value
        self.num_fill_value = num_fill_value
        self.rare_threshold = rare_threshold
        self.correlation_threshold = correlation_threshold

from pandas_dq import dq_report
def data_cleaning_suggestions(df, target=None):
    if isinstance(df, pd.DataFrame):
        dqr = dq_report(data=df, target=target, html=False, csv_engine="pandas", verbose=1)
    else:
        print("Input must be a dataframe. Please check input and try again.")
    return dqr
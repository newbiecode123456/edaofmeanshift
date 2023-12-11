import numpy as np
import pandas as pd
from autoviz.AutoViz_Utils import *
import hvplot.pandas  # noqa
import copy
import pdb

import warnings
warnings.filterwarnings("ignore")
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import logging
logging.getLogger("param").setLevel(logging.ERROR)
from bokeh.util.warnings import BokehUserWarning 
import warnings 
warnings.simplefilter(action='ignore', category=BokehUserWarning)
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from matplotlib import io
import io
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import re
import pdb
import pprint
import matplotlib
matplotlib.style.use('fivethirtyeight')
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
import os

import hvplot.pandas
import holoviews as hv
from holoviews import opts
#hv.notebook_extension('bokeh')
hv.extension('bokeh', 'matplotlib')
#hv.extension('bokeh')
import panel as pn
import panel.widgets as pnw
import holoviews.plotting.bokeh

def save_image_data_hv(fig, chart_count, chart_format):
    if chart_format == 'svg':
        imgdata = io.StringIO()
        fig.savefig(imgdata, format=chart_format)
        imgdata.seek(0)
        svg_data = imgdata.getvalue()
        return svg_data
    elif chart_format in ['png','jpg']:
        imgdata = BytesIO()
        fig.savefig(imgdata, format=chart_format, bbox_inches='tight', pad_inches=0.0)
        imgdata.seek(0)
        figdata_png = base64.b64encode(imgdata.getvalue())
        return figdata_png
def append_panels(hv_panel, imgdata_list, chart_format):
    imgdata_list.append(hv.output(hv_panel, backend='bokeh', fig=chart_format))
    return imgdata_list
def display_dmap(dmap):
    renderer = hv.renderer('bokeh')
    app = renderer.app(dmap)
    server = renderer.app(dmap, show=True, new_window=True)
    return server
def display_obj(dmap_in):
    renderer = hv.renderer('bokeh')
    app = renderer.app(dmap_in)
    server = renderer.app(dmap_in, show=True, new_window=True)
    display(server)
def display_server(dmap):
    server = pn.serve(dmap, start=True, show=False)
    return server
def AutoViz_Holo(filename, sep=',', depVar='', dfte=None, header=0, verbose=0,
                        lowess=False,chart_format='svg',max_rows_analyzed=150000,
                            max_cols_analyzed=30, save_plot_dir=None):

    corr_limit = 0.7

    target_dir = 'AutoViz'
    if not depVar is None:
        if depVar != '':
            target_dir = copy.deepcopy(depVar)
    if save_plot_dir is None:
        mk_dir = os.path.join(".","AutoViz_Plots")
    else:
        mk_dir = copy.deepcopy(save_plot_dir)
    if chart_format == 'html' and not os.path.isdir(mk_dir):
        os.mkdir(mk_dir)
    mk_dir = os.path.join(mk_dir,target_dir)
    if chart_format == 'html' and not os.path.isdir(mk_dir):
        os.mkdir(mk_dir)

    start_time = time.time()
    try:
        dfin, dep,IDcols,bool_vars,cats,nums,discrete_string_vars,date_vars,classes,problem_type,selected_cols = classify_print_vars(
                                            filename,sep,max_rows_analyzed, max_cols_analyzed,
                                            depVar,dfte,header,verbose)
    except:
        print('Not able to read or load file. Please check your inputs and try again...')
        return None

    ls_objects = []
    imgdata_list = list()
    height_size = 400
    width_size = 500
    

    if len(nums) > 0:
        if problem_type == 'Clustering':

            print('No scatter plots with depVar when no depVar is given.')
        else:
            drawobj1 = draw_scatters_hv(dfin,nums,chart_format,problem_type,
                          dep, classes, lowess, mk_dir, verbose)
            ls_objects.append(drawobj1)

        if len(nums) >= 2:
            drawobj2 = draw_pair_scatters_hv(dfin, nums, problem_type, chart_format, dep,
                           classes, lowess, mk_dir, verbose)
            ls_objects.append(drawobj2)
    drawobj3 = draw_distplot_hv(dfin, cats, nums, chart_format, problem_type, dep, classes, mk_dir, verbose)
    ls_objects.append(drawobj3)

    drawobj4 = draw_kdeplot_hv(dfin, cats, nums, chart_format, problem_type, dep, ls_objects, mk_dir, verbose)
    if not drawobj4:

        ls_objects = copy.deepcopy(drawobj4)
    if len(nums) > 0:
        drawobj5 = draw_violinplot_hv(dfin, dep, nums, chart_format, problem_type, mk_dir, verbose)
        ls_objects.append(drawobj5)
    if len(nums) > 0:
        drawobj6 = draw_heatmap_hv(dfin, nums, chart_format, date_vars, dep, problem_type, classes, 
                            mk_dir, verbose)
        ls_objects.append(drawobj6)
    if len(date_vars) > 0:
        drawobj7 = draw_date_vars_hv(dfin,dep,date_vars, nums, chart_format, problem_type, mk_dir, verbose)
        ls_objects.append(drawobj7)
    if len(nums) > 0 and len(cats) > 0:
        drawobj8 = draw_cat_vars_hv(dfin, dep, nums, cats, chart_format, problem_type, mk_dir, verbose)
    print('Time to run AutoViz (in seconds) = %0.0f' %(time.time()-start_time))
    return dfin

def draw_cat_vars_hv(dfin, dep, nums, cats, chart_format, problem_type, mk_dir, verbose=0):

    dft = copy.deepcopy(dfin)
    image_count = 0
    imgdata_list = list()
    N = len(nums)
    cols = 2
    width_size = 600
    height_size = 400
    jitter = 0.05
    alpha = 0.5
    size = 5
    transparent = 0.5
    colortext = 'brycgkbyrcmgkbyrcmgkbyrcmgkbyr'
    colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
    plot_name = 'cat_var_plots'
    if problem_type == 'Clustering':

        pass
    elif problem_type == 'Regression':
        if isinstance(dep, str):
            if dep not in nums:
                nums.append(dep)
        else:
            nums += dep
            nums = find_remove_duplicates(nums)
    else:
        if isinstance(dep, str):
            if dep not in cats:
                cats.append(dep)
        else:
            cats += dep
            cats = find_remove_duplicates(cats)
    quantileable = [x for x in nums if len(dft[x].unique()) > 20]
    if len(quantileable) <= 1:
        quantileable = [x for x in nums if len(dft[x].unique()) > 2]
    cmap_list = ['Blues','rainbow', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
    x = pnw.Select(name='X-Axis', value=cats[0], options=cats)
    y = pnw.Select(name='Y-Axis', value=quantileable[0], options=quantileable)

    @pn.depends(x.param.value, y.param.value) 
    def create_figure(x, y):
        opts = dict(cmap=cmap_list[0], line_color='black')
        #opts['size'] = bubble_size
        opts['alpha'] = alpha
        opts['tools'] = ['hover']
        opts['toolbar'] = 'above'
        opts['colorbar'] = True
        conti_df = dft[[x,y]].groupby(x).mean().reset_index()
        return hv.Bars(conti_df).opts(width=width_size, height=height_size, 
                xrotation=70, title='Average of each numeric var by categorical var')

    widgets = pn.WidgetBox(x, y)

    hv_panel = pn.Row(widgets, create_figure).servable('Cross-selector')    
    if verbose == 2:
        imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
        image_count += 1
    if chart_format in ['server', 'bokeh_server', 'bokeh-server']:
        #server = pn.serve(hv_panel, start=True, show=True)
        print('%s can be found in URL below:' %plot_name)
        hv_panel.show()
    elif chart_format == 'html':
        save_html_data(hv_panel, chart_format, plot_name, mk_dir)
    else:
        display(hv_panel)  ### drawobj.show()        
    return hv_panel
def draw_kdeplot_hv(dfin, cats, nums, chart_format, problem_type, dep, ls_objects, mk_dir, verbose=0):
    dft = copy.deepcopy(dfin)
    image_count = 0
    imgdata_list = list()
    N = len(nums)
    cols = 2
    plot_name = 'kde_plots'
    width_size = 600
    height_size = 400
    def return_dynamic_objects(dfout, dep, title='Distribution of Target variable'):
        width_size = 600
        height_size = 400
        pdf1 = pd.DataFrame(dfout[dep].value_counts().reset_index())
        pdf2 = pd.DataFrame(dfout[dep].value_counts(1).reset_index())
        drawobj41 = pdf1.hvplot(kind='bar', color='lightblue', title=title).opts(
                        height=height_size, width=width_size,xrotation=70)
        drawobj42 = pdf2.hvplot(kind='bar', color='lightgreen', title=title)
        return (drawobj41+drawobj42)

    if problem_type.endswith('Classification'):
        colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
        dmap = hv.DynamicMap(return_dynamic_objects(dfin, dep, title='Percent Distribution of Target variable'
                        ).opts(shared_axes=False).opts(title='Histogram and KDE of Target = %s' %dep)).opts(
                            height=height_size, width=width_size)
        dmap.opts(framewise=True,axiswise=True) 
        hv_all = pn.pane.HoloViews(dmap)#, sizing_mode="stretch_both")
        #ls_objects.append(drawobj41)
        #ls_objects.append(drawobj42)
    else:
        if not isinstance(dep, list):
            if dep == '':
                return ls_objects
            else:
                dmap = hv.DynamicMap(return_dynamic_objects(dfin, dep, title=f'Histogram and KDE of Target = {dep}')).opts(width=width_size)
                dmap.opts(framewise=True,axiswise=True) ## both must be True for your charts to have dynamically varying axes!
                hv_all = pn.pane.HoloViews(dmap)
                #ls_objects.append(drawobj41)
                #ls_objects.append(drawobj42)
    if verbose == 2:
        imgdata_list = append_panels(hv_all, imgdata_list, chart_format)
        image_count += 1
    if chart_format.lower() in ['server', 'bokeh_server', 'bokeh-server']:
        ### If you want it on a server, you use drawobj.show()
        #(drawobj41+drawobj42).show()
        print('%s can be found in URL below:' %plot_name)
        server = pn.serve(hv_all, start=True, show=True)
    elif chart_format == 'html':
        save_html_data(hv_all, chart_format, plot_name, mk_dir)
    else:
        display(hv_all)
    return ls_objects
def draw_scatters_hv(dfin, nums, chart_format, problem_type,
                  dep=None, classes=None, lowess=False, mk_dir='AutoViz_Plots', verbose=0):
    dfin = copy.deepcopy(dfin)
    dft = copy.deepcopy(dfin)
    image_count = 0
    imgdata_list = list()
    classes = copy.deepcopy(classes)
    N = len(nums)
    cols = 2
    width_size = 600
    height_size = 400
    jitter = 0.05
    alpha = 0.5
    bubble_size = 10
    colortext = 'brycgkbyrcmgkbyrcmgkbyrcmgkbyr'
    colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
    plot_name = 'scatterplots'

    if problem_type == 'Regression':
        alpha = 0.5
        colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
        def load_symbol(symbol, **kwargs):
            color = next(colors)
            return hv.Scatter((dft[symbol].values,dft[dep].values)).opts(framewise=True).opts(size=bubble_size,
                    color=color, alpha=alpha, height=height_size, width=width_size).opts(
                    xlabel='%s' %symbol).opts(ylabel='%s' %dep).opts(
                   title='Scatter Plot of %s against %s variable' %(symbol,dep))
        dmap = hv.DynamicMap(load_symbol, kdims='Select_Variable').redim.values(Select_Variable=nums).opts(framewise=True)
        hv_panel = pn.panel(dmap)
        widgets = hv_panel[0]
        hv_all = pn.Column(pn.Row(*widgets))
        if verbose == 2:
            imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
            image_count += 1
    else:
        target_vars = dft[dep].unique()
        x = pn.widgets.Select(name='x', options=nums)
        y = pn.widgets.Select(name='y', options=nums)
        kind = pn.widgets.Select(name='kind', value='scatter', options=['scatter'])
        hv_plot = dft.hvplot(x=dep, y=y, kind=kind, height=height_size, width=width_size, size=bubble_size,
                        title='Scatter Plot of each independent numeric variable against target variable')
        hv_panel = pn.panel(hv_plot)
        hv_all = pn.Row(pn.WidgetBox(y), hv_plot)

        #    hv_string = ''
        #    target_list = np.unique(dfin[dep].values)
        #    lowerbound = dfin[Select_numeric_variable].min()
        #    upperbound = dfin[Select_numeric_variable].max()
        #    for each_t in target_list:
        #        if not isinstance(each_t, str):
        #            each_ts = str(each_t) 
        #        else:
        #            each_ts = copy.deepcopy(each_t)
        #        next_color = next(colors)
        #        #add_string = "hv.Scatter((dfin[dfin['"+dep+"']=="+each_ts+"]['"+dep+"'].values,dfin[dfin['"+dep+"']=="+each_ts+"]['"+Select_numeric_variable+"'].values)).opts(color='"+next_color+"',jitter=eval('"+str(jitter)+"'),alpha=eval('"+str(alpha)+"'),size=eval('"+str(bubble_size)+"'),height=eval('"+str(height_size)+"'),width=eval('"+str(width_size)+"'))"
        #        add_string = "hv.Scatter((dfin[dfin['"+dep+"']=="+each_ts+"]['"+dep+"'].values,dfin[dfin['"+dep+"']=="+each_ts+"]['"+Select_numeric_variable+"'].values)).opts(color='"+next_color+"',jitter=eval('"+str(jitter)+"'),alpha=eval('"+str(alpha)+"'),ylim=(eval('"+str(lowerbound)+"'),eval('"+str(upperbound)+"')),height=eval('"+str(height_size)+"'),width=eval('"+str(width_size)+"'))"
        #        hv_string += add_string + " * "
        #    return eval(hv_string[:-2]).opts(
        #            legend_position='top_left',title='Scatter Plot of each Numeric Variable against Target variable')
        #dmap = hv.DynamicMap(select_widget,  kdims=['Select_numeric_variable']).redim.values(Select_numeric_variable=nums).opts(framewise=True)
        #hv_panel = pn.panel(dmap)
        #widgets = hv_panel[0]
        #hv_all = pn.Column(pn.Row(*widgets))
        if verbose == 2:
            imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
            image_count += 1
    if chart_format in ['server', 'bokeh_server', 'bokeh-server']:
        #server = pn.serve(hv_all, start=True, show=True)
        print('%s can be found in URL below:' %plot_name)
        hv_all.show()
    elif chart_format == 'html':
        save_html_data(hv_all, chart_format, plot_name, mk_dir)
    else:
        display(hv_all)  ###  drawobj.show()        
    return hv_all

def draw_pair_scatters_hv(dfin,nums,problem_type,chart_format, dep=None,
                       classes=None, lowess=False, mk_dir='AutoViz_Plots', verbose=0):


    dft = dfin[:]
    image_count = 0
    imgdata_list = list()
    if len(nums) <= 1:
        return
    classes = copy.deepcopy(classes)
    height_size = 400
    width_size = 600
    alpha = 0.5
    bubble_size = 10
    cmap_list = ['rainbow', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
    plot_name = 'pair_scatters'
    if problem_type in ['Regression', 'Clustering']:
        #x = pn.widgets.Select(name='x', options=nums)
        #y = pn.widgets.Select(name='y', options=nums)
        #kind = pn.widgets.Select(name='kind', value='scatter', options=['bivariate', 'scatter'])
        #xlimi = (dft[x.value].min(), dft[x.value].max())
        #ylimi = (dft[y.value].min(), dft[y.value].max())
        #plot = dft.hvplot(x=x, y=y, kind=kind,  color=next(colors), alpha=0.5, xlim=xlimi, ylim=ylimi,
        #            title='Pair-wise Scatter Plot of two Independent Numeric variables')
        #hv_panel = pn.Row(pn.WidgetBox(x, y, kind),plot)
        
        quantileable = [x for x in nums if len(dft[x].unique()) > 20]
        if len(quantileable) <= 1:
            quantileable = [x for x in nums if len(dft[x].unique()) > 2]
        
        x = pnw.Select(name='X-Axis', value=quantileable[0], options=quantileable)
        y = pnw.Select(name='Y-Axis', value=quantileable[1], options=quantileable)
        size = pnw.Select(name='Size', value='None', options=['None'] + quantileable)
        if problem_type == 'Clustering':
            color = pnw.Select(name='Color', value='None', options=['None'])
        else:
            color = pnw.Select(name='Color', value='None', options=['None', dep])
        @pn.depends(x.param.value, y.param.value, color.param.value) 
        def create_figure(x, y, color):
            opts = dict(cmap=cmap_list[0], width=width_size, height=height_size, line_color='black')
            if color != 'None':
                opts['color'] = color 
            opts['size'] = bubble_size
            opts['alpha'] = alpha
            opts['tools'] = ['hover']
            opts['toolbar'] = 'above'
            opts['colorbar'] = True
            return hv.Points(dft, [x, y], label="%s vs %s" % (x.title(), y.title()),
                title='Pair-wise Scatter Plot of two Independent Numeric variables').opts(**opts)

        widgets = pn.WidgetBox(x, y, color)

        hv_panel = pn.Row(widgets, create_figure).servable('Cross-selector')
        #colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
        #def load_symbol(symbol, variable, **kwargs):
        #    color = next(colors)
        #    return hv.Scatter((dft[symbol].values,dft[variable].values)).opts(framewise=True).opts(size=5,
        #            color=color, alpha=alpha, height=height_size, width=width_size).opts(
        #            xlabel='%s' %symbol).opts(ylabel='%s' %variable).opts(
        #           title='Scatter Plot of %s against %s variable' %(symbol,variable))
        #dmap = hv.DynamicMap(load_symbol, kdims=['Select_X','Select_Y']).redim.values(Select_X=nums, Select_Y=nums).opts(framewise=True)
        #hv_panel = pn.panel(dmap)
        #widgets = hv_panel[0]
        #hv_panel = pn.Column(pn.Row(*widgets))
        if verbose == 2:
            imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
            image_count += 1
    else:
        quantileable = [x for x in nums if len(dft[x].unique()) > 20]
        if len(quantileable) <= 1:
            quantileable = [x for x in nums if len(dft[x].unique()) > 2]

        x = pnw.Select(name='X-Axis', value=quantileable[0], options=quantileable)
        y = pnw.Select(name='Y-Axis', value=quantileable[1], options=quantileable)
        size = pnw.Select(name='Size', value='None', options=['None'] + quantileable)
        color = pnw.Select(name='Color', value='None', options=['None',dep])

        @pn.depends(x.param.value, y.param.value, color.param.value) 
        def create_figure(x, y, color):
            opts = dict(cmap=cmap_list[0], width=width_size, height=height_size, line_color='black')
            if color != 'None':
                opts['color'] = color 
            opts['size'] = bubble_size
            opts['alpha'] = alpha
            opts['tools'] = ['hover']
            opts['toolbar'] = 'above'
            opts['colorbar'] = True
            return hv.Points(dft, [x, y], label="%s vs %s" % (x.title(), y.title()),
                title='Pair-wise Scatter Plot of two Independent Numeric variables').opts(**opts)

        widgets = pn.WidgetBox(x, y, color)

        hv_panel = pn.Row(widgets, create_figure).servable('Cross-selector')
        #target_vars = dft[dep].unique()
        #x = pn.widgets.Select(name='x', options=nums)
        #y = pn.widgets.Select(name='y', options=nums)
        #kind = pn.widgets.Select(name='kind', value='scatter', options=['bivariate', 'scatter'])

        #plot = dft.hvplot(x=x, y=y, kind=kind, by=dep, height=height_size, alpha=0.5,
        #                title='Pair-wise Scatter Plot of two Independent Numeric variables')
        #hv_panel = pn.Row(pn.WidgetBox(x, y, kind), plot)
        if verbose == 2:
            imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
            image_count += 1
    if chart_format in ['server', 'bokeh_server', 'bokeh-server']:
        #server = pn.serve(hv_panel, start=True, show=True)
        print('%s can be found in URL below:' %plot_name)
        hv_panel.show()
    elif chart_format == 'html':
        save_html_data(hv_panel, chart_format, plot_name, mk_dir)
    else:
        display(hv_panel)  
    return hv_panel
def draw_distplot_hv(dft, cats, conti, chart_format,problem_type,dep=None, 
                    classes=None, mk_dir='AutoViz_Plots', verbose=0):
    dft = copy.deepcopy(dft)
    image_count = 0
    imgdata_list = list()
    
    conti = list(set(conti))
    nums = copy.deepcopy(conti)
    classes = copy.deepcopy(classes)
    colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
    imgdata_list = list()
    width_size = 600  
    height_size = 400
    gap = 0.4 
    plot_name = 'distplots'
    if dep==None or dep=='' or problem_type == 'Regression':
        transparent = 0.7
        binsize = 30
        for each_conti,k in zip(conti,range(len(conti))):
            if dft[each_conti].isnull().sum() > 0:
                dft[[each_conti]] = dft[[each_conti]].fillna(0)
        
        if not isinstance(dep, list):
            
            if dep == '':
                pass
            elif len(cats) > 0:
                colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
                def select_widget(each_cat):

                    width_size=15
                    conti_df = dft[[dep,each_cat]].groupby(each_cat).mean().reset_index()
                    row_ticks = dft[dep].unique().tolist()
                    color_list = next(colors)
                    pivotdf = pd.DataFrame(conti_df.to_records()).set_index(each_cat)
                    plot = pivotdf.hvplot(kind='bar',stacked=False,use_index=False, color=color_list,
                                          title='Mean Target = %s by each Categorical Variable' %dep).opts(xrotation=70)
                    return plot
                dmap = hv.DynamicMap(select_widget,  kdims=['Select_Cat_Variable']).redim.values(Select_Cat_Variable=cats)
                dmap.opts(framewise=True,axiswise=True) 
                hv_panel = pn.panel(dmap)
                widgets = hv_panel[0]
                hv_all = pn.Column(pn.Row(*widgets))
                if verbose == 2:
                    imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
                    image_count += 1
                if chart_format in ['server', 'bokeh_server', 'bokeh-server']:
                    #server = pn.serve(hv_all, start=True, show=True)
                    print('%s can be found in URL below:' %plot_name)
                    hv_all.show()
                elif chart_format == 'html':
                    save_html_data(hv_all, chart_format, plot_name, mk_dir, additional="_cats")
                else:
                    display(hv_all)  
                    #display_obj(hv_all)  
        if len(conti) > 0:
            try:
                jitter = 0.5
                colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
                transparent = 0.5
                def select_variable_to_plot(num_var):

                    color = next(colors)
                    xlimi = (dft[num_var].min(), dft[num_var].max())
                    hv_look = hv.Distribution(np.histogram(dft[num_var]), num_var).opts(color=color,
                                        height=height_size, width=width_size, alpha=transparent,
                                    title='KDE (Distribution) Plot of Numeric Variables').redim.range(num_var=xlimi)
                    return hv_look

                dmap = hv.DynamicMap(select_variable_to_plot,  kdims=['Select_Variable']).redim.values(Select_Variable=nums)
                dmap.opts(framewise=True,axiswise=True) 

                hv_panel = pn.panel(dmap)
                widgets = hv_panel[0]
                hv_all = pn.Column(pn.Row(*widgets))
            except:
                print('Error in Distribution Plot')
                hv_panel = []
        if verbose == 2:
            imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
            image_count += 1
        if chart_format in ['server', 'bokeh_server', 'bokeh-server']:
            #server = pn.serve(hv_all, start=True, show=True)
            print('%s can be found in URL below:' %plot_name)
            hv_all.show()
        elif chart_format == 'html':
            save_html_data(hv_all, chart_format, plot_name, mk_dir, additional="_nums")
        else:
            display(hv_all)  
            #display_obj(hv_all)  
    else:
        transparent = 0.7
        binsize = 30
        alpha = 0.5
        height_size = 400
        width_size = 600
        target_vars = dft[dep].unique().tolist()
        if type(classes[0])==int:
            classes = [str(x) for x in classes]
        for each_conti,k in zip(conti,range(len(conti))):
            if dft[each_conti].isnull().sum() > 0:
                dft[[each_conti]] = dft[[each_conti]].fillna(0)
        if len(cats) > 0:
            def select_widget(Select_categorical_var):

                colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')

                conti_df = dft[[dep,Select_categorical_var]].groupby([dep,Select_categorical_var]).size().nlargest(
                                    width_size).reset_index(name='Values')
                pivot_df = conti_df.pivot(index=Select_categorical_var, columns=dep, values='Values').fillna(0)
                row_ticks = dft[dep].unique().tolist()
                color_list = []
                for i in range(len(row_ticks)):
                    color_list.append(next(colors))
                pivotdf = pd.DataFrame(pivot_df.to_records()).set_index(Select_categorical_var)
                plot = pivotdf.hvplot(kind='bar',stacked=True,use_index=True,
                            title='Target = %s Histogram by each Categorical Variable' %dep).opts(
                                height=height_size,width=width_size, xrotation=70)
                return plot
            dmap = hv.DynamicMap(select_widget,  kdims=['Select_categorical_var']).redim.values(
                                                Select_categorical_var=cats)
            hv_panel = pn.panel(dmap)
            widgets = hv_panel[0]
            hv_all = pn.Column(pn.Row(*widgets))
            if verbose == 2:
                imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
                image_count += 1
            if chart_format in ['server', 'bokeh_server', 'bokeh-server']:
                #server = pn.serve(hv_all, start=True, show=True)
                print('%s can be found in URL below:' %plot_name)
                hv_all.show()
            elif chart_format == 'html':
                save_html_data(hv_all, chart_format, plot_name, mk_dir, additional="_cats")
            else:
                display(hv_all)  
                #display_obj(hv_all)
        if len(conti) > 0:
            try:
                colortext = 'brycgkbyrcmgkbyrcmgkbyrcmgkbyr'
                target_vars = dft[dep].unique().tolist()
                color_list = list(colortext[:len(target_vars)])
                jitter = 0.5
                colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
                transparent = 0.5
                def select_widget(Select_numeric_variable):

                    color = next(colors)
                    overlay = hv.NdOverlay({group: hv.Distribution(np.histogram(dft[dft[dep]==group][Select_numeric_variable].values)) for i,group in enumerate(target_vars)})
                    hv_look = overlay.opts(opts.Distribution(alpha=0.5, height=height_size, width=width_size)).opts(
                        title='KDE (Distribution) Plots of all Numeric Variables by Classes').opts(
                        xlabel='%s' %dep).opts(ylabel='%s' %Select_numeric_variable)
                    return hv_look
                dmap = hv.DynamicMap(select_widget,  kdims=['Select_numeric_variable']).redim.values(Select_numeric_variable=nums)

                hv_panel = pn.panel(dmap)
                widgets = hv_panel[0]
                hv_all = pn.Column(pn.Row(*widgets))
            except:
                print('Error in Distribution Plot')
            if verbose == 2:
                imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
                image_count += 1
            if chart_format in ['server', 'bokeh_server', 'bokeh-server']:
                #server = pn.serve(hv_all, start=True, show=True)
                print('%s can be found in URL below:' %plot_name)
                hv_all.show()
            elif chart_format == 'html':
                save_html_data(hv_all, chart_format, plot_name, mk_dir, additional="_nums")
            else:
                display(hv_all)  
                #display_obj(hv_all)  

    return hv_all

def draw_violinplot_hv(dft, dep, nums,chart_format, modeltype='Regression', 
                    mk_dir='AutoViz_Plots', verbose=0):
    dft = copy.deepcopy(dft)
    image_count = 0
    imgdata_list = list()
    width_size = 800
    height_size = 500
    if type(dep) == str:
        nums = [x for x in nums if x not in [dep]]
    else:
        nums = [x for x in nums if x not in dep]
    colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
    plot_name = 'violinplots'
    
    if modeltype in ['Regression', 'Clustering']:
        
        number_in_each_row = 30
        cmap_list = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
        
        df_p = dft[nums]
        if df_p.shape[1] < number_in_each_row:
            iter_limit = number_in_each_row
        else:
            iter_limit = max(number_in_each_row, int(df_p.shape[1]/5+0.5))
        #print('Current number of Numeric Variables = %d ' %(df_p.shape[1],))
        
        drawobjv_list = [] 
        drawobj_list = [] 
        counter = 0
        for i in range(0,df_p.shape[1],iter_limit):
            new_end = i+iter_limit
            #print('i = ',i,"new end = ", new_end)
            if i == 0:
                title_string = 'using first %d variables...' %(iter_limit)
                #print(title_string )
            else:
                title_string = 'using next %d variables...' %(iter_limit)
                #print(title_string )
            conti = nums[i:new_end]
            from sklearn.preprocessing import StandardScaler
            SS = StandardScaler()
            data = pd.DataFrame(SS.fit_transform(dft[conti]),columns=conti)
            var_name = 'drawobjv_list['+str(counter)+']'
            drawobj_list.append(var_name)
            drawobjv_list.append(var_name)
            drawobj = data.hvplot(kind='violin', label='Violin Plot %s (Standard Scaled)' %title_string,
                                   rot=70  #height=height_size,width=width_size
                                 )
            drawobjv_list[counter] = drawobj
            counter += 1
        combined_charts = "("+"".join([x+'+' for x in drawobj_list])[:-1]+")"
        hv_all = pn.panel(eval(combined_charts))        #### This is where we add them to the list ######        
        if verbose == 2:
            imgdata_list = append_panels(hv_all, imgdata_list, chart_format)
            image_count += 1

        if chart_format in ['server', 'bokeh_server', 'bokeh-server']:
            print('%s can be found in URL below:' %plot_name)
            server = pn.serve(hv_all, start=True, show=True)
            #hv_all.show()  
        elif chart_format == 'html':
            save_html_data(hv_all, chart_format, plot_name, mk_dir)
        else:
            display(hv_all)
            #display_obj(hv_all) 
    else:
        number_in_each_row = 30
        df_p = dft[nums]
        if df_p.shape[1] < number_in_each_row:
            iter_limit = number_in_each_row
        else:
            iter_limit = max(number_in_each_row, int(df_p.shape[1]/5+0.5))
        target_vars = np.unique(dft[dep])
        dmaps = []
        combined_charts = ''
        counter = 0
        drawobjv_list = [] 
        drawobj_list = [] 
        for symbol in target_vars:
            color = next(colors)
            sup_title = 'Violin Plot for %s = %s' %(dep,symbol)
            for i in range(0,df_p.shape[1],iter_limit):
                new_end = i+iter_limit
                if i == 0:
                    title_string = 'first %d variables' %(df_p.shape[1])
                else:
                    title_string = 'next %d variables' %(df_p.shape[1])
                conti = nums[i:new_end]
                from sklearn.preprocessing import StandardScaler
                SS = StandardScaler()
                data = pd.DataFrame(SS.fit_transform(dft[conti]),columns=conti)
                data[dep] = dft[dep].values
                dft_sym = data[data[dep] == symbol][conti]
                var_name = 'drawobjv_list['+str(counter)+']'
                drawobj_list.append(var_name)
                drawobjv_list.append(var_name)
                drawobj =  dft_sym.hvplot(kind='violin',title='%s: %s' %(sup_title, title_string)).opts(framewise=True).opts(
                        box_color=color, height=height_size, width=width_size)
                drawobjv_list[counter] = drawobj
                counter += 1
        combined_charts = "".join([x+'+' for x in drawobj_list])[:-1]
        hv_all = pn.panel(eval(combined_charts))
        if chart_format.lower() in ['server', 'bokeh_server', 'bokeh-server']:
            print('%s can be found in URL below:' %plot_name)
            server = pn.serve(hv_all, start=True, show=True)
        elif chart_format == 'html':
            save_html_data(hv_all, chart_format, plot_name, mk_dir)
        else:
            display(hv_all)
        if verbose == 2:
            imgdata_list = append_panels(hv_all, imgdata_list, chart_format)
            image_count += 1
    return hv_all
def draw_date_vars_hv(df,dep,datevars, nums, chart_format, modeltype='Regression',
                        mk_dir='AutoViz_Plots', verbose=0):
    df = copy.deepcopy(df)
    imgdata_list = list()
    image_count = 0
    N = len(nums)
    cols = 2
    width_size = 600
    height_size = 400
    jitter = 0.05
    alpha = 0.5
    size = 5
    transparent = 0.5
    colortext = 'brycgkbyrcmgkbyrcmgkbyrcmgkbyr'
    colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
    plot_name = 'timeseries_plots'

    if modeltype == 'Regression':
        if isinstance(dep, str):
            if dep not in nums:
                nums.append(dep)
        else:
            nums += dep
            nums = find_remove_duplicates(nums)
    else:
            nums = find_remove_duplicates(nums)
    quantileable = nums[:]
    cmap_list = ['Blues','rainbow', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']

    x = pnw.Select(name='X-Axis', value=datevars[0], options=datevars)
    y = pnw.Select(name='Y-Axis', value=quantileable[0], options=quantileable)

    @pn.depends(x.param.value, y.param.value) 
    def create_figure(x, y):
        opts = dict(cmap=cmap_list[0], line_color='black')
        #opts['size'] = bubble_size
        opts['alpha'] = alpha
        opts['tools'] = ['hover']
        opts['toolbar'] = 'above'
        opts['colorbar'] = True
        dft = df.set_index(df[x])
        conti_df = df[[x,y]].set_index(df[x]).drop(x, axis=1)
        return hv.Curve(conti_df).opts(
            line_width=1, line_color=next(colors),line_dash='dotted', line_alpha=0.5).opts(
            width=width_size, height=height_size,title='Time Series plots of Numeric vars')

    widgets = pn.WidgetBox(x, y)

    hv_panel = pn.Row(widgets, create_figure).servable('Cross-selector')    
    
    if chart_format in ['server', 'bokeh_server', 'bokeh-server']:
        #server = pn.serve(hv_panel, start=True, show=True)
        print('%s can be found in URL below:' %plot_name)
        hv_panel.show()
    elif chart_format == 'html':
        save_html_data(hv_panel, chart_format, plot_name, mk_dir)
    else:
        display(hv_panel)  
    return hv_panel

def draw_heatmap_hv(dft, conti, chart_format, datevars=[], dep=None,
                            modeltype='Regression',classes=None, mk_dir='AutoViz_Plots', verbose=0):
    dft = copy.deepcopy(dft)
    width_size = 600
    height_size = 400
    cmap_list = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    if len(conti) <= 1:
        return
    elif len(conti) <= 10:
        height_size = 500
        width_size = 600
    else:
        height_size = 800
        width_size = 1200
    plot_name = 'heatmaps'
    if isinstance(dft.index, pd.DatetimeIndex) :
        dft = dft[:]
        timeseries_flag = True
        pass
    else:
        dft = dft[:]
        try:
            dft.index = pd.to_datetime(dft.pop(datevars[0]),infer_datetime_format=True)
            timeseries_flag = True
        except:
            if verbose == 1 and len(datevars) > 0:
                print('No date vars could be found or %s could not be indexed.' %datevars)
            elif verbose == 1 and len(datevars) == 0:
                print('No date vars could be found in data set')
            timeseries_flag = False
    imgdata_list = list()
    if not modeltype in ['Regression','Clustering']:
        if dft[dep].dtype == object or dft[dep].dtype == np.int64:
            dft[dep] = dft[dep].factorize()[0]
        image_count = 0
        N = len(conti)
        target_vars = dft[dep].unique()
        plotc = 1
        #rows = len(target_vars)
        rows = 1
        cols = 1
        if timeseries_flag:
            dft_target = dft[[dep]+conti].diff()
        else:
            dft_target = dft[:]
        dft_target[dep] = dft[dep].values
        corre = dft_target.corr()
        if timeseries_flag:
            heatmap = corre.hvplot.heatmap(height=height_size, width=width_size, colorbar=True, 
                    cmap=cmap_list, rot=70,
            title='Time Series: Heatmap of all Differenced Continuous vars for target = %s' %dep)
        else:
            heatmap = corre.hvplot.heatmap(height=height_size, width=width_size,  colorbar=True,
                    cmap=cmap_list,
                    rot=70,
            title='Heatmap of all Continuous Variables including target');
        hv_plot = heatmap * hv.Labels(heatmap).opts(opts.Labels(text_font_size='7pt'))
        hv_panel = pn.panel(hv_plot)
        if verbose == 2:
            imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
            image_count += 1
    else:
        image_count = 0
        if dep is None or dep == '':
            pass
        else:
            conti += [dep]
        dft_target = dft[conti]
        if timeseries_flag:
            dft_target = dft_target.diff().dropna()
        else:
            dft_target = dft_target[:]
        N = len(conti)
        corre = dft_target.corr()
        if timeseries_flag:
            heatmap = corre.hvplot.heatmap(height=height_size, width=width_size, colorbar=True, 
                    cmap=cmap_list,
                                           rot=70,
                title='Time Series Data: Heatmap of Differenced Continuous vars including target').opts(
                        opts.HeatMap(tools=['hover'], toolbar='above'))
        else:
            heatmap = corre.hvplot.heatmap(height=height_size, width=width_size, colorbar=True, 
                    cmap=cmap_list,
                                           rot=70,
            title='Heatmap of all Continuous Variables including target').opts(
                                    opts.HeatMap(tools=['hover'],  toolbar='above'))
        hv_plot = heatmap * hv.Labels(heatmap).opts(opts.Labels(text_font_size='7pt'))
        hv_panel = pn.panel(hv_plot)
        if verbose == 2:
            imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
            image_count += 1
    if chart_format in ['server', 'bokeh_server', 'bokeh-server']:
        print('%s can be found in URL below:' %plot_name)
        server = pn.serve(hv_panel, start=True, show=True)
    elif chart_format == 'html':
        save_html_data(hv_panel, chart_format, plot_name, mk_dir)
    else:
        display(hv_panel)
        #display_obj(hv_panel)
    return hv_panel

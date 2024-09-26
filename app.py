# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 23:54:24 2024

@author: UOBASUB
"""

import csv
import io
from datetime import datetime, timedelta, date, time
import pandas as pd 
import numpy as np
import math
from google.cloud.storage import Blob
from google.cloud import storage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
np.seterr(divide='ignore', invalid='ignore')
pd.options.mode.chained_assignment = None
from scipy.signal import argrelextrema
from scipy import signal
from scipy.misc import derivative
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import plotly.io as pio
pio.renderers.default='browser'
import bisect
#import yfinance as yf
#import dateutil.parser


def find_spikes(data, high_percentile=97, low_percentile=3):
    # Compute the high and low thresholds
    high_threshold = np.percentile(data, high_percentile)
    low_threshold = np.percentile(data, low_percentile)
    
    # Find and collect spikes
    spikes = {'high_spikes': [], 'low_spikes': []}
    for index, value in enumerate(data):
        if value > high_threshold:
            spikes['high_spikes'].append((index, value))
        elif value < low_threshold:
            spikes['low_spikes'].append((index, value))
    
    return spikes

symbolNumList = ['183748', '106364', '42006053', '230943', '393','163699', '923', '42018437']
symbolNameList = ['ES', 'NQ', 'YM','CL', 'GC', 'HG', 'NG', 'RTY']

intList = ['1','2','3','4','5','6','10','15']

vaildClust = [str(i) for i in range(3,20)]

vaildTPO = [str(i) for i in range(10,500)]

gclient = storage.Client(project="stockapp-401615")
bucket = gclient.get_bucket("stockapp-storage")

styles = {
    'main_container': {
        'display': 'flex',
        'flexDirection': 'row',  # Align items in a row
        'justifyContent': 'space-around',  # Space between items
        'flexWrap': 'wrap',  # Wrap items if screen is too small
        #'marginTop': '20px',
        'background': '#E5ECF6',  # Soft light blue background
        'padding': '20px',
        #'borderRadius': '10px'  # Optional: adds rounded corners for better aesthetics
    },
    'sub_container': {
        'display': 'flex',
        'flexDirection': 'column',  # Align items in a column within each sub container
        'alignItems': 'center',
        'margin': '10px'
    },
    'input': {
        'width': '150px',
        'height': '35px',
        'marginBottom': '10px',
        'borderRadius': '5px',
        'border': '1px solid #ddd',
        'padding': '0 10px'
    },
    'button': {
        'width': '100px',
        'height': '35px',
        'borderRadius': '10px',
        'border': 'none',
        'color': 'white',
        'background': '#333333',  # Changed to a darker blue color
        'cursor': 'pointer'
    },
    'label': {
        'textAlign': 'center'
    }
}


#import pandas_ta as ta
#from collections import Counter
from google.api_core.exceptions import NotFound
from dash import Dash, dcc, html, Input, Output, callback, State
initial_inter = 60*1000  # 60 seconds
subsequent_inter = 15*1000  # 30 seconds
app = Dash()
app.title = "Initial Title"
app.layout = html.Div([
    
    dcc.Graph(id='table'),
    dcc.Interval(
        id='interval',
        interval=initial_inter,
        n_intervals=0,
      ),
    html.Div([
        html.Div([
            dcc.Input(id='input-on-submit', type='text', style=styles['input']),
            html.Button('Submit', id='submit-val', n_clicks=0, style=styles['button']),
            html.Div(id='container-button-basic', children="Enter a symbol from 'ES', 'NQ', 'YM', 'CL', 'GC', 'HG', 'NG', 'RTY'", style=styles['label']),
        ], style=styles['sub_container']),
        dcc.Store(id='stkName-value'),
        
        html.Div([
            dcc.Input(id='input-on-interv', type='text', style=styles['input']),
            html.Button('Submit', id='submit-interv', n_clicks=0, style=styles['button']),
            html.Div(id='interv-button-basic',children="Enter interval from 5, 10, 15, 30", style=styles['label']),
        ], style=styles['sub_container']),
        dcc.Store(id='interv-value'),
        
        html.Div([
            dcc.Input(id='input-on-tpo', type='text', style=styles['input']),
            html.Button('Submit', id='submit-tpo', n_clicks=0, style=styles['button']),
            html.Div(id='tpo-button-basic', children="Enter a top ranked order number from 10 - 500", style=styles['label']),
        ], style=styles['sub_container']),
        dcc.Store(id='tpo-value'),
    ], style=styles['main_container']),
    
    
    dcc.Store(id='data-store'),
    dcc.Store(id='previous-interv'),
    dcc.Store(id='previous-stkName'),
    dcc.Store(id='interval-time', data=initial_inter),
  
])

@callback(
    Output('stkName-value', 'data'),
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('input-on-submit', 'value'),
    prevent_initial_call=True
)

def update_output(n_clicks, value):
    value = str(value).upper().strip()
    
    if value in symbolNameList:
        print('The input symbol was "{}" '.format(value))
        return str(value).upper(), str(value).upper()
    else:
        return 'The input symbol '+str(value)+" is not accepted please try different symbol from  |'ES', 'NQ',  'YM',  'BTC', 'CL', 'GC'|", 'The input symbol was '+str(value)+" is not accepted please try different symbol  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  "

@callback(
    Output('interv-value', 'data'),
    Output('interv-button-basic', 'children'),
    Input('submit-interv', 'n_clicks'),
    State('input-on-interv', 'value'),
    prevent_initial_call=True
)
def update_interval(n_clicks, value):
    value = str(value)
    
    if value in intList:
        print('The input interval was "{}" '.format(value))
        return str(value), str(value), 
    else:
        return 'The input interval '+str(value)+" is not accepted please try different interval from  |'1' '2' '3' '5' '10' '15'|", 'The input interval '+str(value)+" is not accepted please try different interval from  |'1' '2' '3' '5' '10' '15'|"



@callback(
    Output('tpo-value', 'data'),
    Output('tpo-button-basic', 'children'),
    Input('submit-tpo', 'n_clicks'),
    State('input-on-tpo', 'value'),
    prevent_initial_call=True
)
def update_tpo(n_clicks, value):
    value = str(value)
    
    if value in vaildTPO:
        print('The input top rank order was "{}" '.format(value))
        return str(value), str(value), 
    else:
        return 'The input top rank order was '+str(value)+" is not accepted please try different number from  10 - 500", 'The input top rank order '+str(value)+" is not accepted please try different number from  10 - 500"




@callback(
    [Output('data-store', 'data'),
        Output('table', 'figure'),
        Output('previous-stkName', 'data'),
        Output('previous-interv', 'data'),
        Output('interval', 'interval')],
    [Input('interval', 'n_intervals')],
    [State('stkName-value', 'data'),
        State('interv-value', 'data'),
        State('data-store', 'data'),
        State('previous-stkName', 'data'),
        State('previous-interv', 'data'),
        State('tpo-value', 'data'),
        State('interval-time', 'data'),
    ],
)
    
def update_graph_live(n_intervals, sname, interv, stored_data, previous_stkName, previous_interv, tpoNum, interval_time): #interv
    
    #print(sname, interv, stored_data, previous_stkName)
    #print(interv)

    if sname in symbolNameList:
        stkName = sname
        symbolNum = symbolNumList[symbolNameList.index(stkName)]   
    else:
        stkName = 'NQ' 
        sname = 'NQ'
        symbolNum = symbolNumList[symbolNameList.index(stkName)]
        
    if interv not in intList:
        interv = '3'
        
    if stkName != previous_stkName or interv != previous_interv:
        stored_data = None


    if tpoNum not in vaildTPO:
        tpoNum = '200'
        
        
    print('inFunction '+sname)	
    
    blob = Blob('FuturesOHLC'+str(symbolNum), bucket) 
    FuturesOHLC = blob.download_as_text()
        

    csv_reader  = csv.reader(io.StringIO(FuturesOHLC))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    
    
    aggs = [ ]  
    newOHLC = [i for i in csv_rows]

    for i in newOHLC:
        hourss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).hour
        if hourss < 10:
            hourss = '0'+str(hourss)
        minss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).minute
        if minss < 10:
            minss = '0'+str(minss)
        opttimeStamp = str(hourss) + ':' + str(minss) + ':00'
        aggs.append([int(i[2])/1e9, int(i[3])/1e9, int(i[4])/1e9, int(i[5])/1e9, int(i[6]), opttimeStamp, int(i[0]), int(i[1])])
        
            
    newAggs = []
    for i in aggs:
        if i not in newAggs:
            newAggs.append(i)
    
    
            
       
    df = pd.DataFrame(newAggs, columns = ['open', 'high', 'low', 'close', 'volume', 'time', 'timestamp', 'name',])
    
    df['strTime'] = df['timestamp'].apply(lambda x: pd.Timestamp(int(x) // 10**9, unit='s', tz='EST') )
    
    df.set_index('strTime', inplace=True)
    df['volume'] = pd.to_numeric(df['volume'], downcast='integer')
    df_resampled = df.resample(interv+'min').agg({
        'timestamp': 'first',
        'name': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'time': 'first',
        'volume': 'sum'
    })
    
    df_resampled.reset_index(drop=True, inplace=True)
    
    df = df_resampled
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True) 

    


    
    blob = Blob('FuturesTrades'+str(symbolNum), bucket) 
    FuturesTrades = blob.download_as_text()
    
    
    csv_reader  = csv.reader(io.StringIO(FuturesTrades))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
       

    #STrades = [i for i in csv_rows]
    AllTrades = []
    for i in csv_rows:
        hourss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).hour
        if hourss < 10:
            hourss = '0'+str(hourss)
        minss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).minute
        if minss < 10:
            minss = '0'+str(minss)
        opttimeStamp = str(hourss) + ':' + str(minss) + ':00'
        AllTrades.append([int(i[1])/1e9, int(i[2]), int(i[0]), 0, i[3], opttimeStamp])
       

    mTrade = [i for i in AllTrades ]
    
     
    mTrade = sorted(mTrade, key=lambda d: d[1], reverse=True)
    
    [mTrade[i].insert(4,i) for i in range(len(mTrade))] 
    
    newwT = []
    for i in mTrade:
        newwT.append([i[0],i[1],i[2],i[5], i[4],i[3],i[6]])
    
    
    dtime = df['time'].dropna().values.tolist()
    dtimeEpoch = df['timestamp'].dropna().values.tolist()
    
    
    tempTrades = [i for i in AllTrades]
    tempTrades = sorted(tempTrades, key=lambda d: d[6], reverse=False) 
    tradeTimes = [i[6] for i in AllTrades]
    tradeEpoch = [i[2] for i in AllTrades]
    
    
    if stored_data is not None:
        print('NotNew')
        startIndex = next(iter(df.index[df['time'] == stored_data['timeFrame'][len(stored_data['timeFrame'])-1][0]]), None)#df['timestamp'].searchsorted(stored_data['timeFrame'][len(stored_data['timeFrame'])-1][9])
        timeDict = {}
        make = []
        for ttm in range(startIndex,len(dtimeEpoch)):
            
            make.append([dtimeEpoch[ttm],dtime[ttm],bisect.bisect_left(tradeEpoch, dtimeEpoch[ttm])])
            timeDict[dtime[ttm]] = [0,0,0]
            
        for tr in range(len(make)):
            if tr+1 < len(make):
                tempList = AllTrades[make[tr][2]:make[tr+1][2]]
            else:
                tempList = AllTrades[make[tr][2]:len(AllTrades)]
            for i in tempList:
                if i[5] == 'B':
                    timeDict[make[tr][1]][0] += i[1]
                elif i[5] == 'A':
                    timeDict[make[tr][1]][1] += i[1] 
                elif i[5] == 'N':
                    timeDict[make[tr][1]][2] += i[1]
            try:    
                timeDict[make[tr][1]] += [timeDict[make[tr][1]][0]/sum(timeDict[make[tr][1]]), timeDict[make[tr][1]][1]/sum(timeDict[make[tr][1]]), timeDict[make[tr][1]][2]/sum(timeDict[make[tr][1]])]   
            except(ZeroDivisionError):
                timeDict[make[tr][1]]  += [0,0,0] 
                
        timeFrame = [[i,'']+timeDict[i] for i in timeDict]
    
        for i in range(len(timeFrame)):
            timeFrame[i].append(dtimeEpoch[startIndex+i])
            
        for pott in timeFrame:
            #print(pott)
            pott.insert(4,df['timestamp'].searchsorted(pott[8]))
            
            
        stored_data['timeFrame'] = stored_data['timeFrame'][:len(stored_data['timeFrame'])-1] + timeFrame
        
        bful = []
        for it in range(len(make)):
            if it+1 < len(make):
                tempList = AllTrades[0:make[it+1][2]]
            else:
                tempList = AllTrades
            #print(make[0][2],make[it+1][2], len(tempList))
            nelist = sorted(tempList, key=lambda d: d[1], reverse=True)[:200]
                        
            bful.append([make[it][1], sum([i[1] for i in nelist if i[5] == 'B']), sum([i[1] for i in nelist if i[5] == 'A'])])
  
        
        dst = [[bful[row][0], bful[row][1], 0, bful[row][2], 0] for row in  range(len(bful))]
        
        stored_data['tro'] = stored_data['tro'][:len(stored_data['tro'])-1] + dst
        
        bolist = [0]
        for i in range(len(stored_data['tro'])-1):
            bolist.append(stored_data['tro'][i+1][1] - stored_data['tro'][i][1])
            
        solist = [0] 
        for i in range(len(stored_data['tro'])-1):
            solist.append(stored_data['tro'][i+1][3] - stored_data['tro'][i][3])
            
        newst = [[stored_data['tro'][i][0], stored_data['tro'][i][1], bolist[i], stored_data['tro'][i][3], solist[i] ] for i in range(len(stored_data['tro']))]
        
        stored_data['tro'] = newst
            
    
    
    if stored_data is None:
        print('Newstored')
        timeDict = {}
        make = []
        for ttm in range(len(dtimeEpoch)):
            
            make.append([dtimeEpoch[ttm],dtime[ttm],bisect.bisect_left(tradeEpoch, dtimeEpoch[ttm])]) #min(range(len(tradeEpoch)), key=lambda i: abs(tradeEpoch[i] - dtimeEpoch[ttm]))
            timeDict[dtime[ttm]] = [0,0,0]
            
            
        
        for tr in range(len(make)):
            if tr+1 < len(make):
                tempList = AllTrades[make[tr][2]:make[tr+1][2]]
            else:
                tempList = AllTrades[make[tr][2]:len(AllTrades)]
            for i in tempList:
                if i[5] == 'B':
                    timeDict[make[tr][1]][0] += i[1]
                elif i[5] == 'A':
                    timeDict[make[tr][1]][1] += i[1] 
                elif i[5] == 'N':
                    timeDict[make[tr][1]][2] += i[1]
            try:    
                timeDict[make[tr][1]] += [timeDict[make[tr][1]][0]/sum(timeDict[make[tr][1]]), timeDict[make[tr][1]][1]/sum(timeDict[make[tr][1]]), timeDict[make[tr][1]][2]/sum(timeDict[make[tr][1]])]   
            except(ZeroDivisionError):
                timeDict[make[tr][1]]  += [0,0,0] 
    
                          
        timeFrame = [[i,'']+timeDict[i] for i in timeDict]
    
        for i in range(len(timeFrame)):
            timeFrame[i].append(dtimeEpoch[i])
            
        for pott in timeFrame:
            #print(pott)
            pott.insert(4,df['timestamp'].searchsorted(pott[8]))
            
        
        bful = []
        for it in range(len(make)):
            if it+1 < len(make):
                tempList = AllTrades[0:make[it+1][2]]
            else:
                tempList = AllTrades
            nelist = sorted(tempList, key=lambda d: d[1], reverse=True)[:200]
                        
            bful.append([make[it][1], sum([i[1] for i in nelist if i[5] == 'B']), sum([i[1] for i in nelist if i[5] == 'A'])])
            
        bolist = [0]
        for i in range(len(bful)-1):
            bolist.append(bful[i+1][1] - bful[i][1])
            
        solist = [0]
        for i in range(len(bful)-1):
            solist.append(bful[i+1][2] - bful[i][2])
            #buyse/sellle
            
        
        dst = [[bful[row][0], bful[row][1], bolist[row], bful[row][2], solist[row]] for row in  range(len(bful))]
            
        stored_data = {'timeFrame': timeFrame, 'tro':dst} 
        

     
    previous_stkName = sname
    previous_interv = interv

        
    if interval_time == initial_inter:
        interval_time = subsequent_inter
    
    if stkName != previous_stkName or interv != previous_interv:
        interval_time = initial_inter
        
    fig = go.Figure()
    
    transposed_data = list(zip(*stored_data['tro'][::-1]))
    default_color = "#EBF0F8"  # Default color for all cells
    defaultTextColor = 'black'
    #special_color = "#FFD700"  # Gold color for the highlighted cell
    
    buysideSpikes = find_spikes([i[2] for i in stored_data['tro'][::-1]])
    sellsideSpikes = find_spikes([i[4] for i in stored_data['tro'][::-1]])
    
    # Create a color matrix for the cells
    color_matrix = [[default_color for _ in range(len(transposed_data[0]))] for _ in range(len(transposed_data))]
    textColor_matrix = [[defaultTextColor for _ in range(len(transposed_data[0]))] for _ in range(len(transposed_data))]
    
    for b in buysideSpikes['high_spikes']:
        color_matrix[2][b[0]] = 'teal'
        #textColor_matrix[2][b[0]] = 'white'
    for b in buysideSpikes['low_spikes']:
        color_matrix[2][b[0]] = 'crimson'
        #textColor_matrix[2][b[0]] = 'white'

    for b in sellsideSpikes['high_spikes']:
        color_matrix[4][b[0]] = 'crimson'
        #textColor_matrix[4][b[0]] = 'white'
    for b in sellsideSpikes['low_spikes']:
        color_matrix[4][b[0]] = 'teal'
        #textColor_matrix[4][b[0]] = 'white'
    
    fig.add_trace(
        go.Table(
            header=dict(values=["Time", "Buyers", "Buyers Change", "Sellers", "Sellers Change",]),
            cells=dict(values=transposed_data, fill_color=color_matrix, font=dict(color=textColor_matrix)),  # Transpose data to fit the table
        ),
    )
    
    fig.update_layout(title=stkName + '  '+ str(datetime.now().time()),height=800, xaxis_rangeslider_visible=False, showlegend=False)


    return stored_data, fig, previous_stkName, previous_interv, interval_time

#[(i[2]-i[3],i[0]) for i in timeFrame ]
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False)

'''
import time  
start_time = time.time()


end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
'''
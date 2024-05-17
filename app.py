# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:53:32 2024

@author: UOBASUB
"""

from google.cloud.storage import Blob
from google.cloud import storage
import csv
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date, time
import pandas as pd 
import numpy as np
import plotly.io as pio
pio.renderers.default='browser'
import bisect
from collections import Counter

symbolNumList = ['5602', '13743', '669', '80420', '2552',  '1256', '320510', '42009544']
symbolNameList = ['ES','NQ', 'GC',  'YM', 'RTY',  'PL', 'CL', 'BTC' ]

intList = ['1','2','3','4','5','6','10','15']

gclient = storage.Client(project="stockapp-401615")
bucket = gclient.get_bucket("stockapp-storage")

#import pandas_ta as ta
from dash import Dash, dcc, html, Input, Output, callback, State
inter = 22000 #250000#80001
app = Dash()
app.layout = html.Div([
    
    dcc.Graph(id='graph'),
    dcc.Interval(
        id='interval',
        interval=inter,
        n_intervals=0,
      ),

    html.Div(dcc.Input(id='input-on-submit', type='text')),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='container-button-basic',children="Enter a symbol from |'ES' 'NQ' 'GC' 'HG' 'YM' 'RTY' 'SI' 'CL' 'NG' | and submit"),
    dcc.Store(id='stkName-value'),
    
    html.Div(dcc.Input(id='input-on-interv', type='text')),
    html.Button('Submit', id='submit-interv', n_clicks=0),
    html.Div(id='interv-button-basic',children="Enter a symbol from |5 10 15 30 | and submit"),
    dcc.Store(id='interv-value'),
    
    
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
        return 'The input symbol '+str(value)+" is not accepted please try different symbol from  |'ES' 'NQ' 'GC' 'HG' 'YM' 'RTY' 'SI' 'CL' 'NG'|", 'The input symbol was '+str(value)+" is not accepted please try different symbol  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  "

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


@callback(Output('graph', 'figure'),
          Input('interval', 'n_intervals'),
          State('stkName-value', 'data'),
          State('interv-value', 'data'))

    
def update_graph_live(n_intervals, data, interv): #interv
    print('inFunction')	
    #print(interv)

    if data in symbolNameList:
        stkName = data
        symbolNum = symbolNumList[symbolNameList.index(stkName)]   
    else:
        stkName = 'NQ'  
        symbolNum = symbolNumList[symbolNameList.index(stkName)]
        
    if interv not in intList:
        interv = '5'
         
   
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
    df_resampled = df.resample(interv+'T').agg({
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
    tradeTimes = [i[6] for i in tempTrades]
    
    timeDict = {}
    cdDict = {}
    for ttm in dtime:
        for tradMade in tempTrades[bisect.bisect_left(tradeTimes, ttm):]:
            if datetime.strptime(tradMade[6], "%H:%M:%S") > datetime.strptime(ttm, "%H:%M:%S") + timedelta(minutes=int(interv)):
                try:
                    timeDict[ttm] += [timeDict[ttm][0]/sum(timeDict[ttm]), timeDict[ttm][1]/sum(timeDict[ttm]), timeDict[ttm][2]/sum(timeDict[ttm])]
                except(KeyError,ZeroDivisionError):
                    timeDict[ttm] = [0,0,0]
                break
            
            if ttm not in timeDict:
                timeDict[ttm] = [0,0,0]
            if ttm in timeDict:
                if tradMade[5] == 'B':
                    timeDict[ttm][0] += tradMade[1]#tradMade[0] * tradMade[1]
                elif tradMade[5] == 'A':
                    timeDict[ttm][1] += tradMade[1]#tradMade[0] * tradMade[1] 
                elif tradMade[5] == 'N':
                    timeDict[ttm][2] += tradMade[1]#tradMade[0] * tradMade[1] 
                    
            if ttm not in cdDict:
                cdDict[ttm] = []   
            if ttm in cdDict:
                cdDict[ttm].append([tradMade[0], tradMade[1], tradMade[5]])
            
               
                

    for i in timeDict:
        if len(timeDict[i]) == 3:
            try:
                timeDict[i] += [timeDict[i][0]/sum(timeDict[i]), timeDict[i][1]/sum(timeDict[i]), timeDict[i][2]/sum(timeDict[i])]#
            except(ZeroDivisionError,KeyError):
                timeDict[i] += [0, 0,0]
                
    
    
    dcount = {}
    for cd in cdDict:
        count_dict = Counter([i[0] for i in cdDict[cd]])
        sorted_data = sorted(cdDict[cd], key=lambda x: x[0])
        for i in count_dict:
            for x in sorted_data:
                if i == x[0]:
                    if cd not in dcount:
                        dcount[cd] = {}
                    if cd in dcount:
                        if i not in dcount[cd]:
                            dcount[cd][i] = [0,0,0,0]
                        if i in dcount[cd]:
                            if x[2] == 'B':
                                dcount[cd][i][0] += x[1]
                                #dcount[cd][i][2] = round(dcount[cd][i][0]/(dcount[cd][i][0]+dcount[cd][i][1]),2)
                                dcount[cd][i][2] = dcount[cd][i][0] - dcount[cd][i][1]
                                dcount[cd][i][3] = dcount[cd][i][0] + dcount[cd][i][1]
                            if x[2] == 'A':
                                dcount[cd][i][1] += x[1]
                                #dcount[cd][i][3] = round(dcount[cd][i][1]/(dcount[cd][i][0]+dcount[cd][i][1]),2)
                                dcount[cd][i][2] = dcount[cd][i][0] - dcount[cd][i][1]
                                dcount[cd][i][3] = dcount[cd][i][0] + dcount[cd][i][1]
                if x[0] > i:
                    break
                                
    timeFrame = [[i,'']+timeDict[i] for i in timeDict]

    for i in range(len(timeFrame)):
        timeFrame[i].append(dtimeEpoch[i])
        
    
    for i in timeFrame:
        try:
            i.append(dict(sorted(dcount[i[0]].items(), reverse=True)))
        except(KeyError):
            dcount[i[0]] = {}
            i.append(dcount[i[0]])
            
    
    fig = go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=False,
                        specs=[[{}],
                               [{}]],
                        horizontal_spacing=0.02, vertical_spacing=0.03,
                         row_width=[0.50, 0.50,] ) #,row_width=[0.30, 0.70,] column_widths=[0.80,0.20],

        
    
    OptionTimeFrame = timeFrame        
    for pott in OptionTimeFrame:
        pott.insert(4,df['timestamp'].searchsorted(pott[8]))
        
    optColor = [     'teal' if float(i[2]) > float(i[3]) #rgba(0,128,0,1.0)
                else 'crimson' if float(i[3]) > float(i[2])#rgba(255,0,0,1.0)
                else 'rgba(128,128,128,1.0)' if float(i[3]) == float(i[2])
                else i for i in OptionTimeFrame]

    fig.add_trace(
        go.Bar(
            x=pd.Series([i[0] for i in OptionTimeFrame]),
            y=pd.Series([float(i[2]) if float(i[2]) > float(i[3]) else float(i[3]) if float(i[3]) > float(i[2]) else float(i[2]) for i in OptionTimeFrame]),
            #textposition='auto',
            #orientation='h',
            #width=0.2,
            marker_color=optColor,
            hovertext=pd.Series([i[0]+' '+i[1] for i in OptionTimeFrame]),
            
        ),
         row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=pd.Series([i[0] for i in OptionTimeFrame]),
            y=pd.Series([float(i[3]) if float(i[2]) > float(i[3]) else float(i[2]) if float(i[3]) > float(i[2]) else float(i[3]) for i in OptionTimeFrame]),
            #textposition='auto',
            #orientation='h',
            #width=0.2,
            marker_color= [  'crimson' if float(i[2]) > float(i[3]) #rgba(255,0,0,1.0)
                        else 'teal' if float(i[3]) > float(i[2]) #rgba(0,128,0,1.0)
                        else 'rgba(128,128,128,1.0)' if float(i[3]) == float(i[2])
                        else i for i in OptionTimeFrame],
            hovertext=pd.Series([i[0]+' '+i[1] for i in OptionTimeFrame]),
            
        ),
        row=2, col=1
    )

    bms = pd.Series([i[2] for i in OptionTimeFrame]).rolling(3).mean()
    sms = pd.Series([i[3] for i in OptionTimeFrame]).rolling(3).mean()
    #xms = pd.Series([i[3]+i[2] for i in OptionTimeFrame]).rolling(4).mean()
    fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=bms, line=dict(color='teal'), mode='lines', name='Buy VMA'), row=2, col=1)
    fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=sms, line=dict(color='crimson'), mode='lines', name='Sell VMA'), row=2, col=1)
    
    
    difList = [(i[2]-i[3],i[0]) for i in OptionTimeFrame]
    coll = [     'teal' if i[0] > 0
                else 'crimson' if i[0] < 0
                else 'gray' for i in difList]
    fig.add_trace(go.Bar(x=pd.Series([i[1] for i in difList]), y=pd.Series([i[0] for i in difList]), marker_color=coll), row=1, col=1)
    
    posti = sum([i[0] for i in difList if i[0] > 0])/len([i[0] for i in difList if i[0] > 0])
    negati = sum([i[0] for i in difList if i[0] < 0])/len([i[0] for i in difList if i[0] < 0])

    fig.add_trace(go.Scatter(x=df['time'],
                             y= [posti]*len(df['time']) ,
                             line_color='teal',
                             text = str(posti),
                             textposition="bottom left",
                             name=str(posti),
                             showlegend=False,
                             mode= 'lines',
                            ),
                    row=1, col=1
                 )

    fig.add_trace(go.Scatter(x=df['time'],
                             y= [negati]*len(df['time']) ,
                             line_color='crimson',
                             text = str(negati),
                             textposition="bottom left",
                             name=str(negati),
                             showlegend=False,
                             mode= 'lines',
                            ),
                    row=1, col=1
                 )
    #fig.add_hline(y=posti, row=1, col=1)
    #fig.add_hline(y=negati, row=1, col=1)

    fig.update_layout(title=stkName + str(datetime.now().time()),height=800, xaxis_rangeslider_visible=False, showlegend=False)
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    #fig.show()
    #print("The time difference is :", timeit.default_timer() - starttime)

    return fig


if __name__ == '__main__': 
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False)
    
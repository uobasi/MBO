# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:56:12 2024

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
import timeit

    
FutureMBOSymbolList = ['ESH4','NQH4','CLH4', 'GCG4', 'NGG4', 'HGH4', 'YMH4', 'BTCZ3', 'RTYH4']
FutureMBOSymbolNumList = ['17077', '750', '463194', '41512', '56065', '31863', '204839', '75685', '7062', ]

#currMBOSymbolList = ['6AH4','6BH4','6CH4', '6EH4', '6JH4', '6SH4', '6NH4']
#currMBOSymbolNumList =  ['156755', '156618', '1545', '156627', '156657', '156650', '2259',]

#stkName = 'NQH4'  

from dash import Dash, dcc, html, Input, Output, callback, State
inter = 5000
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
    html.Div(id='container-button-basic',children="Enter a symbol from |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'| and submit"),
    dcc.Store(id='stkName-value')
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
    
    if value in FutureMBOSymbolList:
        print('The input symbol was "{}" '.format(value))
        return str(value).upper(), str(value).upper()
    
    else:
        return 'The input symbol was '+str(value)+" is not accepted please try different symbol from  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  ", 'The input symbol was '+str(value)+" is not accepted please try different symbol  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  "

@callback(Output('graph', 'figure'),
          Input('interval', 'n_intervals'),
          State('stkName-value', 'data'))

    
def update_graph_live(n_intervals, data):
    print('inFunction')	

    if data in FutureMBOSymbolList:
        stkName = data
        symbolNum = FutureMBOSymbolNumList[FutureMBOSymbolList.index(stkName)]

    else:
        stkName = 'NQH4'  
        symbolNum = FutureMBOSymbolNumList[FutureMBOSymbolList.index(stkName)]
        
    
    gclient = storage.Client(project="stockapp-401615")
    bucket = gclient.get_bucket("stockapp-storage")
    blob = Blob('levelTwoMBO', bucket) 
    levelTwoMBO = blob.download_as_text()
    
    
    csv_reader  = csv.reader(io.StringIO(levelTwoMBO))
    
    

    



    csv_rows = []
    [csv_rows.append(row) for row in csv_reader]
       
        
        
    levelTwoMBO = csv_rows[::-1]
    levelTwoMBO = [i for i in levelTwoMBO if i[6] == symbolNum and (i[4] == 'F' or i[4] == 'T')]
    
    """The event action. Can be [A]dd, [C]ancel, [M]odify, clea[R] book, [T]rade, or [F]ill.
    Trade	T	An aggressing order traded.
    Fill	F	A resting order was filled."""


    
    minAgg = []
    for i in levelTwoMBO:
        if i[4] == 'F':
            if int(levelTwoMBO[0][0]) - (60000000000*10) <= int(i[0]):
                minAgg.append(i)
                
    dic = {}
    for i in minAgg:
        if i[2] not in dic:
            dic[i[2]] = [0,0]
        if i[2] in dic:
            if i[5] == 'A':
                dic[i[2]][0] += int(i[3])
            elif i[5] == 'B':
                dic[i[2]][1] += int(i[3])
                
    newDict = []
    for i in dic:
        newDict.append([str(i)+'A',dic[i][0]])
        newDict.append([str(i)+'B',dic[i][1]])
        
    newDict.sort(key=lambda x:float(x[0][:len(x[0])-1]), reverse=True)
    '''
    starttime = timeit.default_timer()
    print("The start time is :",starttime) 
    
    print("The time difference is :", timeit.default_timer() - starttime)
    
    '''
    
    
    
    minAgg2 = []
    for i in levelTwoMBO:
        if i[4] == 'T':
            if int(levelTwoMBO[0][0]) - (60000000000*10) <= int(i[0]):
                minAgg2.append(i)
                
    dic2 = {}
    for i in minAgg2:
        if i[2] not in dic2:
            dic2[i[2]] = [0,0]
        if i[2] in dic2:
            if i[5] == 'A':
                dic2[i[2]][0] += int(i[3])
            elif i[5] == 'B':
                dic2[i[2]][1] += int(i[3])
                
    
                
    newDict2 = []
    for i in dic2:
        newDict2.append([str(i)+'A',dic2[i][0]])
        newDict2.append([str(i)+'B',dic2[i][1]])
        
    newDict2.sort(key=lambda x:float(x[0][:len(x[0])-1]), reverse=True)
    
    
    Ask = sum([i[1] for i in newDict2 if 'A' in i[0]])
    Bid = sum([i[1] for i in newDict2 if 'B' in i[0]])
    
    dAsk = round(Ask / (Ask+Bid),2)
    dBid = round(Bid / (Ask+Bid),2)
    
    lAsk = sum([i[1] for i in newDict if 'A' in i[0]])
    lBid = sum([i[1] for i in newDict if 'B' in i[0]])
    
    ldAsk = round(lAsk / (lAsk+lBid),2)
    ldBid = round(lBid / (lAsk+lBid),2)
    
    fig = go.Figure()
    
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True,
                        specs=[[{}, {}],], #[{}, {}, ]'+ '<br>' +' ( Put:'+str(putDecHalf)+'('+str(NumPutHalf)+') | '+'Call:'+str(CallDecHalf)+'('+str(NumCallHalf)+') '
                        horizontal_spacing=0.02, vertical_spacing=0.03, subplot_titles=('MO (S:'+str(dAsk)+'('+str(Ask)+')|(B'+str(dBid)+'('+str(Bid)+')', 'LO (S:'+str(ldAsk)+'('+str(lAsk)+')|(B'+str(ldBid)+'('+str(lBid)+')' ),
                         column_widths=[0.5,0.5], ) #row_width=[0.15, 0.85,],row_width=[0.30, 0.70,]


    
    fig.add_trace(
        go.Bar(
            x=pd.Series([i[1] for i in newDict2]),
            y=pd.Series([float(i[0][:len(i[0])-1]) for i in newDict2]),
            text=pd.Series([i[0]  + '(' + str(i[1])+')' for i in newDict2]),
            textposition='auto',
            orientation='h',
            #width=0.2,
            marker_color=[     'red' if 'A' in i[0] 
                        else 'green' if 'B' in i[0]
                        else i for i in newDict2],
            hovertext=pd.Series([i[0]  + ' ' + str(i[1]) for i in newDict2]),
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=pd.Series([i[1] for i in newDict]),
            y=pd.Series([float(i[0][:len(i[0])-1]) for i in newDict]),
            text=pd.Series([i[0]  + '(' + str(i[1])+')' for i in newDict]),
            textposition='auto',
            orientation='h',
            #width=0.2,
            marker_color=[     'red' if 'A' in i[0] 
                        else 'green' if 'B' in i[0]
                        else i for i in newDict],
            hovertext=pd.Series([i[0]  + ' ' + str(i[1]) for i in newDict]),
        ),
        row=1, col=2
    )
    
    fig.update_layout(title=stkName+' ' +str(datetime.now().time()),height=800, xaxis_rangeslider_visible=False, showlegend=False)

    
    fig.update_xaxes(autorange="reversed", row=1, col=2)

    #fig.update_layout(title=stkName+' Aggressive order (Sell:'+str(dAsk)+'('+str(Ask)+')|(Buy'+str(dBid)+'('+str(Bid)+')'+ str(datetime.now().time()),height=800, xaxis_rangeslider_visible=False, showlegend=False)
    #fig.show()
    #print("The time difference is :", timeit.default_timer() - starttime)

    return fig


if __name__ == '__main__': 
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False)
    



                
                
        
        
    
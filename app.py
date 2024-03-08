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

    
FutureMBOSymbolNumList = ['17077', '750', '44740', '1101', '204839',  '7062', '2259', '156627', '156755', '1545', '4122', '270851', '948' ]
FutureMBOSymbolList = ['ESH4','NQH4', 'GCJ4', 'HGK4', 'YMH4', 'RTYH4', '6NH4', '6EH4', '6AH4', '6CH4', 'SIK4', 'CLJ4', 'NGJ4'  ]

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
    html.Div(id='container-button-basic',children="Enter a symbol from |'ESH4' 'NQH4' 'GCJ4' 'HGK4' 'YMH4' 'RTYH4' 'SIK4' 'CLJ4' 'NGJ4' | and submit"),
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
    blob = Blob('levelTwoMBO'+str(symbolNum), bucket) 
    levelTwoMBO = blob.download_as_text()
    
    
    csv_reader  = csv.reader(io.StringIO(levelTwoMBO))


    csv_rows = []
    [csv_rows.append(row) for row in csv_reader]
       
        
        
    levelTwoMBO = csv_rows[::-1]
    levelTwoMBO = [i for i in levelTwoMBO if i[6] == symbolNum and (i[4] == 'T')]
    

    
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
    
    

    
    fig = go.Figure()

    
    fig.add_trace(
        go.Bar(
            x=pd.Series([i[1] for i in newDict2]),
            y=pd.Series([float(i[0][:len(i[0])-1]) for i in newDict2]),
            text=pd.Series([i[0] for i in newDict2]),
            textposition='auto',
            orientation='h',
            #width=0.2,
            marker_color=[     'red' if 'A' in i[0] 
                        else 'green' if 'B' in i[0]
                        else i for i in newDict2],
            hovertext=pd.Series([i[0]  + ' ' + str(i[1]) for i in newDict2]),
        ),
        #row=1, col=2
    )
    
    Ask = sum([i[1] for i in newDict2 if 'A' in i[0]])
    Bid = sum([i[1] for i in newDict2 if 'B' in i[0]])
    
    dAsk = round(Ask / (Ask+Bid),2)
    dBid = round(Bid / (Ask+Bid),2)

    fig.add_hline(y=float(levelTwoMBO[0][2]))
    

    fig.update_layout(title=stkName + ' MO '+str(Ask)+'(Sell:'+str(dAsk)+') | '+str(Bid)+ '(Buy'+str(dBid)+') '+ str(datetime.now().time()),height=800, xaxis_rangeslider_visible=False, showlegend=False)
    #fig.show()
    #print("The time difference is :", timeit.default_timer() - starttime)

    return fig


if __name__ == '__main__': 
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False)
    



                
                
        
        
    
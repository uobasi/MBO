# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 03:03:30 2025

@author: uobas
"""

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

    
#FutureMBOSymbolList = ['ESH4','NQH4','CLH4', 'GCG4', 'NGG4', 'HGH4', 'YMH4', 'BTCZ3', 'RTYH4']
#FutureMBOSymbolNumList = ['17077', '750', '463194', '41512', '56065', '31863', '204839', '75685', '7062', ]

FutureMBOSymbolNumList =  ['42140878', '42002475', '42005850']
FutureMBOSymbolList = ['ES', 'NQ', 'YM']


gclient = storage.Client(project="stockapp-401615")
bucket = gclient.get_bucket("stockapp-storage-east1")


#stkName = 'NQH4'  

from dash import Dash, dcc, html, Input, Output, callback, State
inter = 6000
app = Dash()
app.layout = html.Div([
    
    dcc.Graph(id='graph'),
    dcc.Interval(
        id='interval',
        interval=inter,
        n_intervals=0,
      ),

    #html.Div(dcc.Input(id='input-on-submit', type='text')),
    #html.Button('Submit', id='submit-val', n_clicks=0),
    #html.Div(id='container-button-basic',children="Enter a symbol from |'ES' 'NQ'| and submit"),
    #dcc.Store(id='stkName-value')

    html.Div([
        html.Div([
            dcc.Input(id='input-on-submit', type='text', className="input-field"),
            html.Button('Submit', id='submit-val', n_clicks=0, className="submit-button"),
            html.Div(id='container-button-basic', children="Enter a symbol from ES, NQ", className="label-text"),
        ], className="sub-container"),
        dcc.Store(id='stkName-value'),

    ], className="main-container"),
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
        stkName = 'NQ'  
        symbolNum = FutureMBOSymbolNumList[FutureMBOSymbolList.index(stkName)]
        
       
    
    blob = Blob('FuturesTrades'+str(symbolNum), bucket) 
    levelTwoMBO = blob.download_as_text()
    csv_reader  = csv.reader(io.StringIO(levelTwoMBO))


    csv_rows = []
    [csv_rows.append(row) for row in csv_reader]
       
        
        
    levelTwoMBO = csv_rows[::-1]
    #levelTwoMBO = [i for i in levelTwoMBO if i[6] == symbolNum and (i[4] == 'T')]
    

    
    minAgg2 = []
    for i in levelTwoMBO:
        #if i[4] == 'T':
            if int(levelTwoMBO[0][0]) - (60000000000*30) <= int(i[0]):
                minAgg2.append(i)
                
    dic2 = {}
    for i in minAgg2:
        if float(int(i[1]) / 1e9) not in dic2:
            dic2[float(int(i[1]) / 1e9)] = [0,0]
        if float(int(i[1]) / 1e9) in dic2:
            if i[3] == 'A':
                dic2[float(int(i[1]) / 1e9)][0] += int(i[2])
            elif i[3] == 'B':
                dic2[float(int(i[1]) / 1e9)][1] += int(i[2])
                
    
                
    newDict2 = []
    for i in dic2:
        newDict2.append([str(i)+'A',dic2[i][0]])
        newDict2.append([str(i)+'B',dic2[i][1]])
        
    newDict2.sort(key=lambda x:float(x[0][:len(x[0])-1]), reverse=True)
    
    total_vol = {price: vals[0] + vals[1] for price, vals in dic2.items()}

    # Point of Control (price with max total volume)
    poc = float(max(total_vol, key=total_vol.get))
    poc_volume = total_vol[poc]


    
    # Sort by price
    sorted_prices = sorted(total_vol.keys())[::-1]
    volumes = np.array([total_vol[p] for p in sorted_prices])
    
    # Total & 70% cutoff
    total = volumes.sum()
    target = total * 0.7
    
    # Start with POC
    poc_idx = sorted_prices.index(poc)
    value_area = {poc}
    current_sum = total_vol[poc]
    
    low_idx = poc_idx
    high_idx = poc_idx
    
    # Expand outwards until we reach ~70%
    while current_sum < target:
        left = total_vol[sorted_prices[low_idx - 1]] if low_idx > 0 else -1
        right = total_vol[sorted_prices[high_idx + 1]] if high_idx < len(sorted_prices)-1 else -1
        
        # Pick side with higher vol
        if right >= left:
            high_idx += 1
            current_sum += total_vol[sorted_prices[high_idx]]
            value_area.add(sorted_prices[high_idx])
        else:
            low_idx -= 1
            current_sum += total_vol[sorted_prices[low_idx]]
            value_area.add(sorted_prices[low_idx])
    
    # High/Low value areas
    low_va = min(value_area)
    high_va = max(value_area)
    
    #print("Low Value Area:", low_va)
    #print("High Value Area:", high_va)

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
    
    #fig.add_hline(y=csv_rows[len(csv_rows)-1][2])
    
    fig.add_hline(
        y=float(int(levelTwoMBO[0][1]) / 1e9),#float(csv_rows[-1][2]), 
        line_color="black",
        annotation_text=str(float(int(levelTwoMBO[0][1]) / 1e9)),
        annotation_position="top right"
    )
    
    fig.add_hline(
        y=float(poc),
        line_color="blue",
        annotation_text='POC '+str(poc),
        annotation_position="top right"
    )
    
    
    # y_val = float(csv_rows[-1][2])

    # fig.add_trace(
    #     go.Scatter(
    #         x=[min(pd.Series([i[1] for i in newDict2])) , max(pd.Series([i[1] for i in newDict2]))],
    #         # or use your actual x-axis range if you have it
    #         y=[y_val, y_val],
    #         mode="lines+text",
    #         line=dict(color="black"),
    #         text=[str(y_val), ""],            # put text on the first point only
    #         textposition="top right",
    #         showlegend=False,
    #         name="Last price line"
    #     )
    # )
        
    Ask = sum([i[1] for i in newDict2 if 'A' in i[0]])
    Bid = sum([i[1] for i in newDict2 if 'B' in i[0]])
    
    dAsk = round(Ask / (Ask+Bid),2)
    dBid = round(Bid / (Ask+Bid),2)
    
    fig.add_shape(
        type="rect",
        x0=0, x1=max(total_vol.values()),  # full width of your bars
        y0=low_va, y1=high_va,
        fillcolor="crimson",
        opacity=0.09,
        layer="below",        # keep it behind bars
        line_width=0,
        xref="x",
        yref="y"
    )
    

    fig.update_layout(title=stkName + ' MO '+str(Ask)+'(Sell:'+str(dAsk)+') | '+str(Bid)+ '(Buy'+str(dBid)+') '+ str(datetime.now().time()), height=800, xaxis_rangeslider_visible=False, showlegend=False, paper_bgcolor='#E5ECF6')
    #fig.show()
    #print("The time difference is :", timeit.default_timer() - starttime)

    return fig


if __name__ == '__main__': 
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False)
    



                
                
        
        
    
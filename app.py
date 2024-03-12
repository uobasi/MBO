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

def find_clusters(numbers, threshold):
    clusters = []
    current_cluster = [numbers[0]]

    # Iterate through the numbers
    for i in range(1, len(numbers)):
        # Check if the current number is within the threshold distance from the last number in the cluster
        if abs(numbers[i] - current_cluster[-1]) <= threshold:
            current_cluster.append(numbers[i])
        else:
            # If the current number is outside the threshold, store the current cluster and start a new one
            clusters.append(current_cluster)
            current_cluster = [numbers[i]]

    # Append the last cluster
    clusters.append(current_cluster)
    
    return clusters
    
FutureMBOSymbolNumList = ['5602', '13743', '44740', '1101', '80420', '2552', '2259', '156627', '156755', '1545', '4122', '270851', '948' ]
FutureMBOSymbolList = ['ES','NQ', 'GC', 'HG', 'YM', 'RTY', '6N', '6E', '6A', '6C', 'SI', 'CL', 'NG'  ]

#stkName = 'NQH4' 
gclient = storage.Client(project="stockapp-401615")
bucket = gclient.get_bucket("stockapp-storage") 

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
    html.Div(id='container-button-basic',children="Enter a symbol from |'ES' 'NQ' 'GC' 'HG' 'YM' 'RTY' 'SI' 'CL' 'NG'| and submit"),
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
        return 'The input symbol was '+str(value)+" is not accepted please try different symbol from  |'ES' 'NQ' 'GC' 'HG' 'YM' 'RTY' 'SI' 'CL' 'NG'|  ", 'The input symbol was '+str(value)+" is not accepted please try different symbol  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  "

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
        
       
    
    blob = Blob('levelTwoMBO'+str(symbolNum), bucket) 
    levelTwoMBO = blob.download_as_text()
    
    
    csv_reader  = csv.reader(io.StringIO(levelTwoMBO))


    csv_rows = []
    [csv_rows.append(row) for row in csv_reader]
       
        
        
    levelTwoMBO = csv_rows[::-1]
    levelTwoMBO = [i for i in levelTwoMBO if i[6] == symbolNum and (i[4] == 'T')]
    

    
    minAgg2 = []
    for i in levelTwoMBO:
        if i[4] == 'T' and int(i[3]) >= 2: #
            if int(levelTwoMBO[0][0]) - (60000000000*7) <= int(i[0]):
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

    blob = Blob('FuturesTrades'+str(symbolNum), bucket) 
    FuturesTrades = blob.download_as_text()
    
    
    csv_reader  = csv.reader(io.StringIO(FuturesTrades))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    
    STrades = [i for i in csv_rows if i[4] == symbolNum]
    AllTrades = []
    for i in STrades:
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
    
    newwT = [[i[0],i[1],i[2],i[5], i[4],i[3],i[6]] for i in mTrade]    

    ntList = []
    checkDup = []
    for i in newwT:
        if i[0] not in checkDup:
            ntList.append(i)
    
    

    
    fig = go.Figure()

    
    fig.add_trace(
        go.Bar(
            x=pd.Series([i[1] for i in newDict2]),
            y=pd.Series([float(i[0][:len(i[0])-1]) for i in newDict2]),
            text=pd.Series([i[0] for i in newDict2]),
            textposition='auto',
            orientation='h',
            marker_color=[     'red' if 'A' in i[0] 
                        else 'green' if 'B' in i[0]
                        else i for i in newDict2],
            hovertext=pd.Series([i[0]  + ' ' + str(i[1]) for i in newDict2]),
        ),
    )
    
    Ask = sum([i[1] for i in newDict2 if 'A' in i[0]])
    Bid = sum([i[1] for i in newDict2 if 'B' in i[0]])
    
    dAsk = round(Ask / (Ask+Bid),2)
    dBid = round(Bid / (Ask+Bid),2)

    fig.add_hline(y=float(levelTwoMBO[0][2]))

    sortadlist = ntList[:40]
    data = [i[0] for i in sortadlist]
    data.sort(reverse=True)
    differences = [abs(data[i + 1] - data[i]) for i in range(len(data) - 1)]
    average_difference = sum(differences) / len(differences)
    cdata = find_clusters(data, average_difference)
    
    mazz = max([len(i) for i in cdata])
    for i in cdata:
        if len(i) >= 3:
            opac = round((len(i)/mazz)/1.2,2)
            tmt = list(set([float(i[0][:len(i[0])-1]) for i in newDict2]))   
            tmt.sort(reverse=False)     
            #if (float(i[0]) >= tmt[0] and float(i[0]) <= tmt[len(tmt)-1]) or (float(i[len(i)-1]) >= tmt[0] and float(i[len(i)-1]) <= tmt[len(tmt)-1]):
            if (abs(float(i[0]) - float(levelTwoMBO[0][2])) / ((float(i[0]) + float(levelTwoMBO[0][2])) / 2)) * 100 <= 0.05 or (abs(float(i[len(i)-1]) - float(levelTwoMBO[0][2])) / ((float(i[len(i)-1]) + float(levelTwoMBO[0][2])) / 2)) * 100 <= 0.05:
                fig.add_shape(type="rect",
                          y0=i[0], y1=i[len(i)-1], x0=0, x1=max([i[1] for i in newDict2]),
                          fillcolor="darkcyan",
                          opacity=opac)
                
                bidCount = 0
                askCount = 0
                for x in sortadlist:
                    if x[0] >= i[len(i)-1] and x[0] <= i[0]:
                        if x[3] == 'B':
                            bidCount+= x[1]
                        elif x[3] == 'A':
                            askCount+= x[1]
    
                if bidCount+askCount > 0:       
                    askDec = round(askCount/(bidCount+askCount),2)
                    bidDec = round(bidCount/(bidCount+askCount),2)
                else:
                    askDec = 0
                    bidDec = 0
    
    
                
                fig.add_trace(go.Scatter(x=pd.Series(max([i[1] for i in newDict2]))  ,
                                     y= [i[0]]*max([i[1] for i in newDict2]) ,
                                     line_color='rgba(0,139,139,'+str(opac)+')',
                                     text =str(i[0])+ ' (' + str(len(i))+ ') Ask:('+ str(askDec) + ') '+str(askCount)+' | Bid: ('+ str(bidDec) +') '+str(bidCount),
                                     textposition="bottom left",
                                     name=str(i[0])+ ' (' + str(len(i))+ ') Ask:('+ str(askDec) + ') '+str(askCount)+' | Bid: ('+ str(bidDec) +') '+str(bidCount),
                                     showlegend=False,
                                     mode= 'lines',
                                    
                                    ),
                        )
    
                fig.add_trace(go.Scatter(x=pd.Series(max([i[1] for i in newDict2])) ,
                                     y= [i[len(i)-1]]*max([i[1] for i in newDict2]),
                                     line_color='rgba(0,139,139,'+str(opac)+')',
                                     text = str(i[len(i)-1])+ ' (' + str(len(i))+ ') Ask:('+ str(askDec) + ') '+str(askCount)+' | Bid: ('+ str(bidDec) +') '+str(bidCount),
                                     textposition="bottom left",
                                     name= str(i[len(i)-1])+ ' (' + str(len(i))+ ') Ask:('+ str(askDec) + ') '+str(askCount)+' | Bid: ('+ str(bidDec) +') '+str(bidCount),
                                     showlegend=False,
                                     mode= 'lines',
                                    
                                    ),
                        )
        
    

    fig.update_layout(title=stkName + ' MO '+str(Ask)+'(Sell:'+str(dAsk)+') | '+str(Bid)+ '(Buy'+str(dBid)+') '+ str(datetime.now().time()),height=800, xaxis_rangeslider_visible=False, showlegend=False)
    #fig.show()
    #print("The time difference is :", timeit.default_timer() - starttime)

    return fig


if __name__ == '__main__': 
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False)
    



                
                
        
        
    
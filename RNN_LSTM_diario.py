import matplotlib.pyplot as plt
import numpy as np
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
np.random.seed(1234)

import pandas as pd
import numpy as np

def NS(s,o):
    """
    Nash Sutcliffe efficiency coefficient
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficient coefficient
    """
    #s,o = filter_nan(s,o)
    return 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)

###apenas na primeira vez
data=None
model=None

global_start_time = time.time()
epochs = 5
ratio = 0.5
sequence_length = 50


if data is None:
    print('Loading data... ')

    
    ###DADOS DE VERDADE ## MUDAR COM A SERIE DESEJADA
    #fIn = r"todos_os_dados_light.csv"
    fIn= r"G:\Python_Projs\python3\machine_learning\58250000_Consistido_e_NaoConsistido_bloco1.csv"
    fIn2= r"G:\Python_Projs\python3\machine_learning\58250000_Consistido_e_NaoConsistido_bloco2.csv"
    
    
    dfIn = pd.read_csv(fIn,parse_dates=True)
    dfIn = dfIn.fillna(0)
    df1 = pd.DataFrame(index=pd.to_datetime(dfIn.iloc[:,0]),data=dfIn.iloc[:,-1].values,columns=['values'])
    ############
    
    tam =len(df1)
    janela=50
    df2 = pd.DataFrame()
    
    ##JANELA DESLIZANTE
    ###adicionar coluna em pandas demora muito. adicionar apenas depois
    npO = np.zeros(janela)
    matTI=time.time()
    lst1=[]
    for n in range(tam-janela+1):
        if n%1000==0 and n!=0:
            matP = time.time() - matTI
            print('{0:.1f}% - Janela {1} de {2} - tempo: {3:.1f} de {4:.1f} estimado.'.format((n/(tam-janela))*100,n+1,tam-janela,matP,matP/(float(n)/(tam-janela))))
        #newc=df1['values'].shift(-n).reset_index(drop=True)[:janela].values
        newc=df1['values'][n:n+janela].values
        lst1.append(newc)
        #npO = np.vstack((npO,newc))
        #df2[n]=df1['values'].shift(-n).reset_index(drop=True)[:janela]
    df2 = pd.DataFrame(lst1)
    lst1=[]
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    df2_fit = sc.fit(df2.iloc[:,0])
    df2 = sc.transform(df2)
    
    #df2 = df2.sample(frac=1).reset_index(drop=True)
    
    
    # Splitting the dataset into the Training set and Test set
    # Maintaining time order
    row = int(round(0.9 * df2.shape[0]))
    train = df2[:row, :]
    np.random.shuffle(train)
    #train = train.sample(frac=1).reset_index(drop=True)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = df2[row:, :-1]
    y_test = df2[row:, -1]
    dom_y=df1.index.values[row+janela-1:]
    '''
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    '''
    
    
    ## O ultimo valor é a dimensão da medida - no caso é apenas precipitacao. se fosse mais variaveis aumentaria
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
else:
    X_train, y_train, X_test, y_test = data

print('\nData Loaded. Compiling...\n')

if model is None:
    model = Sequential()
    layers = [1, 50, 100, 1]

    model.add(LSTM(
        layers[1],
        input_shape=(None, layers[0]),
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    
try:
    model.fit(
        X_train, y_train,
        batch_size=64, nb_epoch=epochs, validation_split=0.05)
    predicted = model.predict(X_test)
    predicted = np.reshape(predicted, (predicted.size,))
except KeyboardInterrupt:
    print('Training duration (s) : ', time.time() - global_start_time)

try:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_test[:100])
    plt.plot(predicted[:100])
    plt.show()
except Exception as e:
    print(str(e))
print('Training duration (s) : ', time.time() - global_start_time)
coefNS = NS(predicted,y_test)
print('Coeficiente NASH-SUTCLIFF: {0:.3f}'.format(coefNS))
###############
#Imprime grafico toda a serie de validacao

import matplotlib.dates as mdates
plt.close()
years = mdates.YearLocator()
months = mdates.MonthLocator()
yearsFmt = mdates.DateFormatter('%Y')
monthFmt = mdates.DateFormatter('%b%Y')
y_pred = sc.inverse_transform(predicted)
yinv_test = sc.inverse_transform(y_test)
fig, ax = plt.subplots()
plt.plot(dom_y,y_pred,label='previsao')
plt.plot(dom_y,yinv_test,linewidth=.8,linestyle='--',label='dados')
ax.text(.98, .75,
        'NASH-SUTCLIFF: {0:.3f}'.format(coefNS),
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes)
#df1.loc[dom_y,:].plot()
ax.xaxis.set_minor_locator(months)
ax.xaxis.set_major_formatter(monthFmt)
plt.legend()
fig.autofmt_xdate()
plt.show()

#df1.loc[dom_y,:].plot()

###############
#Novas previsoes!

#PEGA AMOSTRA RANDOMICA DE 50 VALORES DE CHUVA
num = (np.random.randint(0,high=df1.shape[0]-janela,size=1)[0])
nx_data = df1.iloc[num:num+janela+1,:].values

nx = nx_data[:-1]
ny=nx_data[-1]
newx = np.reshape(sc.transform(nx), (1, nx.shape[0], 1))
newy = ny
newpred = model.predict(newx)
newpred = np.reshape(newpred, (newpred.size,))
ynp = sc.inverse_transform(newpred)

print('\nValor conhecido:\t{0}\nValor Previsto:  \t{1:.2f}\n\tDif.:\t\t{2:.1f}%'.format(ny[0],
      ynp[0],((ynp[0]-ny[0])/ny[0])*100))


####
#Previsao de n DIAS!
#PEGA Ultima AMOSTRA DE 50 VALORES DE CHUVA
n=3
nx_data = df1.iloc[-50:,-1].values
new_preds=[]
for i in range(n):      # 5 de 5 dias
    nx = nx_data[:-1]
    ny=nx_data[-1]
    newx = np.reshape(sc.transform(nx), (1, nx.shape[0], 1))
    newy = ny
    newpred = model.predict(newx)
    newpred = np.reshape(newpred, (newpred.size,))
    ynp = sc.inverse_transform(newpred)
    new_preds.append(ynp[0])
    nx_data = np.append(nx_data,ynp[0])[-50:]
    
print('\nValores previstos\n{0}'.format(new_preds))

ng=np.append(df1.iloc[-50:,-1].values,new_preds)
plt.plot(ng,label='Previsao '+str(n)+' dias')
plt.plot(df1.iloc[-50:,-1].values)
plt.legend()
plt.grid()
plt.plot()

####

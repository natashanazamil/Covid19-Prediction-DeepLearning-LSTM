#%% imports
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam


# Paths
TRAIN_DATA_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
TEST_DATA_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')
MODEL_PATH = os.path.join(os.getcwd(),'natasha_model','model.h5')

#%% 1. Data Loading
df = pd.read_csv(TRAIN_DATA_PATH)

#%% 2. EDA
df.head()
df.info()
df.isna().sum()


#%% 3. Data Cleaning
df['cases_new'] = pd.to_numeric(df['cases_new'], errors='coerce')

plt.figure()
plt.plot(df['cases_new'])
plt.show()

df.isna().sum()

#%% Missing data --> Interpolation method
df['cases_new'] = df['cases_new'].interpolate(method='polynomial', order=2)

plt.figure()
plt.plot(df['cases_new'])
plt.show()

df.isna().sum()
#%% 4. Feature Selection
data = df['cases_new'].values

#%% 5. Data Preprocessing

mms = MinMaxScaler()
data = mms.fit_transform(np.expand_dims(data,-1))

#%%
X_train = []
y_train = []

win_size = 30

for i in range(win_size, len(data)):
    X_train.append(data[i-win_size:i])
    y_train.append(data[i])

#%% list to array
X_train = np.array(X_train)
y_train = np.array(y_train)

#%% Train-Test split

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.3, shuffle=True, random_state=123)

#%% 6. Model Development

input_shape = np.shape(X_train)[1:]

model = Sequential()
model.add(LSTM(64, input_shape=input_shape,return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))
model.summary()
plot_model(model,show_shapes=True)

optimizer = Adam(learning_rate=0.01)
model.compile(loss='mae',optimizer=optimizer,metrics='mape')

# Callbacks
log_dir = os.path.join(os.getcwd(),'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = TensorBoard(log_dir=log_dir)
es_callback = EarlyStopping(monitor='val_mape', mode=min, patience=80, restore_best_weights =True)
hist = model.fit(X_train,y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=500, 
                    callbacks=[tb_callback, es_callback])

print(hist.history.keys())

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['Training loss','Validation loss'])
plt.show()


#%% 7. Model Analysis 
# Test with test data
# Load test data
df_test = pd.read_csv(TEST_DATA_PATH)

#%% Clean test data
df_test.isna().sum()
df_test.info()

df_test['cases_new'] = pd.to_numeric(df_test['cases_new'],errors='coerce')
df_test['cases_new'] = df_test['cases_new'].interpolate(method='polynomial', order=2)

# Combine train and test data
df_combine = pd.concat((df,df_test))

df_combine = df_combine['cases_new'].values
df_combine = mms.transform(np.expand_dims(df_combine, axis=-1))

# Test data preprocessing
X_actual = []
y_actual = []
for i in range(len(df),len(df_combine)):
    X_actual.append(df_combine[i-win_size:i])
    y_actual.append(df_combine[i])

X_actual = np.array(X_actual)
y_actual = np.array(y_actual)

# %% Predict test data using model developed
y_pred = model.predict(X_actual)

# Analyse result
plt.figure()
plt.plot(y_pred, color='red')
plt.plot(y_actual, color='blue')
plt.legend(['Predicted Cases','Actual Cases'])
plt.xlabel('Time')
plt.ylabel('Nuber of Cases')
plt.show()

print('MAE is {}'.format(mean_absolute_error(y_actual, y_pred)))
print('MAPE is {}'.format(mean_absolute_percentage_error(y_actual, y_pred)))
print('R2 value is {}'.format(r2_score(y_actual, y_pred)))

#%% 8. Model Deployment
model.save(MODEL_PATH)
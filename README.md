# Covid19-Prediction-DeepLearning-LSTM

Covid-19 cases are spread globally putting us into a pandemic. Safety precaution decisions were difficult to make since the number of cases wasn't predicted. Hence, in this project, Malaysias Covid-19 cases were predicted based on the past 30 days.

**MAPE: 8%**

## Model Summary<br/>
![img](/images/model_summary.PNG)<br/>

The model is set to 500 epochs but contains **EarlyStopping**. The model stopped training at the 167th epoch

## Tensorboard - Training Epoch Loss 
![img](/images/epoch_loss.PNG)<br/>
The graph shows that the loss is unstable as it goes up and down throughout the training. However, it does not show that it is overfitting. By the end of the training, I've validated it with a testing dataset.

## Prediction Based On Test Data
![img](/images/output.png)<br/>
Based on the graph, the predictions were able to follow the pattern of the actual.<br/>

* MAE: 0.04654559743555002
* MAPE: 0.08795534036711634
* R2-Score: 0.9700399216304348

## Data source: 
GitHub - MoH-Malaysia/covid19-public : Official data on the COVID-19 epidemic in Malaysia. Powered by CPRC, CPRC Hospital System, MKAK, and MySejahtera.

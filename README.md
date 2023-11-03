# ML-Zoomcamp-Midterm-Project
The subject of this project is time series forecasting. More specifically, forecasting car sales in Norway. The choice of this project is related to my 
job because at the moment I am building a forecasting engine for my company. So, it was really helpful for me to expand my knowledge and I will be able to
apply new things to my professional environment.

Time series forecasting is a classic area of machine learning applications (just like classification and regression). It is a really interesting topic and there is a lot 
of research around it. In this project I did not applied the classic statistical approaches like Holt-Winters and the ARIMA family. I wanted to experiment with the use
of the machine learning models that we studied in the bootcamp. 

The key point of the whole effort is to transform the data into a form that is understandable from the scikit-learn libraries and XGBoost.  This means  that I had to
transform my data in a form so that the formula F(X) = Y could be applied. Additionally, since the objective of the model is to predict future sales in a monthly basis,
the target variable was in a vector form Y = (SalesMonth1, SalesMonth2, ....., SalesMonth12). I worked with the assumption that my model will be providing sales predictions for 
a year ahead forecasting horizon (12 months). And yes, the scikit-learn libraries and XGBoost do have the robustness to handle vector predictions (Y1, Y2, Y3...)


## Instructions and order of execution
If you want to follow my logic of execution this is how I worked:
- At first I downloaded the original dataset from supchains.com/download --> NORWAY CAR SALES. You can also find it in the repo by the name norway_new_car_sales_by_make1.csv
- Then, I did some basic exploration in order to check for missing values, minimum and maximum dates and provided some plotting functions in the EDA.ipynb notebook
- The most important part of this work is the python file make_datasets_w_categorical.py. This file help us build a custom train-test split function for tabularized time series
  enhanced with external categorical variables. It is highly recommended to reproduce the code step-by-step in order to understand the logic

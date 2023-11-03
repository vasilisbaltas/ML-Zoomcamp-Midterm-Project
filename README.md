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
- The transformed dataset(tabularized) suitable for ML training and prediction was also saved in the file tabular_sales_forML.csv
- Two models were explored, random forest and xgboost, and a hyperparameter tuning was also performed. The corresponding notebooks are RandomForest.ipynb and XGBoost.ipynb.
- It should be noted that the Random Forest algorithm appeared to provide us with better results, hence its model(random_forest.bin) was preferred for the production environment.
  Moreover, if you wish to reproduce the xgboost cross-validation procedure(XGBoost.ipynb) I highly recommend to avoid jupyter notebook and do it on your own machine with a different IDE.
  I had runtime problems because it need moderate computing power and time...
  
- I created a pipenv virtual environment for production with the following installation command (did not included xgboost):
  pipenv install pandas numpy scikit-learn matplotlib flask waitress

  The corresponding files (Pipfile, Pipfile.lock) can be found in the repository

- I built a docker image by using the Dockerfile in this repo:
  
  docker build -t <image_name> .

- In order to run the docker container:
  
  docker run -it --rm -p 9696:9696 <image_name>

- While the container is running you can use the notebook sending_requests.ipynb to obtain 12-month future sales forecasts for car brands in Norway.
  Please provide to your http call only car brands stated in the list - I have not implemented exception handling for brands that do not exist


Finally, in order ton run all the above, please place all the files in the same folder.

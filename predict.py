import pickle
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify


TIME_WINDOW = 24                              # historical observations input  to the model

### LOAD THE MODEL
model_file = 'random_forest.bin'
with open(model_file,'rb') as f_in:
    model = pickle.load(f_in)


df = pd.read_csv('tabular_sales_forML.csv', index_col='Make')


# cars = {'brands' : ['Audi','Volkswagen','Toyota']}


app = Flask('sales_forecasting')

@app.route('/predict', methods=['POST'])
def predict():

     cars = request.get_json()

     future_sales = {}
     for car in cars['brands']:

         past_sales = df[df.index==car]
         X = past_sales.iloc[:, -TIME_WINDOW - 1:].values                   ### X = past 24 months sales + the encoded brand

         future_sales[car] = model.predict(X)[0].astype(int).tolist()

     return jsonify(future_sales)




if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696)



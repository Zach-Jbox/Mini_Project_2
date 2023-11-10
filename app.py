from flask import Flask, session, redirect, render_template,request,jsonify,url_for
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
import pandas as pd
import numpy as np
import joblib
import requests
import json

app = Flask(__name__)

#Setting up the connection to the sql database
conn = sqlite3.connect('layering.db', check_same_thread=False)
c = conn.cursor()
conn.commit()

#Homepage route
@app.route('/')
def home():
    return render_template('home_page.html')

#Pipeline Training route
@app.route("/pipeline", methods=['GET', 'POST'])
def pipeline():
    
    if request.method == 'POST':
        
        #Converting the sql table into a pandas dataframe and replacing the NULL values with Nan values, then replacing the Nan values with the mean of the columnn
        sql_query = pd.read_sql_query("SELECT * FROM circles", conn)
        df = pd.DataFrame(sql_query, columns = ['Test1', 'Test2', 'layer'])
        df = df.replace("NULL", np.nan)
        df = df.fillna(df.mean())
        
        #Splitting the data into X and y and training the model
        X = df[['Test1', 'Test2']]
        y = df[['layer']]
        y = y.values.ravel()
        X_numerical = X.select_dtypes(exclude="object").columns
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)

        #Creating a pipeline and training the data with the SVC() Model
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, X_numerical),
            ])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', SVC())
        ])

        #Fit the pipeline on training data
        pipeline.fit(X_train, y_train)

        #Make predictions on test data
        y_pred = pipeline.predict(X_test)

        #Saving the pipeline model for later use
        joblib.dump(pipeline, 'pipeline.pkl')
        
        #Saving the accuracy values and sending them to the html index page
        accuracy = accuracy_score(y_test, y_pred)
        return render_template("index.html", accuracy=accuracy)
    
    return render_template("index.html")

#Creating a route for reading in the user inputs from the html page and training the data
@app.route("/reading", methods=['GET', 'POST'])
def reading():
    
    #Reading the user inputs in from the html page
    if request.method == 'POST':
        new_data1 = request.form.get('new_data1')
        new_data2 = request.form.get('new_data2')
        
        #If both user inputs are recorded then the following if statment will be executed
        if new_data1 and new_data2:
            #Assigning the users inputs to a numpy array
            new = np.append(new_data1, new_data2)
            #Reshaping the array to fit the rest of the data
            new = np.reshape(new, (-1, 2))
            #Converting the user inputs to a pandas dataframe and assigning them names
            user_input = pd.DataFrame(new, columns = ['Test1','Test2',])
            #Loading the pipline model from earlier
            loaded_pipeline = joblib.load('pipeline.pkl')
            #Training the new user inputed data into the model and training it 
            new_predictions = loaded_pipeline.predict(user_input)
            #Converting the prediction value into an integer
            prediction = int(new_predictions[0])

            #Creating a new sql table named users
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS "users" (
                "input1" REAL,
                "input2" REAL,
                "layer" INTEGER
            );
            """
            #Execute the SQL statement and commiting the new values to the new table in the database
            c.execute(create_table_sql)
            c.execute("INSERT INTO users (input1, input2, layer) VALUES(?, ?, ?)", (new_data1, new_data2, prediction))
            conn.commit()

            #The final prediction is as stated below and fed to the html page prediction.html
            final_prediction = f"Your prediction is: {prediction}"
            return render_template('prediction.html', new_data1=new_data1, new_data2=new_data2, prediction=prediction, final_prediction=final_prediction)
        
        #new if statment that tells the user to input both values in order to get a response
        if new_data1 or new_data2:
            final_prediction = f"Please enter a value in all fields."
            return render_template('prediction.html', final_prediction=final_prediction)
        
        #Reloads the page if nothing is entered
        else:
            return render_template('prediction.html')     
    
    return render_template("prediction.html")

app.run(debug=True,port=8100)

# I really enjoyed learning how the pipeline function can be used to save machine learning
# models and use them to train new data, learning how to update the database was tricky but overall the
# process wasn't too complicated

# I chose to use the SVM, specifically the SVC() model as I have used it before and am more familiar
# with the algorithm, I looked into the NN model and took the opportunity to learn about the model but 
# overall decided to stick with what I know best

# I used jupyter notebook to test the pipeline model, I used jupyter notebook as it was the best way to
# see what parts of my code were not working and fixed the errors based on what errors I recieved
# I also ran the code on the html and css docs to make changes based on what I thought looked best 


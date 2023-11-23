# Disaster-Response-Pipeline
A web app where an emergency worker can input a new message and get classification results in several categories. 

![1](https://github.com/kwankhede/Disaster-Response-Pipeline/blob/webapp/ai_kapil.png)

## Project Overview
As a part of Udacity's Data Science Nanodegree, I got a chance to use my data engineering skills in this project where I applies these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

- This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. This project shows off my software skills, including my ability to create basic data pipelines and write clean, organized code!

## Project Components
There are three components for this project.

### 1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

### 2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

### 3. Flask Web App
We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:

Modify file paths for database and model as needed
Add data visualizations using Plotly in the web app. One example is provided for you
Github and Code Quality
Your project will also be graded based on the following:

Use of Git and Github
Strong documentation
Clean and modular code

Follow the RUBRIC when you work on your project to assure you meet all of the necessary criteria for developing the pipelines and web app.

Go to http://0.0.0.0:3001/

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
    
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.

         `python run.py`

3. Go to http://0.0.0.0:3001/


Screenshopts of the web app:

![Webapp](https://github.com/kwankhede/Disaster-Response-Pipeline/blob/master/images/webapp.png)

![Genre](https://github.com/kwankhede/Disaster-Response-Pipeline/blob/master/images/graph1.png)

![Distribution of disaster message categories](https://github.com/kwankhede/Disaster-Response-Pipeline/blob/master/images/graph2.png)



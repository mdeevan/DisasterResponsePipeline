# Disaster Response Pipeline Project

# Motivation
Following a disaster, affecting a large number of people, millions and millions of communications are sent via social media to the disaster response organization. The project is apply the natural language processing and machine learning to classify the text and tweets sent during such disasters. For this Udacity project, Figure Eight has provided a pre-labeled tweets and text messages from real-life disasters and objective is to classify the message into the pre-defined categories.

Project consists of three part

1. Build a ETL Pipeline. 
Read and combine the categories and messages data, then clean the data by removing the duplicates, and saving it as a database

2. Build a Machine Learning Pipeline
Build a machine learning model, by reading the database saved in ETL step, training and testing the model, and finally saving the model as a pickle file

3. Web Application
Web application that reads in the pickle file and categorizes it the message into the one or more of the pre-defined categories



# Installation
Following packages are required  
By using pip3 or yum as follows, install the following packages  

pip3 install <package name>  
yum install <package name>  	


nltk  
sqlalchemy  
sklearn  
numpy  
pandas  
seaborn  
matplotlib  



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database  
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves  
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

#### Please note:  
Due to the size of the pkl file, it is not added to the project. ML pipeline needs to be executed to generate the required classifier.pkl, and it takes a little while to execute

Accuracy of the classification is affected by the amount of data.

Creating the model takes longer to complete, and further fine tuning of the model can help produce better classification.

### Sample Output
![image](https://user-images.githubusercontent.com/4894031/140642008-158fbe10-34c2-4544-9a7e-236f58b36ba3.png)

   

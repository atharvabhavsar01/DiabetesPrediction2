from django.shortcuts import render 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def home(request):
    return render(request,'home.html')

def predict(request):
    data = pd.read_csv("diabetes.csv")
    #storing the reamining var in x 
    X = data.drop("Outcome", axis=1)

    #result col in y
    Y = data["Outcome"]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train,Y_train)
    
    value1 = float(request.POST['pregnancies'])
    value2 = float(request.POST['glucose'])
    value3 = float(request.POST['blood_pressure'])
    value4 = float(request.POST['skin_thickness'])
    value5 = float(request.POST['insulin'])
    value6 = float(request.POST['bmi'])
    value7 = float(request.POST['Diabetes_ped'])
    value8 = float(request.POST['Age'])

    prediction = model.predict([[value1,value2,value3,value4,value5,value6,value7,value8]])

    print(prediction)
    
    result= ""

    if prediction==[1]:
        result = "Positive"
    else:
        result= "negative"

    return render(request,'home.html',{"prediction":result})

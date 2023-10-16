# Software Engineer (ML & LLMs) Challenge documentation

## Description

This document describes the implementation of the Software Engineer (ML & LLMs) Challenge from Latam - Airlines. The solution consists of a machine learning model to predict the delay probability of a flight, deployed as an API on Google Cloud Platform (GCP). The code is well-written, organized, and documented, following SOLID principles. It is also extensible and scalable, making it a good foundation for future development.

## Part I: Model implementation

The model is implemented in Model.py, using the XGBoost classifier algorithm. XGBoost was chosen for its speed and efficiency, even for large datasets. The model is trained on a dataset of historical flight data. This class ensure the following:

- Preprocessing data, from api request and traing model
- Training the model using XGboost classifier algorithm
- Predict the delay probability of a flight

## Model Selection

Accordingly with the challenge both models presented in the Jupyter Notebook, both acted the same way with the training data. Despite the results of both algorithm XGboost and LogisticRegression the choise of xgb_model_2 was based on its advantages i.e. designed to be fast and efficient, even for large datasets. This would help to make the code extensible following SOLID principles.

## Part II: API deployment

In order to implement an API deployment FASTAPI framework was implmented. The API has two main endpoints:

- /health: Check if API is working correctly (GET request)
- /predict: Accepts data from flights and returns if fly will be delayed

The API has validators to avoid bad inputs.


## Part III: GCP Deployment 

Dockerfile and app.yaml where created in order to build and deploy application on the cloud. Specifying requirements on the requirements.txt file

## Part IV: CI/CD Implementation

A GitHub Actions workflow is used to automate the build, test, and deployment of the API. This workflow ensures that the API is always up-to-date and ready to use.

## Conclusion

The Software Engineer (ML & LLMs) Challenge from Latam - Airlines was successfully implemented using the latest machine learning and cloud computing technologies. The resulting solution is a well-designed, scalable, and easy-to-use API that can be used to predict the delay probability of a flight.



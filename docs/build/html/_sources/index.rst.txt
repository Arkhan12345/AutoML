.. AutoML App documentation master file, created by
   sphinx-quickstart on Thu Nov  7 19:03:14 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AutoML App's documentation!
======================================
Teammates:

Aspi Arkhan (s5068290)

H.W. Beintema (s5149169)

Significant decision choices:

.. code-block::
   
   # DSC-0001-use-numpy: 
   # Date: 2024-10-14
   # Decision: Use numpy to create ndarrays
   # Status: Accepted
   # Motivation: lack of functionality in standard python
   # Reason: numpy arrays have excellent additional functionality and are globally used
   # Limitations: not suited for complex array manipulation
   # Alternatives: tensorflow

.. code-block::

   # DSC-0002-Use-Deepcopy:
   # Date: 2024-10-14
   # Decision: Use Deepcopy for data returns
   # Status: Accepted
   # Motivation: prevent data leaks from objects
   # Reason: Deepcopy copies the data of objects on a highly detailed level
   # Limitations: data copy is still mutable
   # Alternatives: pickle copy

.. code-block::
   
   # DSC-0001: 
   # Date: 2024-11-10
   # Decision: Gave Modelling.py the same functionality of Dataset.py
   # Status: Accepted
   # Motivation: Not possible to import Dataset.py into Modelling.py
   # Reason: Unconventional naming of Dataset.py
   # Limitations: Could not abide to the instructions

**Modelling.py**
This code provides an interface using Streamlit to preprocces, train and evaluate machine learning pipelines.
It uses the AutoMLSystem singleton for managing datasets, models, and artifacts.
It has a custom PipelineModelling class that encapsulates every process.


*_select_dataset* displays a dropdown to allow the user to select a dataset.
*_features* identifies input and target features in the dataset.
*models* prompts the user to select a model based on the task type.
*split*prompts the user to select a test set proportion.
*metrics* prompts the user to select evaluation metrics based on the task type.
*summary* creates and displays a summary of the pipeline configuration.
*train* trains the model and displays the results.
*save* prompts the user for name and version to save the pipeline as an artifact.

**Deployment.py**
Streamlit page that lists saved artifacts alongside their information.

**Datasets.py**
Streamlit page that manages datasets.

**metric.py**
Contains the Metric abstract base class, 3 regression metric classes and 3 classification metric classes.
*MeanAbsoluteError* class calculates the mean absolute error between the predicted and actual values.
*MeanSquaredError* class calculates the mean squared error between the predicted and actual values.
*RootMeanSquaredError* class calculates the root mean squared error between the predicted and actual values.
*Accuracy* class calculates the accuracy of the model.
*WeightedPrecision* class calculates the weighted precision of the model.
*WeightedRecall* class calculates the weighted recall of the model.

**artifact.py**
Contains the Artifact abstract base class containing every information about the artifact:
name, asset path, version, data, metadata, type, tags and id.
*generate_id* method generates a unique id for the artifact.
*get_metadata* returns the metadata of the artifact.

**detect_feature_types**
Function that detects the feature types of a dataset.
Accepts only categorical and numerical features and no NaN values

**model**
Contains the Model abstract base class, 3 regression model classes and 3 classification model classes.
*LassoRegression* class implements lasso regression using a scikit wrapper.
*MultipleLinearRegression* class implements multiple linear regression using a scikit wrapper.
*RidgeRegression* class implements ridge regression using a scikit wrapper.
*DTreeClassifier* class implements decision tree classification using a scikit wrapper.
*KNearestClassifier* class implements k-nearest neighbors classification using a scikit wrapper.
*SupportVectorClassifier* class implements support vector classification using a scikit wrapper.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`search`

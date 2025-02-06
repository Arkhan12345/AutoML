Streamlit application that allows users to train models with an AutoML platform.
Implements a pipeline that orchestrates preprocessing, data split, training and evaluation.

Streamlit page:
`ST/page/datasets`: Page to manage the datasets. Upload CSV and convert it into a dataset. Save dataset artifact object.

`ST/page/modelling`: Page to model a pipeline. 
- `ST/modelling/list`: Load uploaded datasets from the artifact registry.
- `ST/modelling/features`: Select input features, a target feature, and detect task type (classification or regression).
- `ST/modelling/models`: Prompt the user to select a model based on the task type.
- `ST/modelling/split`: Select train-test split.
- `ST/modelling/metrics`: Select metrics.
- `ST/modelling/summary`: Provides pipeline summary
- `ST/modelling/train`: Trains the class and returns results.
- `ST/modelling/save`: Prompt the user to give a name and version for the pipeline and convert it into an artifact which can be saved.

`ST/page/deployment`: Page where you can see existing saved pipelines.
- `ST/deployment/load`: Select existing pipelines and show a pipeline summary.
- `ST/deployment/predict`: Once the user loads a pipeline, prompt them for a CSV on which they can perform predictions.

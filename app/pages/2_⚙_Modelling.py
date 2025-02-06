import os
import pickle
import streamlit as st
from typing import Optional, List, Tuple, Union

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types


from autoop.core.ml.pipeline import Pipeline

from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.lasso_regression import LassoRegression
from autoop.core.ml.model.regression.ridge_regression import RidgeRegression

from autoop.core.ml.model.classification.k_neighbours_classifier import (
    KNearestClassifier,
)
from autoop.core.ml.model.classification.support_vector import (
    SupportVectorClassifier,
)
from autoop.core.ml.model.classification.decision_tree import DTreeClassifier

from autoop.core.ml.metric import get_metric, METRICS
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline\
                  to train a model on a dataset.")


class PipelineModelling:
    """Encapsulates the pipeline modelling."""
    def __init__(self):
        """Initialize AutoML system and load available datasets."""
        self.automl = AutoMLSystem.get_instance()
        self.datasets = self._list()

    def _list(self) -> List[Dataset]:
        """
        Loads the list of available datasets from the AutoML system registry.

        Returns:
            List[Dataset]: A list of datasets registered in the AutoML system.
        """
        return self.automl.registry.list(type="dataset")

    def _select_dataset(self) -> Optional[Dataset]:
        """
        Select a dataset from the AutoML system registry using
        a Streamlit selectbox.

        Returns:
            Optional[Dataset]: A dataset selected from the AutoML system
            registry, or None.
        """
        dataset_list = [
            f"{ds.name} (ID: {ds.id.split(':')[-1]})" for ds in self.datasets
        ]
        selected_name = st.selectbox("Select a Dataset:", dataset_list)
        if selected_name:
            dataset = next(ds for ds in self.datasets
                           if f"{ds.name} (ID: {ds.id.split(':')[-1]})" ==
                           selected_name)
            return Dataset(
                name=dataset.name,
                asset_path=dataset.asset_path,
                version=dataset.version,
                data=dataset.data
            )
        return None

    def _features(self, dataset: Dataset) -> (Tuple[List[Feature],
                                                    Feature, str]):
        """
        Detects the features in a dataset and returns a tuple
        containing the list of input features, the target feature,
        and the task type.

        Parameters:
            dataset (Dataset): The dataset to detect features from.

        Returns:
            Tuple[List[Feature], Feature, str]: A tuple containing the
            list of input features, the target feature, and the task type.
        """
        df = dataset.read()
        features = detect_feature_types(dataset)
        numerical_features = [f.name for
                              f in features if f.type == "numerical"]
        categorical_features = [f.name for
                                f in features if f.type == "categorical"]

        input_features = []
        input_feature_names = st.multiselect("Select input features",
                                             numerical_features +
                                             categorical_features)
        for feature_name in input_feature_names:
            if feature_name in numerical_features:
                feature_type = "numerical"
            elif feature_name in categorical_features:
                feature_type = "categorical"
            else:
                raise ValueError(f"Unknown feature type for {feature_name}")
            input_features.append(
                Feature(name=feature_name, type=feature_type))

        available_features = [
            feature for feature in df.columns if
            feature not in input_feature_names
        ]

        target_feature_name = st.selectbox("Select target feature:",
                                           available_features)
        if target_feature_name in numerical_features:
            feature_type = "numerical"
            task_type = "regression"
        else:
            feature_type = "categorical"
            task_type = "classification"
        target_feature = Feature(name=target_feature_name, type=feature_type)

        st.write(f"Detected Task Type: {task_type}")
        return input_features, target_feature, task_type

    def models(self, task_type: str) -> Union[LassoRegression, RidgeRegression,
                                              MultipleLinearRegression,
                                              KNearestClassifier,
                                              SupportVectorClassifier,
                                              DTreeClassifier]:
        """
        Prompts the user to select a machine learning model.

        Args:
            task_type (str): The type of task, which can be either
            "regression" or "classification".

        Returns:
            Union[LassoRegression, RidgeRegression, MultipleLinearRegression,
                KNearestClassifier, SupportVectorClassifier, DTreeClassifier]:
                The selected model instance corresponding to
                the chosen task type.
        """
        if task_type == "regression":
            model = st.selectbox("Select Regression Model:",
                                 [LassoRegression(),
                                  RidgeRegression(),
                                  MultipleLinearRegression()])
        else:
            model = st.selectbox("Select Classification Model:",
                                 [KNearestClassifier(),
                                  SupportVectorClassifier(),
                                  DTreeClassifier()])
        return model

    def split(self) -> float:
        """
        Prompts the user to select the proportion of the dataset to be used
        for the test set.

        Returns:
            float: The proportion of the dataset to be used as the test set,
            as selected by the user.
        """
        return st.slider("Test Set Proportion", 0.1, 0.8, 0.2)

    def metrics(self, task_type: str) -> List[str]:
        """
        Prompts the user to select metrics for the given task type.

        Args:
            task_type (str): The type of task, which can be either
            "regression" or "classification".

        Returns:
            List[str]: A list containing a single string representing the
            selected metric.
        """
        if task_type == "regression":
            selected_metrics = st.multiselect(
                "Select Metrics:", METRICS[:3], default=[METRICS[0]]
            )
        else:
            selected_metrics = st.multiselect(
                "Select Metrics:", METRICS[3:], default=[METRICS[3]]
            )

        metrics = [get_metric(metric) for metric in selected_metrics]
        return metrics

    def summary(self, model, dataset: Dataset, input_features: List[Feature],
                target_feature: Feature, test_size: float, metrics: List[str]):

        """
        Shows a summary of the pipeline and returns the pipeline object.

        This function receives the model, dataset, input and target features,
        split ratio and metrics as parameters.
        It creates a pipeline object with the given parameters and shows a
        summary of the pipeline if the button is clicked.
        The summary includes the model name, input and target feature names,
        split ratio and metric names.
        Finally, it returns the pipeline object.
        """
        pipeline = Pipeline(
                metrics=metrics,
                dataset=dataset,
                model=model,
                input_features=input_features,
                target_feature=target_feature,
                split=test_size
            )

        if st.button("Show Pipeline Summary"):
            st.write("Pipeline Summary")
            st.write("Model:", model)
            st.write("Input Features:", [f.name for f in input_features])
            st.write("Target Feature:", target_feature.name)
            st.write("Split Ratio:", test_size)
            st.write("Metrics:", [metric.name for metric in metrics])

        return pipeline

    def train(self, pipeline: Pipeline):
        """
        Train a model with the given pipeline and display the results.

        Parameters:
        pipeline (Pipeline): the pipeline to use for training and evaluation

        Returns:
        None
        """
        if st.button("Train & Evaluate Model"):
            results = pipeline.execute()
            st.write("Model Trained")

            st.header("Pipeline Results")
            st.write("Test Metrics:")
            for metric_obj, value in results["test metrics"]:
                st.write(f"{metric_obj.name}: {value}")

            st.write("Training Metrics:")
            for metric_obj, value in results["training metrics"]:
                st.write(f"{metric_obj.name}: {value}")

    def save(self, pipeline: Pipeline):
        """
        Prompts the user for a name and version, trains the pipeline,
        and saves it as an artifact.

        Args:
            pipeline (Pipeline): The configured pipeline object to be
            saved as an artifact.
        """
        st.header("Save Pipeline")

        pipeline_name = st.text_input("Enter a name:")
        pipeline_version = st.text_input("Enter a version:")
        asset_path = os.path.abspath("pipeline.py")

        if st.button("Train and Save Pipeline"):
            try:
                artifact_data = {
                    "model": pipeline.model,
                    "input_features": [feature.name for feature
                                       in pipeline._input_features],
                    "target_feature": pipeline._target_feature.name,
                    "split_ratio": pipeline._split,
                    "metrics": [metric.name for metric in pipeline._metrics],
                }

                serialized_data = pickle.dumps(artifact_data)
                artifact = Artifact(
                    name=pipeline_name,
                    asset_path=asset_path,
                    version=pipeline_version,
                    data=serialized_data,
                    type="pipeline"
                )

                self.automl.registry.register(artifact)
                st.success(f"Pipeline '{pipeline_name}' version\
                    '{pipeline_version}' saved successfully!")

            except Exception as e:
                st.error(f"Error saving the pipeline: {e}")


def main():
    page = PipelineModelling()

    st.header("Step 1: Load the Dataset")
    dataset = page._select_dataset()
    if dataset:
        df = dataset.read()
        st.write(df)

        st.header("Step 2: Feature Selection")
        input_features, target_feature, task_type = page._features(dataset)

        st.header("Step 3: Model Selection")
        model = page.models(task_type)

        st.header("Step 4: Split the Data")
        split_ratio = page.split()

        st.header("Step 5: Select Metrics")
        metrics = page.metrics(task_type)

        st.header("Step 6: Pipeline Summary")
        pipeline = page.summary(model, dataset,
                                input_features,
                                target_feature,
                                split_ratio,
                                metrics)

        st.header("Step 7: Train and Evaluate Model")
        page.train(pipeline)

        st.header("Extra requirements: Save the pipeline")
        page.save(pipeline)


if __name__ == "__main__":
    main()

from typing import List, Optional, Tuple
import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types


class DatasetManagement:
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

    def select_dataset(self) -> Optional[Dataset]:
        """Enable users to upload a dataset."""
        st.header("Register a Dataset")

        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of Uploaded Dataset:")
                st.dataframe(df.head())

                dataset_name = st.text_input("Enter a name forthe dataset:",
                                             value="uploaded_dataset")
                dataset_version = st.text_input("Enter a \
                                                version for the dataset:",
                                                value="1.0.0")

                if st.button("Register Dataset"):
                    dataset = Dataset.from_dataframe(
                        name=dataset_name,
                        asset_path=f"{dataset_name}.csv",
                        data=df,
                        version=dataset_version,
                    )
                    self.automl.registry.register(dataset)
                    st.success(f"Dataset '{dataset_name}' with version\
                               '{dataset_version}' was registered\
                                successfully!")
            except Exception as e:
                st.error(f"An error occurred while uploading the file: {e}")

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

        target_feature_name = st.selectbox("Select target feature:",
                                           df.columns)
        if target_feature_name in numerical_features:
            feature_type = "numerical"
            task_type = "regression"
        else:
            feature_type = "categorical"
            task_type = "classification"
        target_feature = Feature(name=target_feature_name, type=feature_type)

        st.write(f"Detected Task Type: {task_type}")
        return input_features, target_feature, task_type

    def preview_datasets(self):
        """Display a dropdown to select a registered dataset
        and show its preview."""
        st.header("Registered Datasets")

        dataset_list = [
            f"{ds.name} (ID: {ds.id.split(':')[-1]})" for ds in self.datasets
        ]
        selected_name = st.selectbox("Select a Registered Dataset \
                                     to Preview:", dataset_list)
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


def main():
    page = DatasetManagement()

    dataset = page.select_dataset()
    if dataset:
        df = dataset.read()
        st.write(f"Dataset selected: {dataset.name}")
        st.write(df)

        st.header("Feature Selection")
        input_features, target_feature, task_type = page._features(dataset)
        st.write("Input Features:", [f.name for f in input_features])
        st.write("Target Feature:", target_feature.name)
        st.write("Task Type:", task_type)

    dataset = page.preview_datasets()
    if dataset:
        df = dataset.read()
        st.write(df)


if __name__ == "__main__":
    main()

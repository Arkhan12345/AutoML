import pandas as pd
import streamlit as st
from app.core.system import AutoMLSystem
import pickle

from autoop.core.storage import NotFoundError


st.set_page_config(page_title="Deployment")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# Deployment")
write_helper_text("In this section, you can load pipelines,\
                  upload datasets, and generate predictions.")


class DeploymentPage:
    def __init__(self):
        """Initialize the AutoML system and retrieve saved pipelines."""
        self.automl = AutoMLSystem.get_instance()
        try:
            self.saved_pipelines = self.automl.registry.list(type="pipeline")
        except NotFoundError as e:
            st.error(f"Error: {e}. Please ensure the specified path exists.")
            self.saved_pipelines = []

    def load(self):
        """Load a saved pipeline."""
        if self.saved_pipelines:
            pipeline_labels = [
                f"{p.name} (version:{p.version})" for p in self.saved_pipelines
            ]
            selected_label = st.selectbox("Select a Pipeline:",
                                          pipeline_labels)
        else:
            st.write("No saved pipelines found.")
            return None

        if selected_label:
            name, version = selected_label.rsplit("(version:", 1)
            pipeline_name = name.strip()
            pipeline_version = version.rstrip(")").strip()

            selected_pipeline = next(
                (
                    p for p in self.saved_pipelines
                    if p.name == pipeline_name and
                    str(p.version) == pipeline_version
                ),
                None
            )
            pipeline_data = pickle.loads(selected_pipeline.data)
            st.write(f"### Pipeline: {selected_pipeline.name} (Version:\
                     {selected_pipeline.version})")
            st.markdown(f"""
            - **Model:** {pipeline_data['model']}
            - **Input Features:** {', '.join(pipeline_data.get
                                             ('input_features'))}
            - **Target Feature:** {pipeline_data.get('target_feature')}
            - **Split Ratio:** {pipeline_data.get('split_ratio')}
            - **Metrics:** {', '.join(pipeline_data.get('metrics'))}
            """)

            return pipeline_data

        return None

    def upload_dataset(self):
        """Allow the user to upload a dataset."""
        uploaded_file = st.file_uploader("Upload Dataset:", type=["csv"])
        if uploaded_file:
            dataset = pd.read_csv(uploaded_file)
            st.write("### Uploaded Dataset")
            st.write(dataset.head())
            return dataset
        return None

    def validate_dataset(self, dataset, pipeline_data):
        """Check if the dataset matches the pipeline input features."""
        pipeline_features = set(pipeline_data.get('input_features'))
        dataset_features = set(dataset.columns)
        missing_features = pipeline_features - dataset_features

        if missing_features:
            st.error(f"The dataset is missing these features:\
                     {', '.join(missing_features)}")
            return False
        st.success("Dataset is compatible with the pipeline.")
        return True

    def generate_predictions(self, pipeline_data, dataset):
        model = pipeline_data['model']

        input_features = pipeline_data.get('input_features')
        target_feature = pipeline_data.get('target_feature')

        X = dataset[input_features].values
        predictions = model.predict(X)

        st.write("### Predictions")
        st.write(predictions)

        if target_feature in dataset.columns:
            y_true = dataset[target_feature].values
            metrics = [pickle.loads(metric) for
                       metric in pipeline_data.get('metrics')]
            st.write("### Metrics")
            for metric in metrics:
                result = metric.evaluate(y_true, predictions)
                st.write(f"{metric.name}: {result}")

    def run(self):
        """Main method to handle the deployment process."""
        st.header("Step 1: Load Pipeline")
        pipeline_data = self.load()

        if pipeline_data:
            st.header("Step 2: Upload Dataset")
            dataset = self.upload_dataset()

            if dataset is not None:
                if self.validate_dataset(dataset, pipeline_data):
                    st.header("Step 3: Generate Predictions")
                    self.generate_predictions(pipeline_data, dataset)


if __name__ == "__main__":
    page = DeploymentPage()
    page.run()

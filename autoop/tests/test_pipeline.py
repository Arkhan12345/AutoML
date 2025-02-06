from sklearn.datasets import fetch_openml
import unittest
import pandas as pd

from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.metric import MeanSquaredError


class TestPipeline(unittest.TestCase):

    def setUp(self) -> None:
        """
        Sets up a pipeline for testing with the Adult dataset.
        """
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        self.dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        self.features = detect_feature_types(self.dataset)
        self.pipeline = Pipeline(
            dataset=self.dataset,
            model=MultipleLinearRegression(),
            input_features=list(filter(lambda x: x.name != "age",
                                       self.features)),
            target_feature=Feature(name="age", type="numerical"),
            metrics=[MeanSquaredError()],
            split=0.8
        )
        self.ds_size = data.data.shape[0]

    def test_init(self):
        """
        Tests that the object created is an instance of Pipeline.
        """
        self.assertIsInstance(self.pipeline, Pipeline)

    def test_preprocess_features(self):
        """
        Tests the _preprocess_features method of the Pipeline.

        Ensures that the number of artifacts registered in the pipeline
        matches the number of features after preprocessing.
        """
        self.pipeline._preprocess_features()
        self.assertEqual(len(self.pipeline._artifacts), len(self.features))

    def test_split_data(self):
        """
        Tests that the _split_data method correctly splits the preprocessed
        data into a training and test set.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.assertEqual(self.pipeline._train_X[0].shape[0],
                         int(0.8 * self.ds_size))
        self.assertEqual(self.pipeline._test_X[0].shape[0],
                         self.ds_size - int(0.8 * self.ds_size))

    def test_train(self):
        """
        Tests the _train method of the Pipeline.

        Ensures that the model is trained and has some parameters.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.assertIsNotNone(self.pipeline._model.parameters)

    def test_evaluate(self):
        """
        Tests the _evaluate method of the Pipeline.

        Ensures that the model is evaluated and that the prediction and
        metrics results are not None.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.pipeline._evaluate()
        self.assertIsNotNone(self.pipeline._predictions)
        self.assertIsNotNone(self.pipeline._metrics_results)
        self.assertEqual(len(self.pipeline._metrics_results), 1)

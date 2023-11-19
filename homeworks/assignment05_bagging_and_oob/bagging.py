import numpy as np


class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray) -> None:
        """
        Generate indices for every bag and store in self.indices_list list
        """
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            self.indices_list.append(
                np.random.randint(0, data_length, size=data_length)
            )
        return None

    def fit(
        self, model_constructor: callable, data: np.ndarray, target: np.ndarray
    ) -> None:
        """
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed
        to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        """
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert (
            len(set(list(map(len, self.indices_list)))) == 1
        ), "All bags should be of the same length!"
        assert list(map(len, self.indices_list))[0] == len(
            data
        ), "All bags should contain `len(data)` number of elements!"
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag, target_bag = (
                data[self.indices_list[bag]],
                target[self.indices_list[bag]],
            )
            self.models_list.append(
                model.fit(data_bag, target_bag)
            )  # store fitted models here
        if self.oob:
            self.data = data
            self.target = target

        return None

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Get average prediction for every object from passed dataset
        """
        pred = np.mean(
            [model.predict(data) for model in self.models_list], axis=0
        )
        return pred

    def _get_oob_predictions_from_every_model(self) -> None:
        """
        Generates list of lists, where list i contains predictions for
        self.data[i] object
        from all models, which have not seen this object during training phase
        """
        list_of_predictions_lists = [[] for _ in range(len(self.data))]

        for i in range(self.data.shape[0]):
            for model, indices in zip(self.models_list, self.indices_list):
                if i not in indices:
                    list_of_predictions_lists[i].append(
                        model.predict(self.data[i].reshape(1, -1))
                    )

        self.list_of_predictions_lists = np.array(
            list_of_predictions_lists, dtype=object
        )
        return None

    def _get_averaged_oob_predictions(self) -> None:
        """
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase,
        return None instead of prediction
        """
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = []
        for i in range(self.data.shape[0]):
            if len(self.list_of_predictions_lists[i]):
                self.oob_predictions.append(
                    np.mean([self.list_of_predictions_lists[i]])
                )
            else:
                self.oob_predictions.append(None)
        return None

    def OOB_score(self) -> float:
        """
        Compute mean square error for all objects,
        which have at least one prediction
        """
        self._get_averaged_oob_predictions()
        oob_score = 0.0
        for i in range(len(self.oob_predictions)):
            if self.oob_predictions[i]:
                oob_score += (self.oob_predictions[i] - self.target[i]) ** 2.0
        return oob_score**0.5

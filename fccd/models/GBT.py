import warnings
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_fit_params, has_fit_parameter
from sklearn.base import is_classifier
from sklearn.utils.fixes import delayed
from joblib import Parallel
from sklearn.multioutput import _fit_estimator
from fccd.models.baseline import merge_datasets
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor


class ValidatedMultiOutputRegressor(MultiOutputRegressor):
    
    def fit(self, X, y, sample_weight=None, **fit_params):
        # https://stackoverflow.com/questions/66785587/how-do-i-use-validation-sets-on-multioutputregressor-for-xgbregressor

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement"
                             " a fit method")

        X, y = self._validate_data(X, y,
                                   force_all_finite=False,
                                   multi_output=True, accept_sparse=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")

        if (sample_weight is not None and
                not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying estimator does not support"
                             " sample weights.")

        fit_params_validated = _check_fit_params(X, fit_params)

        [(X_test, Y_test)] = fit_params_validated.pop('eval_set')
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator, X, y[:, i], sample_weight, 
                **fit_params_validated, eval_set=[(X_test, Y_test[:, i])])
            for i in range(y.shape[1]))
        return self


class LightGBMModel(BaseEstimator):
    
    EXCLUDE_FROM_NAME = ["device", "metric", "verbosity", "model"]

    def __init__(self, max_depth=2, metric="mse", device="cpu", early_stopping_patience=0, n_estimators=100, name="LightBGM", learning_rate=0.1, n_jobs=0, feature_importance="split"):
        self.max_depth = max_depth
        self.metric = metric
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.name = name
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.verbosity = -1
        self.callbacks = []
        self.n_jobs = n_jobs
        self.feature_importance = feature_importance
        
        self.model = None
        self.name = name
        warnings.filterwarnings(action='ignore', category=UserWarning)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        gbt = LGBMRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            device=self.device,
            verbosity=self.verbosity,
            importance_type=self.feature_importance
        )
        
        if self.early_stopping_patience > 0:
            assert X_val is not None and y_val is not None, "Please provide eval_sets for early stopping"
            
            callbacks = [early_stopping(self.early_stopping_patience, verbose=False)]
            
            self.model = ValidatedMultiOutputRegressor(gbt)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks, eval_metric=self.metric)
        else:
            if X_val is not None and y_val is not None:
                X_train, y_train = merge_datasets(X_train, y_train, X_val, y_val)
            self.model = MultiOutputRegressor(gbt)
            self.model.fit(X_train, y_train)
            
        return self.model

    def predict(self, X):
        return self.model.predict(X)
    
    def __str__(self):
        assert self.name
        out = self.name
        for key, param in self.__dict__.items():
            if key in self.EXCLUDE_FROM_NAME:
                continue
            concat_key = key.replace("_", "")
            out += f"_{concat_key}{param}"
        return out
    
    
class RandomForestModel(BaseEstimator):
    
    def __init__(self, max_depth=2, n_estimators=100, name="RF", n_jobs=1):
        self.max_depth = max_depth
        self.name = name
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.model = None
        self.name = name

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        gbt = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth
        )
        
        if X_val is not None and y_val is not None:
            X_train, y_train = merge_datasets(X_train, y_train, X_val, y_val)
        self.model = MultiOutputRegressor(gbt, n_jobs=self.n_jobs)
        self.model.fit(X_train, y_train)

        return self.model

    def predict(self, X):
        return self.model.predict(X)

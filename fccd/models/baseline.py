from functools import reduce
import numpy as np
from typing import List, Tuple, Union

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import random
from sklearn.base import BaseEstimator


def hamming_distance(a: Union[List, Tuple, np.array], b: Union[List, Tuple, np.array]):
    if type(a) != np.array:
        a = np.asarray(a)
    if type(b) != np.array:
        b = np.asarray(b)
    return np.sum(a != b)


def manhattan_distance(a: Union[List, Tuple, np.array], b: Union[List, Tuple, np.array]):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.abs(a - b)


class KNNBaseline(BaseEstimator):

    def __init__(self, k=3, distance_metric=manhattan_distance, name="KNNBase", force_different_neighbour_columns=True):
        self.name = name
        self.k = k
        self.X = None
        self.y = None
        self.distance_metric = distance_metric        
        self.force_different_neighbour_columns = force_different_neighbour_columns
        

    def fit(self, X, y, X_val=None, y_val=None):
        X, y = merge_datasets(X, y, X_val, y_val)
        self.X = X
        self.y = y

    def predict(self, X):
        assert self.X is not None and self.y is not None, "Please call fit before trying to predict new samples"
        prediction = []
        
        if self.force_different_neighbour_columns:
            get_k_neighours = self._get_k_neighbours_per_column
        else:
            get_k_neighours = self._get_k_neighbours_absolute    
        
        for target_sample in np.asarray(X):
            k_neighbor_idx = get_k_neighours(target_sample)
            prediction += [self.y[k_neighbor_idx].mean(axis=0)]
        return prediction

    def _get_k_neighbours_absolute(self, target_sample):
        distances = np.asarray([
            self.distance_metric(target_sample, other_sample)
            for other_sample in self.X
        ])
        k_closest_neighbours = [self._pop_closest_sample_idx(distances) for _ in range(self.k)]
        k_closest_neighbours_idx = reduce(lambda a, b: a + b, k_closest_neighbours)
        return k_closest_neighbours_idx
    
    def _get_k_neighbours_per_column(self, target_sample):
        closest_neighbours_per_dim = []
        no_dims = len(target_sample)
        dim_indices = range(no_dims)
        
        assert no_dims >= self.k, """ k must be smaller that the number of features when using 
                                            force_different_neighbour_columns=True. 
                                            Found k={k} and no. of features={no_dims}"""
        
        k_dim_indices = random.sample(dim_indices, k=self.k)
        
        for dim in k_dim_indices:
            distances = np.asarray([
                self.distance_metric(target_sample[dim], other_sample[dim])
                for other_sample in self.X
            ])
            closest_sample_idx = self._pop_closest_sample_idx(distances)
            closest_neighbours_per_dim += [closest_sample_idx]
        
        k_closest_neighbours_idx = reduce(lambda a, b: a + b, closest_neighbours_per_dim)
        return k_closest_neighbours_idx

    def _pop_closest_sample_idx(self, distances):
        closest_sample_element_idx = np.argmin(distances)
        closest_sample = self.X[closest_sample_element_idx, :]
        
        closest_sample_idx = np.all(self.X == closest_sample, axis=1)
        distances[closest_sample_idx] = np.inf
        return closest_sample_idx
    
    def __str__(self):
        assert self.name
        out = self.name
        out += f"_k{self.k}_dist{self.distance_metric.__name__}"
        return out


class ResponseSurfaceMethod(BaseEstimator):

    def __init__(self, degree=2, name="RSM"):
        self.degree = degree
        self.feature_transform = PolynomialFeatures(degree)
        self.regressor = LinearRegression()
        self.fitted = False
        self.name = name

    def fit(self, X, y, X_val, y_val):
        X, y = merge_datasets(X, y, X_val, y_val)
        poly_x = self.feature_transform.fit_transform(X)
        self.regressor.fit(poly_x, y)
        self.fitted = True
        return self

    def predict(self, X):
        assert self.fitted, "Please call fit before trying to predict new samples"
        poly_x = self.feature_transform.transform(X)
        prediction = self.regressor.predict(poly_x)

        return prediction
    
    def __str__(self):
        assert self.name
        out = self.name
        regressor_name = self.regressor.__class__.__name__
        out += f"_{self.degree}_{regressor_name}"
        return out


def merge_datasets(x1, y1, x2, y2):
    inputs = [x1, y1, x2, y2]
    if np.all([type(input_) == pd.DataFrame for input_ in inputs]):
        x1 = pd.concat([x1, x2], axis=0)
        y1 = pd.concat([y1, y2], axis=0)
    elif np.all([type(input_) == np.ndarray for input_ in inputs]):
        x1 = np.concatenate([x1, x2], axis=0)
        y1 = np.concatenate([y1, y2], axis=0)

    return x1, y1



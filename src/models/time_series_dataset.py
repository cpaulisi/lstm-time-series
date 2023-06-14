"""This is a script for basic processing of store data

Author: Cullen Paulisick
"""

from typing import List
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class SalesDataset():

    def __init__(
            self,
            train_path: str,
            test_path: str,
            oil_path: str
        ):

        self.test_df = pd.read_csv(test_path)
        self.oil_df = pd.read_csv(oil_path)
        train_df = pd.read_csv(train_path)
        # create sample dataset for training
        # merge and reset index
        self.train_df = train_df.merge(self.oil_df, on='date', how='left').dropna().set_index("id").reset_index()

    @staticmethod
    def sales_by_store(data, family: str="AUTOMOTIVE", col_include = ['date', 'sales']):
        store_data = {}
        for store in data.store_nbr.unique():
            train_df = data.loc[data.family==family.upper()]
            store_data[store] = train_df.loc[train_df.store_nbr==store][col_include].sort_values(by="date", ascending=True).set_index("date")
            # create target by shifting data forward one period
            store_data[store]['sales_shift'] = store_data[store].sales.shift(-1)
            # drop remaining null position at last period
            store_data[store] = store_data[store].rolling(5).mean()
            store_data[store].dropna(inplace=True)
        return store_data

    @staticmethod
    def groups_of_size(arr: np.array, n: int=20):

        chunk_indices = list(range(0, arr.size, n))
        # remove first element to prevent 0-sized first element
        chunk_indices.pop(0)
        splits = np.split(arr, chunk_indices)
        # remove last element if size is non-matching
        if splits[-1].size != n:
            splits = np.delete(splits, -1, 0)
        
        return np.vstack(splits) 

    def get_loaders(
                self, 
                include_exog: bool=False,
                family: str="AUTOMOTIVE",
                seq_length: int=20, 
                batch_size: int=16,
                col_include: List=['date', 'sales', 'dcoilwtico'], 
                shuffle_loader: bool=False,
                test_size: float=0.33,
                random_state: int=42
            ):
        """Get the pytorch data loaders for training and validation

        Parse the sales data to optionally include oil sales as an exogenous variables
        """
        # get data for each store within sales family
        store_data = self.sales_by_store(data=self.train_df, family=family, col_include=col_include)
        # for each store, append to create whole dataset
        sales = np.vstack([self.groups_of_size(df.sales.values, n=seq_length) for df in store_data.values()])
        sales_shift = np.vstack([self.groups_of_size(df.sales_shift.values, n=seq_length) for df in store_data.values()])
        
        # if endogenous only, include 
        if not include_exog:
            X = sales
        else:
            oil = np.vstack([self.groups_of_size(df.dcoilwtico.values, n=seq_length) for df in store_data.values()])
            X = np.dstack([sales, oil])

        if len(X.shape) < 3:
            X = np.expand_dims(X, axis=2)
        
        # get test and train splits
        X_train, X_val, y_train, y_val = train_test_split(X, sales_shift, test_size=test_size, random_state=random_state, shuffle=False)


        # get loaders for training and validation
        train_loader = DataLoader(list(zip(X_train,y_train)), shuffle=shuffle_loader, batch_size=batch_size, drop_last=True) 
        val_loader = DataLoader(list(zip(X_val,y_val)), shuffle=shuffle_loader, batch_size=batch_size, drop_last=True) 

        return train_loader, val_loader
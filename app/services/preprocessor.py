import pandas as pd
import numpy as np
import logging
import os
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

class WeatherDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numerical_cols = []
        self.categorical_cols = []
        self.ordinal_cols = []
        self.nominal_cols = []
        self.num_imputer = SimpleImputer(strategy='median')
        self.mode_vals = {}
        self.iqr_bounds = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.expected_features = []
        self.target_col = 'RainTomorrow'
        self.is_fitted = False

    def fit(self, X, y=None):
        logger.info("Fitting WeatherDataPreprocessor...")
        df = X.copy()
        
        # 1. Handle Duplicates and Drops
        df = df.drop_duplicates()
        if 'row ID' in df.columns:
            df = df.drop(columns=['row ID'])

        # Identify columns
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in ['RainToday', self.target_col]:
            if col in self.numerical_cols:
                self.numerical_cols.remove(col)
                self.categorical_cols.append(col)

        self.ordinal_cols = [col for col in self.categorical_cols if col in ['RainToday', self.target_col]]
        self.nominal_cols = [col for col in self.categorical_cols if col not in self.ordinal_cols]

        # 2. Imputation Fit
        if self.numerical_cols:
            self.num_imputer.fit(df[self.numerical_cols])

        for col in self.categorical_cols:
            self.mode_vals[col] = df[col].mode()[0]
            
        # Apply imputation to temp dataframe to fit later stages
        if self.numerical_cols:
            df[self.numerical_cols] = self.num_imputer.transform(df[self.numerical_cols])
        for col in self.categorical_cols:
            df[col] = df[col].fillna(self.mode_vals[col])

        # 3. Outliers Fit
        for col in self.numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.iqr_bounds[col] = (lower_bound, upper_bound)
            df[col] = np.clip(df[col], lower_bound, upper_bound)

        # 4. Encoding Fit
        for col in self.ordinal_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
            df[col] = le.transform(df[col].astype(str))

        if self.nominal_cols:
            df = pd.get_dummies(df, columns=self.nominal_cols, drop_first=True)

        # 5. Scaling Fit
        if self.numerical_cols:
            self.scaler.fit(df[self.numerical_cols])
            
        self.expected_features = [col for col in df.columns if col != self.target_col]
        self.is_fitted = True
        logger.info("Preprocessor fitting complete.")
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Preprocessor MUST be fitted before calling transform!")
            
        logger.info("Transforming data...")
        df = X.copy()
        
        # 1. Drops
        if 'row ID' in df.columns:
            df = df.drop(columns=['row ID'])
            
        # Extract target if exists
        target_series = None
        if self.target_col in df.columns:
            target_series = df[self.target_col]
            
        # 2. Impute
        if self.numerical_cols:
            present_num = [c for c in self.numerical_cols if c in df.columns]
            if present_num:
                num_indices = [self.numerical_cols.index(c) for c in present_num]
                for i, col in zip(num_indices, present_num):
                    df[col] = df[col].fillna(self.num_imputer.statistics_[i])

        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.mode_vals[col])

        # 3. Outliers
        for col in self.numerical_cols:
             if col in df.columns:
                 lower, upper = self.iqr_bounds[col]
                 df[col] = np.clip(df[col], lower, upper)
            
        # 4. Encoding
        for col in self.ordinal_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
                classes = set(self.label_encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in classes else self.mode_vals[col])
                df[col] = self.label_encoders[col].transform(df[col])

        if self.nominal_cols:
            present_nom = [c for c in self.nominal_cols if c in df.columns]
            if present_nom:
                df = pd.get_dummies(df, columns=present_nom, drop_first=True)
                
        # Align features
        df_features = df.reindex(columns=self.expected_features, fill_value=0)

        # 5. Scaling
        if self.numerical_cols:
            try:
                scaled_values = self.scaler.transform(df_features[self.numerical_cols])
                df_features[self.numerical_cols] = scaled_values
            except KeyError as e:
                logger.error(f"Missing numerical columns during scaling: {e}")
                
        if target_series is not None:
            df_features[self.target_col] = df[self.target_col]

        logger.info("Transformation complete.")
        return df_features

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "preprocessor.joblib")
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessor not found at {path}")
        return joblib.load(path)

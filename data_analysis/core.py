from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Any, List
from pydantic import BaseModel, ValidationError, field_validator

import pandas as pd
import numpy as np
import io
import itertools
import scipy
import scipy.stats as ss
from scipy.stats import chi2_contingency, pointbiserialr, f_oneway, multivariate_normal
from google.colab import files
from IPython.display import display, HTML

# Sklearn preprocessing & decomposition
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import FactorAnalysis

# Plotly & Statsmodels
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

class PlottingMethods:
    """A modular class to generate Plotly figures and return them as embeddable HTML strings."""
    def display_image(self, html_str):
        display(HTML(html_str))

    def plot_bar_chart(self, data, x, y, color=None, barmode='group', title="Bar Chart"):
        fig = px.bar(data, x=x, y=y, color=color, barmode=barmode, title=title)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def plot_pie_chart(self, data, names, values, hole=0.0, title="Pie Chart"):
        fig = px.pie(data, names=names, values=values, hole=hole, title=title)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def plot_histogram(self, data, x, bins=None, title="Histogram"):
        fig = go.Figure(data=[go.Histogram(x=data[x], xbins=dict(start=bins[0], end=bins[-1], size=(bins[1]-bins[0])) if bins else None)])
        fig.update_layout(title=title, xaxis_title=x, yaxis_title="Count")
        return fig.to_html(full_html=False, include_plotlyjs='cdn')


class DataInspector(PlottingMethods):
    """Robust Data Sanitization & Exploration Engine."""
    def __init__(self):
        self.df = None
        self.garbage_strings = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', '']

    def upload_data(self):
        """Prompts user to upload a CSV file in Colab."""
        print("Please upload your CSV file...")
        uploaded = files.upload()
        if not uploaded: return
            
        file_name = list(uploaded.keys())[0]
        self.df = pd.read_csv(io.BytesIO(uploaded[file_name]), na_values=self.garbage_strings)
        
        for col in self.df.columns:
            converted = pd.to_numeric(self.df[col], errors='coerce')
            if not converted.isna().all() or self.df[col].isna().all():
                self.df[col] = converted
        print(f"\nSuccessfully loaded '{file_name}' and sanitized basic nulls.")

    def summary(self):
        if self.df is None: return "No data loaded."
        print(f"Dimensions: {self.df.shape[0]} Rows, {self.df.shape[1]} Columns\n")
        print("Data Types Breakdown:")
        print(self.df.dtypes.value_counts(), "\n")
        print("First 5 Rows Preview:")
        display(self.df.head())

    def handle_missing_values(self, strategy='median', constant_val=None):
        if self.df is None: return
        for col in self.df.columns:
            if self.df[col].isna().sum() > 0:
                if strategy == 'constant' and constant_val is not None:
                    self.df[col] = self.df[col].fillna(constant_val)
                elif pd.api.types.is_numeric_dtype(self.df[col]):
                    if strategy == 'mean': self.df[col] = self.df[col].fillna(self.df[col].mean())
                    elif strategy == 'median': self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        print(f"Missing values handled using '{strategy}' strategy.")

    def remove_duplicates(self):
        initial_len = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {initial_len - len(self.df)} duplicate rows.")

    def handle_outliers(self, target_columns=None, action='drop'):
        if target_columns is None: target_columns = self.df.select_dtypes(include=[np.number]).columns
        initial_len = len(self.df)
        for col in target_columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                if action == 'drop':
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        print(f"Handled outliers in {target_columns}. Removed {initial_len - len(self.df)} rows.")

    def delete_columns(self, cols):
        self.df.drop(columns=cols, inplace=True, errors='ignore')
        print(f"Deleted columns: {cols}")

    def extract_normalized_numeric_data(self, method='standard', exclude_cols=None):
        exclude_cols = exclude_cols or []
        num_df = self.df.select_dtypes(include=[np.number]).drop(columns=exclude_cols, errors='ignore')
        if method == 'minmax': scaler = MinMaxScaler()
        elif method == 'standard': scaler = StandardScaler()
        elif method == 'robust': scaler = RobustScaler()
        else: return num_df
        return pd.DataFrame(scaler.fit_transform(num_df), columns=num_df.columns, index=self.df.index)

    def extract_normalized_categorical_data(self, method='onehot'):
        cat_df = self.df.select_dtypes(exclude=[np.number])
        if cat_df.empty: return cat_df
        if method == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            cols = encoder.fit(cat_df).get_feature_names_out(cat_df.columns)
            return pd.DataFrame(encoder.transform(cat_df), columns=cols, index=self.df.index)
        elif method == 'ordinal':
            return pd.DataFrame(OrdinalEncoder().fit_transform(cat_df), columns=cat_df.columns, index=self.df.index)
        
    def create_normalized_data_df(self, num_method='standard', cat_method='onehot', preserve_cols=None):
        preserve_cols = preserve_cols or []
        num_norm = self.extract_normalized_numeric_data(method=num_method, exclude_cols=preserve_cols)
        cat_norm = self.extract_normalized_categorical_data(method=cat_method)
        preserved = self.df[preserve_cols] if preserve_cols else pd.DataFrame(index=self.df.index)
        return pd.concat([preserved, num_norm, cat_norm], axis=1)

    def plot_numerical(self, columns):
        for col in columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                fig = make_subplots(rows=1, cols=3, subplot_titles=("Distribution (Violin)", "Index Scatter", "Histogram"))
                fig.add_trace(go.Violin(x=self.df[col], name=col, box_visible=True, line_color='blue'), row=1, col=1)
                fig.add_trace(go.Scatter(y=self.df[col], mode='markers', marker=dict(color='orange', opacity=0.6)), row=1, col=2)
                fig.add_trace(go.Histogram(x=self.df[col], marker_color='green'), row=1, col=3)
                fig.update_layout(title_text=f"Univariate Analysis: {col}", showlegend=False, height=400)
                fig.show()

    def plot_relationship(self, x, y):
        x_num, y_num = pd.api.types.is_numeric_dtype(self.df[x]), pd.api.types.is_numeric_dtype(self.df[y])
        if x_num and y_num: fig = px.scatter(self.df, x=x, y=y, trendline='ols', title=f"Scatter: {x} vs {y}")
        elif not x_num and y_num: fig = px.box(self.df, x=x, y=y, points="all", title=f"Boxplot: {x} vs {y}")
        elif x_num and not y_num: fig = px.box(self.df, x=y, y=x, points="all", title=f"Boxplot: {y} vs {x}")
        else:
            counts = self.df.groupby([x, y]).size().reset_index(name='Count')
            fig = px.bar(counts, x=x, y='Count', color=y, barmode='group', title=f"Grouped Bar: {x} vs {y}")
        fig.show()

    def plot_categorical_frequency(self, col):
        if not pd.api.types.is_numeric_dtype(self.df[col]):
            counts = self.df[col].value_counts().reset_index()
            counts.columns = [col, 'Count']
            counts['Percentage'] = (counts['Count'] / counts['Count'].sum() * 100).round(2).astype(str) + '%'
            fig = px.bar(counts, x=col, y='Count', text='Percentage', title=f"Frequency of {col}")
            fig.show()

    def _cramers_v(self, x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        return np.sqrt(phi2 / min(k-1, r-1)) if min(k-1, r-1) > 0 else 0.0

    def _correlation_ratio(self, categories, measurements):
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat) + 1
        y_avg_array, n_array = np.zeros(cat_num), np.zeros(cat_num)
        for i in range(cat_num):
            cat_measures = measurements[fcat == i]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures) if len(cat_measures) > 0 else 0
        y_total_avg = np.average(measurements)
        numerator = np.sum(n_array * np.square(y_avg_array - y_total_avg))
        denominator = np.sum(np.square(measurements - y_total_avg))
        return np.sqrt(numerator / denominator) if denominator != 0 else 0.0

    def plot_all_associations_heatmap(self):
        cols = self.df.columns
        matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
        for col1, col2 in itertools.combinations_with_replacement(cols, 2):
            is_num1, is_num2 = pd.api.types.is_numeric_dtype(self.df[col1]), pd.api.types.is_numeric_dtype(self.df[col2])
            valid_idx = self.df[[col1, col2]].dropna().index
            v1, v2 = self.df.loc[valid_idx, col1], self.df.loc[valid_idx, col2]
            
            if len(v1) < 2: val = 0.0
            elif is_num1 and is_num2: val = v1.corr(v2) 
            elif not is_num1 and not is_num2: val = self._cramers_v(v1, v2)
            else: val = self._correlation_ratio(v2, v1) if is_num1 else self._correlation_ratio(v1, v2)
                
            matrix.loc[col1, col2] = matrix.loc[col2, col1] = val
            
        fig = px.imshow(matrix, text_auto=".2f", color_continuous_scale='RdBu_r', 
                        title="Unified Association Heatmap (Pearson / Cramér's V / Eta)")
        fig.show()

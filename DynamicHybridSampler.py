import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.utils import resample
from sklearn.metrics import silhouette_score
from scipy.stats import variation, skew, entropy
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

class DynamicHybridSampler:
    def __init__(self, target_sample_size=None, hierarchy_cols=None, eps=0.5, min_samples=5, verbose=True):
        self.target_sample_size = target_sample_size
        self.hierarchy_cols = hierarchy_cols or ['region', 'household_id']
        self.eps = eps
        self.min_samples = min_samples
        self.verbose = verbose
        self.explanations = []
        self.sampled_data = None

    def log(self, message):
        if self.verbose:
            logger.info(message)
        self.explanations.append(message)

    def fit(self, df):
        self.original_data = df.copy()
        self.data = df.copy()
        self.sample_pool = pd.DataFrame()
        if not self.target_sample_size:
            self.target_sample_size = min(100, len(df))
        self._profile_data()
        self._rare_category_sampler()
        self._stratified_sampler()
        self._temporal_or_systematic_sampler()
        self._multistage_sampler()
        self._encode_categoricals()
        self._dbscan_sampler()
        self._skew_sampler()
        self._finalize_sample()
        self._validate_sample()

    def _profile_data(self):
        self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.datetime_cols = self.data.select_dtypes(include=['datetime64']).columns.tolist()
        self.col_stats = {}

        self.log("Profiling data...")
        for col in self.data.columns:
            if col in self.categorical_cols:
                freqs = self.data[col].value_counts()
                self.col_stats[col] = {"type": "categorical", "value_counts": freqs}
            elif col in self.numerical_cols:
                self.col_stats[col] = {
                    "type": "numerical",
                    "mean": self.data[col].mean(),
                    "std": self.data[col].std(),
                    "skew": skew(self.data[col].dropna()),
                    "var_coeff": variation(self.data[col].dropna())
                }
            elif col in self.datetime_cols:
                self.col_stats[col] = {"type": "temporal", "range": self.data[col].max() - self.data[col].min()}

    def _rare_category_sampler(self):
        self.log("Sampling rare categories from categorical columns...")
        for col in self.categorical_cols:
            freqs = self.col_stats[col]["value_counts"]
            threshold = max(1, int(len(freqs) * 0.1))
            rare_values = freqs[freqs <= threshold].index.tolist()
            rare_samples = self.data[self.data[col].isin(rare_values)]
            if not rare_samples.empty:
                self.sample_pool = pd.concat([self.sample_pool, rare_samples])
                self.log(f"Included {len(rare_samples)} rare samples from column '{col}'")

    def _select_strat_col(self):
        if not self.categorical_cols:
            return None
        entropies = {col: entropy(self.data[col].value_counts(normalize=True)) for col in self.categorical_cols}
        return max(entropies, key=entropies.get)

    def _stratified_sampler(self):
        strat_col = self._select_strat_col()
        if not strat_col:
            return
        self.log(f"Applying stratified sampling on '{strat_col}'...")
        strata = self.data[strat_col].value_counts(normalize=True)
        for label, frac in strata.items():
            group = self.data[self.data[strat_col] == label]
            n_samples = int(frac * self.target_sample_size / 2)
            if n_samples > 0 and not group.empty:
                group_sample = resample(group, replace=False, n_samples=min(n_samples, len(group)), random_state=42)
                self.sample_pool = pd.concat([self.sample_pool, group_sample])
        self.log(f"Stratified sampling on '{strat_col}' completed")

    def _temporal_or_systematic_sampler(self):
        if self.datetime_cols:
            self.log("Temporal data detected, applying time-bucket sampling...")
            time_col = self.datetime_cols[0]
            self.data = self.data.sort_values(by=time_col)
            time_bins = pd.qcut(self.data[time_col], q=5, duplicates='drop')
            for bin_label in time_bins.unique():
                bin_data = self.data[time_bins == bin_label]
                n = min(2, len(bin_data))
                if n > 0:
                    sample = resample(bin_data, n_samples=n)
                    self.sample_pool = pd.concat([self.sample_pool, sample])
                    self.log(f"Sampled {n} instances from time bin {bin_label}")
        else:
            self.log("No temporal data found, applying systematic sampling...")
            if len(self.data) >= self.target_sample_size:
                interval = max(1, len(self.data) // self.target_sample_size)
                indices = list(range(0, len(self.data), interval))[:self.target_sample_size]
                sys_sample = self.data.iloc[indices]
                self.sample_pool = pd.concat([self.sample_pool, sys_sample])
                self.log(f"Systematic sampling applied with interval {interval}")

    def _multistage_sampler(self):
        self.log("Checking for multi-stage sampling opportunities...")
        if all(col in self.data.columns for col in self.hierarchy_cols):
            top_col, sub_col = self.hierarchy_cols
            regions = self.data[top_col].unique()
            for region in regions:
                households = self.data[self.data[top_col] == region][sub_col].unique()
                sampled_households = np.random.choice(households, min(2, len(households)), replace=False)
                for hh in sampled_households:
                    hh_data = self.data[(self.data[top_col] == region) & (self.data[sub_col] == hh)]
                    if not hh_data.empty:
                        self.sample_pool = pd.concat([self.sample_pool, hh_data.sample(min(1, len(hh_data)))])
            self.log("Multistage sampling applied")
        else:
            self.log("Multistage sampling skipped due to missing hierarchical fields")

    def _encode_categoricals(self):
        self.log("Encoding categorical columns for clustering...")
        self.encoded_data = self.data.copy()
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.encoded_data[col] = le.fit_transform(self.encoded_data[col].astype(str))

    def _dbscan_sampler(self):
        if not self.numerical_cols:
            return
        self.log("Running DBSCAN clustering...")
        clustering_data = self.encoded_data[self.numerical_cols + self.categorical_cols] if self.categorical_cols else self.encoded_data[self.numerical_cols]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)
        pca = PCA(n_components=min(5, scaled_data.shape[1]))
        reduced_data = pca.fit_transform(scaled_data)
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(reduced_data)
        labels = clustering.labels_
        self.data['dbscan_label'] = labels

        if len(set(labels)) > 1:
            try:
                score = silhouette_score(reduced_data, labels)
                self.log(f"Silhouette Score for DBSCAN: {score:.2f}")
            except:
                pass
        else:
            self.log("DBSCAN clustering did not produce multiple clusters")

        for label in np.unique(labels):
            cluster_data = self.data[self.data['dbscan_label'] == label]
            size = min(3, len(cluster_data))
            if size > 0:
                sample = resample(cluster_data, n_samples=size, replace=False)
                self.sample_pool = pd.concat([self.sample_pool, sample])
                label_type = "noise (outlier)" if label == -1 else f"cluster {label}"
                self.log(f"Included {size} samples from {label_type}")

    def _skew_sampler(self):
        self.log("Sampling from skewed numerical features using refined PPS...")
        for col in self.numerical_cols:
            if abs(self.col_stats[col]['skew']) > 1:
                deviations = abs(self.data[col] - self.data[col].median())
                weights = deviations + 1e-6
                weights /= weights.sum()
                if len(weights) > 0:
                    selected = self.data.sample(n=min(3, len(self.data)), weights=weights)
                    self.sample_pool = pd.concat([self.sample_pool, selected])
                    self.log(f"RPPS sampling on skewed column '{col}'")

    def _finalize_sample(self):
        self.sample_pool.drop_duplicates(inplace=True)
        n = min(self.target_sample_size, len(self.sample_pool))
        if n == 0:
            self.sampled_data = pd.DataFrame()
            self.log("Warning: no samples collected.")
        else:
            final_sample = resample(self.sample_pool, n_samples=n, random_state=42)
            self.sampled_data = final_sample.reset_index(drop=True)
            self.log(f"Final sample size: {len(self.sampled_data)}")

    def _validate_sample(self):
        if len(self.sampled_data) < self.target_sample_size:
            self.log("Sample size is below the target.")
        else:
            self.log(f"Sample size achieved: {len(self.sampled_data)}")

    def get_sample(self):
        return self.sampled_data

    def get_explanation(self):
        return self.explanations

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, IsolationForest # Added IsolationForest
from scipy.stats import variation, skew
from collections import defaultdict
import logging
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
# Ignore specific warnings from sklearn clustering if needed
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._dbscan')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.ensemble._forest') # For n_estimators default change

class DynamicHybridSampler:
    """
    Generates a representative sample from a DataFrame using a hybrid of techniques.

    Combines preprocessing (imputation, encoding) with multiple sampling strategies
    including rare category, stratified, temporal/systematic, multi-stage,
    clustering-based, feature importance-based, skew-based, and optional outlier sampling.

    Attributes:
        target_sample_size (int): The desired final sample size.
        use_onehot (bool): Whether to use OneHotEncoder (True) or LabelEncoder (False).
        imputation_strategy (str): Method for numerical imputation ('mean', 'median', 'mode').
                                    Categorical imputation always uses mode.
        stratification_col (str): Name of the column to use for stratified sampling.
                                   If None, the first categorical column is used.
        kmeans_k (int): Number of clusters for KMeans fallback if DBSCAN fails.
        use_outlier_sampler (bool): Whether to include an explicit outlier sampling step.
        explanations (list): Log of actions taken during the sampling process.
        sampled_data (pd.DataFrame): The final generated sample.
    """
    def __init__(self,
                 target_sample_size=100,
                 use_onehot=False,
                 imputation_strategy='median', # Default changed to median
                 stratification_col=None,
                 kmeans_k=5, # Increased default K
                 use_outlier_sampler=True): # Outlier sampling enabled by default
        """Initializes the DynamicHybridSampler."""
        if imputation_strategy not in ['mean', 'median', 'mode']:
            raise ValueError("imputation_strategy must be 'mean', 'median', or 'mode'")

        self.target_sample_size = target_sample_size
        self.use_onehot = use_onehot
        self.imputation_strategy = imputation_strategy
        self.stratification_col = stratification_col
        self.kmeans_k = kmeans_k
        self.use_outlier_sampler = use_outlier_sampler

        self.explanations = []
        self.sampled_data = None
        self.original_data = None
        self.data = None # Working copy
        self.encoded_data = None # Encoded working copy
        self.sample_pool = None
        self.categorical_cols = []
        self.numerical_cols = []
        self.datetime_cols = []
        self.col_stats = {}


    def log(self, message, level='info'):
        """Logs a message and adds it to explanations."""
        if level == 'info':
            logger.info(message)
        elif level == 'warning':
            logger.warning(message)
        self.explanations.append(message)

    def fit(self, df):
        """
        Fits the sampler to the data and generates the sample.

        Args:
            df (pd.DataFrame): The input DataFrame to sample from.

        Returns:
            self: The fitted sampler instance.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if df.empty:
            self.log("Input DataFrame is empty. No sampling performed.", level='warning')
            # Return an empty dataframe with original columns if possible
            cols = df.columns if df is not None else []
            self.sampled_data = pd.DataFrame(columns=cols)
            return self

        self.original_data = df.copy()
        self.data = df.copy()
        # Ensure sample_pool is initialized fresh for each fit
        self.sample_pool = pd.DataFrame(columns=self.data.columns)

        # --- Pipeline ---
        self._handle_missing_data()
        self._profile_data()
        self._rare_category_sampler()
        self._stratified_sampler()
        self._temporal_or_systematic_sampler()
        self._multistage_sampler()
        self._encode_categoricals() # Prepare for steps needing numerical data

        # Samplers requiring encoded/numerical data
        if self.use_outlier_sampler:
            self._outlier_sampler()
        self._feature_importance_sampler()
        self._adaptive_clustering_sampler()
        self._skew_sampler() # Uses original numerical data, run after profiling

        self._finalize_sample()
        self.log(f"Sampling process complete. Final sample size: {len(self.sampled_data) if self.sampled_data is not None else 0}")
        return self

    def _handle_missing_data(self):
        """Handles missing data based on the chosen strategy."""
        self.log(f"Handling missing data using strategy: {self.imputation_strategy} for numeric.")
        cols_imputed = []
        for col in self.data.columns:
            if self.data[col].isnull().any():
                cols_imputed.append(col)
                # Check if column is numeric using pandas API
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    fill_value = 0 # Default fill value
                    if self.imputation_strategy == 'mean':
                        fill_value = self.data[col].mean()
                    elif self.imputation_strategy == 'median':
                        fill_value = self.data[col].median()
                    elif self.imputation_strategy == 'mode':
                         # mode() returns a Series, take the first element if it exists
                         mode_val = self.data[col].mode()
                         if not mode_val.empty:
                              fill_value = mode_val[0]
                         else: # Handle empty mode case (e.g., all NaN)
                              fill_value = 0 # Or some other default
                              self.log(f"Warning: Mode could not be determined for numeric column '{col}'. Filling with 0.", level='warning')
                    # Check if fill_value is NaN (can happen if all values were NaN)
                    if pd.isna(fill_value):
                         fill_value = 0 # Replace NaN fill_value with 0
                         self.log(f"Warning: Calculated fill value for numeric column '{col}' was NaN. Filling with 0.", level='warning')

                    self.data[col].fillna(fill_value, inplace=True)

                else: # Categorical or other non-numeric types
                    mode_val = self.data[col].mode()
                    fill_value = 'Unknown' # Default fallback
                    if not mode_val.empty:
                         fill_value = mode_val[0]
                    else:
                         self.log(f"Warning: Mode could not be determined for non-numeric column '{col}'. Filling with 'Unknown'.", level='warning')
                    self.data[col].fillna(fill_value, inplace=True)

        if cols_imputed:
             self.log(f"Imputed missing values in columns: {', '.join(cols_imputed)}")
        else:
             self.log("No missing values found or handled.")


    def _profile_data(self):
        """Profiles the data to identify column types and basic statistics."""
        self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        # Ensure boolean columns are not treated as numeric for stats like skew if not desired
        self.numerical_cols = self.data.select_dtypes(include=np.number, exclude='boolean').columns.tolist()
        self.datetime_cols = self.data.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
        self.col_stats = {}

        self.log("Profiling data...")
        if not self.numerical_cols and not self.categorical_cols and not self.datetime_cols:
             self.log("Warning: No numerical, categorical, or datetime columns identified.", level='warning')
             return

        for col in self.data.columns:
            # Ensure column exists before accessing stats
            if col not in self.data: continue

            if col in self.categorical_cols:
                # Check for NaNs introduced by operations if any
                freqs = self.data[col].astype(str).value_counts() # Handle potential mixed types / NaNs
                self.col_stats[col] = {"type": "categorical", "value_counts": freqs, "num_unique": self.data[col].nunique()}
            elif col in self.numerical_cols:
                # Drop NaNs for stats calculation if any persist (shouldn't after imputation)
                valid_data = self.data[col].dropna()
                if not valid_data.empty:
                    mean_val = valid_data.mean()
                    self.col_stats[col] = {
                        "type": "numerical",
                        "mean": mean_val,
                        "std": valid_data.std(),
                        "skew": skew(valid_data) if len(valid_data) > 1 else 0, # Skew needs > 1 point
                        "var_coeff": variation(valid_data) if mean_val != 0 and len(valid_data) > 1 else 0
                    }
                else:
                     self.col_stats[col] = {"type": "numerical", "mean": np.nan, "std": np.nan, "skew": np.nan, "var_coeff": np.nan}

            elif col in self.datetime_cols:
                 valid_data = self.data[col].dropna()
                 if not valid_data.empty and len(valid_data) > 0:
                     min_val = valid_data.min()
                     max_val = valid_data.max()
                     # Ensure comparison is valid if only one date exists
                     time_range = max_val - min_val if len(valid_data) > 1 else pd.Timedelta(0)
                     self.col_stats[col] = {"type": "temporal", "range": time_range, "min": min_val, "max": max_val}
                 else:
                     self.col_stats[col] = {"type": "temporal", "range": pd.NaT, "min": pd.NaT, "max": pd.NaT}
        self.log(f"Identified: {len(self.numerical_cols)} numeric, {len(self.categorical_cols)} categoric, {len(self.datetime_cols)} datetime columns.")


    def _add_to_pool(self, samples_to_add):
        """Safely adds samples to the pool, handling potential empty inputs."""
        if samples_to_add is not None and not samples_to_add.empty:
            # Ensure columns match, important if samples come from intermediate dfs
            # Align columns, adding missing ones with NaN if necessary
            samples_aligned = samples_to_add.reindex(columns=self.data.columns)
            try:
                self.sample_pool = pd.concat([self.sample_pool, samples_aligned], ignore_index=True)
            except Exception as e:
                 self.log(f"Error concatenating samples to pool: {e}", level='warning')
            # Simple deduplication within this function might be too slow. Deduplicate at the end.


    def _rare_category_sampler(self):
        """Samples rows containing rare categories."""
        if not self.categorical_cols:
            self.log("Skipping rare category sampling: No categorical columns.")
            return

        self.log("Sampling rare categories from categorical columns...")
        # Dynamic threshold based on data size, ensure at least 1
        min_threshold = max(1, int(len(self.data) * 0.01)) # e.g., categories with < 1% frequency
        self.log(f"Rare category threshold set to <= {min_threshold} occurrences.")
        total_rare_samples = 0

        for col in self.categorical_cols:
            if col not in self.col_stats or self.col_stats[col]['type'] != 'categorical':
                 continue # Skip if profiling failed for this col

            freqs = self.col_stats[col]["value_counts"]
            rare_values = freqs[freqs <= min_threshold].index.tolist()

            if rare_values:
                # Handle potential type mismatches if column was mixed
                try:
                    # Convert both sides to string for robust comparison
                    rare_values_str = list(map(str, rare_values))
                    rare_samples = self.data[self.data[col].astype(str).isin(rare_values_str)]
                except Exception as e:
                    self.log(f"Could not perform filtering for rare values in column '{col}'. Error: {e}", level='warning')
                    continue

                if not rare_samples.empty:
                    num_added = len(rare_samples)
                    self._add_to_pool(rare_samples)
                    total_rare_samples += num_added # Count before potential deduplication
                    # Limit logging if many rare values
                    rare_vals_display = rare_values[:5] + ['...'] if len(rare_values) > 5 else rare_values
                    # self.log(f"Identified {len(rare_values)} rare values ({rare_vals_display}) in '{col}'. Added {num_added} samples.")

        if total_rare_samples > 0:
             self.log(f"Rare category sampling added {total_rare_samples} samples in total.")
        else:
            self.log("No rare categories found meeting the threshold.")


    def _stratified_sampler(self):
        """Performs stratified sampling based on a specified or chosen column."""
        if not self.categorical_cols:
            self.log("Skipping stratified sampling: No categorical columns.")
            return

        # Determine stratification column
        strat_col_actual = self.stratification_col
        if strat_col_actual is None or strat_col_actual not in self.categorical_cols:
            if strat_col_actual is not None: # User specified an invalid one
                 self.log(f"Specified stratification column '{strat_col_actual}' not found or not categorical. Using first categorical column instead.", level='warning')
            # Fallback to first categorical column with reasonable number of unique values
            candidate_cols = [c for c in self.categorical_cols if self.data[c].nunique() > 1 and self.data[c].nunique() < len(self.data) * 0.5]
            if not candidate_cols:
                 self.log("Could not find suitable column for stratification. Skipping.", level='warning')
                 return
            strat_col_actual = candidate_cols[0]
            self.log(f"Using '{strat_col_actual}' for stratification (low-cardinality categorical).")
        else:
             self.log(f"Applying stratified sampling based on specified column: '{strat_col_actual}'")

        # Perform stratification
        try:
            # Handle potential NaNs in stratification column - treat as separate stratum?
            # For simplicity, we'll rely on prior imputation or dropna here.
            valid_data_strat = self.data.dropna(subset=[strat_col_actual])
            if valid_data_strat.empty:
                 self.log(f"No valid data for stratification column '{strat_col_actual}' after dropping NaNs. Skipping.", level='warning')
                 return

            strata_proportions = valid_data_strat[strat_col_actual].value_counts(normalize=True)
            # Allocate a significant fraction of target size for stratification
            required_samples = max(10, self.target_sample_size // 3)
            self.log(f"Aiming for approx {required_samples} stratified samples total.")
            total_stratified_samples = 0

            for label, frac in strata_proportions.items():
                # Ensure at least 1 sample per stratum if fraction > 0 and possible
                n_samples_stratum = max(1, int(np.round(frac * required_samples)))
                group = valid_data_strat[valid_data_strat[strat_col_actual] == label]

                # Sample min(desired, available) without replacement
                n_to_sample = min(n_samples_stratum, len(group))
                if n_to_sample > 0:
                    group_sample = resample(group,
                                            replace=False,
                                            n_samples=n_to_sample,
                                            random_state=42)
                    self._add_to_pool(group_sample)
                    total_stratified_samples += len(group_sample)

            self.log(f"Stratified sampling on '{strat_col_actual}' added {total_stratified_samples} samples.")
        except Exception as e:
             self.log(f"Error during stratified sampling on '{strat_col_actual}': {e}", level='warning')


    def _temporal_or_systematic_sampler(self):
        """Applies time-bucket sampling if temporal data exists, otherwise systematic."""
        if self.datetime_cols:
            time_col = self.datetime_cols[0] # Assume first is relevant
            if time_col not in self.data or self.data[time_col].isnull().all():
                self.log(f"Temporal column '{time_col}' not found or contains only null values. Skipping temporal sampling.", level='warning')
                # Optionally fallback to systematic here?
                # self._systematic_sampler() # Call helper
                return

            self.log(f"Temporal data detected ('{time_col}'), applying time-bucket sampling...")
            try:
                # Sort by time; handle NaNs by dropping them for binning/sampling
                sorted_data = self.data.dropna(subset=[time_col]).sort_values(by=time_col)
                if sorted_data.empty:
                    self.log(f"No valid temporal data in '{time_col}' after dropping NaNs.", level='warning')
                    return

                # Aim for ~5 bins, adjust if fewer unique times
                n_unique_times = sorted_data[time_col].nunique()
                n_bins = min(5, n_unique_times)

                if n_bins <= 1:
                    self.log(f"Not enough unique time points ({n_unique_times}) in '{time_col}' for effective binning. Sampling randomly.", level='warning')
                    # Sample a small number randomly from the available points
                    n_random_time_samples = min(5, len(sorted_data))
                    if n_random_time_samples > 0:
                        sample = sorted_data.sample(n=n_random_time_samples, random_state=42)
                        self._add_to_pool(sample)
                        self.log(f"Sampled {len(sample)} instances randomly due to insufficient time variance.")
                    return

                # Use qcut for quantile-based bins
                try:
                     time_bins = pd.qcut(sorted_data[time_col], q=n_bins, duplicates='drop', labels=False)
                except ValueError as ve:
                     # Fallback if qcut fails (e.g., too many identical timestamps)
                     self.log(f"qcut failed for temporal binning ('{ve}'). Trying equal interval binning.", level='warning')
                     time_bins = pd.cut(sorted_data[time_col].rank(method='first'), bins=n_bins, labels=False) # Bin by rank

                # Determine samples per bin - distribute a small portion of target size
                samples_per_bin = max(1, (self.target_sample_size // 10) // n_bins)
                total_temporal_samples = 0

                for bin_label in np.unique(time_bins):
                    bin_data = sorted_data[time_bins == bin_label]
                    n = min(samples_per_bin, len(bin_data))
                    if n > 0:
                        sample = resample(bin_data, n_samples=n, replace=False, random_state=42)
                        self._add_to_pool(sample)
                        total_temporal_samples += len(sample)
                        # self.log(f"Sampled {n} instances from time bin {bin_label}")
                self.log(f"Temporal sampling added {total_temporal_samples} samples across {n_bins} bins.")

            except Exception as e:
                 self.log(f"Error during temporal sampling on '{time_col}': {e}", level='warning')

        else: # No datetime columns, use Systematic Sampling
            self._systematic_sampler()

    def _systematic_sampler(self):
        """Performs systematic sampling (helper method)."""
        self.log("No temporal data found, evaluating systematic sampling...")
        n_data = len(self.data)
        # Apply systematic sampling if data is significantly larger than target
        if n_data >= self.target_sample_size * 1.5:
             # Aim for a fraction of the target size via systematic sampling
             required_samples = max(10, self.target_sample_size // 5)
             # Ensure interval is at least 1
             interval = max(1, n_data // required_samples)
             # Start from a random point within the first interval for better randomness
             start_index = np.random.randint(0, interval)
             indices = np.arange(start_index, n_data, interval)
             if len(indices) > 0:
                 sys_sample = self.data.iloc[indices]
                 self._add_to_pool(sys_sample)
                 self.log(f"Systematic sampling applied from index {start_index} with interval {interval}, adding {len(sys_sample)} samples.")
             else:
                 self.log("Systematic sampling resulted in 0 indices. Skipping.", level='warning')
        else:
             self.log(f"Systematic sampling not applied. Dataset size ({n_data}) vs target ({self.target_sample_size}).")


    def _multistage_sampler(self):
        """Performs two-stage sampling if 'region' and 'household_id' columns exist."""
        self.log("Checking for multi-stage sampling opportunities ('region', 'household_id')...")
        # Check if columns exist *and* have more than one unique value to be meaningful
        region_col = 'region' # Make configurable?
        hh_col = 'household_id'
        if (region_col in self.data.columns and self.data[region_col].nunique() > 1 and
            hh_col in self.data.columns and self.data[hh_col].nunique() > 1):

            regions = self.data[region_col].unique()
            total_multistage_samples = 0
            # Sample from a subset of regions if there are many (e.g., max 5)
            n_regions_to_sample = min(len(regions), 5)
            regions_to_sample = np.random.choice(regions, size=n_regions_to_sample, replace=False)
            self.log(f"Found hierarchical columns. Sampling from {len(regions_to_sample)} regions.")

            # Estimate samples per region needed
            samples_per_region = max(1, (self.target_sample_size // 10) // n_regions_to_sample) if n_regions_to_sample > 0 else 1


            for region in regions_to_sample:
                region_data = self.data[self.data[region_col] == region]
                households = region_data[hh_col].unique()
                if len(households) > 0:
                    # Sample a few households per selected region (e.g., max 3)
                    n_households_to_sample = min(len(households), 3)
                    sampled_households = np.random.choice(households, size=n_households_to_sample, replace=False)

                    # Estimate samples per household
                    samples_per_hh = max(1, samples_per_region // n_households_to_sample) if n_households_to_sample > 0 else 1


                    for hh in sampled_households:
                        hh_data = region_data[region_data[hh_col] == hh]
                        # Sample one or more individuals per selected household
                        n_individuals_to_sample = min(len(hh_data), samples_per_hh)
                        if n_individuals_to_sample > 0:
                             sample = hh_data.sample(n=n_individuals_to_sample, random_state=42)
                             self._add_to_pool(sample)
                             total_multistage_samples += len(sample)

            self.log(f"Multistage sampling using '{region_col}' -> '{hh_col}' added {total_multistage_samples} samples.")
        else:
            self.log(f"Multistage sampling skipped: required columns ('{region_col}', '{hh_col}') not found or lack variance.")

    def _encode_categoricals(self):
        """Encodes categorical columns using the specified strategy."""
        # Include boolean columns for encoding if needed, e.g., if use_onehot is True
        cols_to_encode = self.data.select_dtypes(include=['object', 'category', 'boolean']).columns.tolist()

        if not cols_to_encode:
             self.log("No categorical or boolean columns to encode.")
             self.encoded_data = self.data.copy() # Still need encoded_data copy
             # Drop datetime columns from encoded data if they exist
             if hasattr(self, 'datetime_cols'):
                 self.encoded_data = self.encoded_data.drop(columns=self.datetime_cols, errors='ignore')
             return

        encoding_type = 'OneHotEncoder' if self.use_onehot else 'LabelEncoder'
        self.log(f"Encoding categorical/boolean columns using {encoding_type}...")
        # Work on a copy to keep self.data with original categoricals
        self.encoded_data = self.data.copy()

        if self.use_onehot:
            try:
                # Consider handle_unknown='ignore' if unseen categories might appear later
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_part = encoder.fit_transform(self.encoded_data[cols_to_encode])
                encoded_df = pd.DataFrame(encoded_part, columns=encoder.get_feature_names_out(cols_to_encode), index=self.encoded_data.index)

                # Drop original columns and concatenate encoded ones
                self.encoded_data = self.encoded_data.drop(columns=cols_to_encode)
                self.encoded_data = pd.concat([self.encoded_data, encoded_df], axis=1)
                self.log(f"Applied OneHotEncoder. New shape: {self.encoded_data.shape}")
            except Exception as e:
                self.log(f"Error during OneHotEncoding: {e}. Falling back to Label Encoding.", level='warning')
                self.use_onehot = False # Force fallback for this run

        # Apply Label Encoding if use_onehot is False or if OneHot failed
        if not self.use_onehot:
            # self.log("Applying LabelEncoder...") # Logged above
            encoders = {}
            for col in cols_to_encode:
                 if col in self.encoded_data.columns: # Check if still exists
                    le = LabelEncoder()
                    # Convert to string to handle mixed types robustly before encoding
                    self.encoded_data[col] = le.fit_transform(self.encoded_data[col].astype(str))
                    encoders[col] = le # Store encoder if needed later
            self.log("LabelEncoder applied.")

        # Drop original datetime columns from encoded data if they exist and cause issues
        if hasattr(self, 'datetime_cols'):
            self.encoded_data = self.encoded_data.drop(columns=self.datetime_cols, errors='ignore')


    def _get_numeric_data_for_ml(self):
         """Selects purely numerical columns from encoded_data, safe for ML."""
         if self.encoded_data is None:
              self.log("Encoded data not available for ML steps.", level='warning')
              return pd.DataFrame()

         # Select only columns that are numbers AFTER encoding
         numeric_ml_cols = self.encoded_data.select_dtypes(include=np.number).columns

         # Exclude potential IDs if they are numeric but high cardinality
         cols_to_exclude = []
         if 'ID' in numeric_ml_cols: # Handle common case explicitly
              cols_to_exclude.append('ID')
         # General heuristic for other potential ID-like columns
         for col in numeric_ml_cols:
              # Avoid re-checking excluded columns
              if col in cols_to_exclude:
                   continue
              # Check if > 80% unique values and looks like an ID
              if 'id' in col.lower() and self.encoded_data[col].nunique() > len(self.encoded_data) * 0.8:
                   cols_to_exclude.append(col)
                   self.log(f"Excluding potential ID column '{col}' from ML features.")

         final_ml_cols = [col for col in numeric_ml_cols if col not in cols_to_exclude]

         # Ensure no NaN/Inf values persist in the data used for ML
         ml_data_subset = self.encoded_data[final_ml_cols].copy()
         # Check for inf values
         if np.isinf(ml_data_subset.to_numpy()).any():
             self.log("Infinite values found in data for ML. Replacing with large finite numbers.", level='warning')
             ml_data_subset.replace([np.inf, -np.inf], np.nan, inplace=True)
             # Re-impute NaNs introduced by replacing inf, perhaps using column means
             for col in ml_data_subset.columns:
                 if ml_data_subset[col].isnull().any():
                     ml_data_subset[col].fillna(ml_data_subset[col].mean(), inplace=True)

         # Final check for NaNs
         if ml_data_subset.isnull().values.any():
             self.log("NaN values still found in data for ML after checks. Filling with 0.", level='warning')
             ml_data_subset.fillna(0, inplace=True)


         num_cols = len(final_ml_cols)
         if num_cols == 0:
              self.log("No valid numeric columns available for ML tasks after filtering.", level='warning')
              return pd.DataFrame()

         self.log(f"Using {num_cols} columns for ML tasks (clustering/importance/outliers).")
         return ml_data_subset


    def _outlier_sampler(self):
        """Identifies and samples potential outliers using Isolation Forest."""
        if not self.use_outlier_sampler: # Check if disabled by user
             self.log("Skipping outlier sampling as per configuration.")
             return

        self.log("Attempting outlier sampling using Isolation Forest...")
        ml_data = self._get_numeric_data_for_ml()

        # Need sufficient data and features for Isolation Forest
        if ml_data.empty or len(ml_data) < 10 or ml_data.shape[1] == 0:
            self.log("Skipping outlier sampling: Not enough numerical data, samples, or features.", level='warning')
            return

        try:
            # Scale data before outlier detection
            scaler = StandardScaler()
            # Use numpy directly for potentially better performance/stability
            scaled_data_np = scaler.fit_transform(ml_data.to_numpy())

            # Isolation Forest - contamination='auto' is often reasonable
            # n_estimators can be increased for larger datasets if needed
            iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
            outlier_labels = iso_forest.fit_predict(scaled_data_np) # -1 for outliers, 1 for inliers

            # Use original DataFrame index to locate outliers
            outlier_indices = ml_data.index[outlier_labels == -1]
            n_outliers = len(outlier_indices)

            if n_outliers > 0:
                # Sample a fraction of the outliers, up to a reasonable limit
                # e.g., 5% of target size, but at least 1, max 10-20?
                n_outliers_to_sample = min(n_outliers, max(1, self.target_sample_size // 20))
                sampled_outlier_indices = np.random.choice(outlier_indices, size=n_outliers_to_sample, replace=False)

                # Get original data rows corresponding to these outliers
                outlier_samples = self.data.loc[sampled_outlier_indices]
                self._add_to_pool(outlier_samples)
                self.log(f"Isolation Forest identified {n_outliers} potential outliers. Added {len(outlier_samples)} outlier samples.")
            else:
                self.log("Isolation Forest did not identify significant outliers with 'auto' contamination.")

        except Exception as e:
            self.log(f"Error during outlier sampling: {e}", level='warning')


    def _feature_importance_sampler(self):
        """Samples based on feature importance if a 'target' column exists."""
        target_col_name = 'target' # Make configurable?
        self.log(f"Checking for feature importance-based sampling ('{target_col_name}' column)...")
        # Check in original columns before encoding
        if target_col_name not in self.data.columns:
            self.log(f"Skipping feature importance sampling: No '{target_col_name}' column found.")
            return

        ml_data = self._get_numeric_data_for_ml()
        # Ensure target is not in the features used for ML
        if target_col_name in ml_data.columns:
             X = ml_data.drop(columns=[target_col_name])
        else:
             X = ml_data

        # Ensure target is aligned with X (using original index) and handle potential NaNs
        y = self.data.loc[X.index, target_col_name]
        valid_target_indices = y.dropna().index
        X = X.loc[valid_target_indices]
        y = y.loc[valid_target_indices]

        if X.empty or y.empty or len(X.columns) == 0:
             self.log("Skipping feature importance: No valid features/target found after encoding/filtering/dropna.", level='warning')
             return
        if y.nunique() < 2:
            self.log(f"Skipping feature importance: Target column '{target_col_name}' has less than 2 unique values.", level='warning')
            return


        self.log("Running feature importance analysis using RandomForestClassifier...")
        try:
            # Simple RF model - parameters can be tuned
            rf = RandomForestClassifier(n_estimators=50, # Increased estimators
                                        random_state=42,
                                        n_jobs=-1,
                                        max_depth=10,
                                        min_samples_leaf=5) # Add regularization
            rf.fit(X, y)

            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            # Select top features
            top_n_features = min(5, len(importances)) # Increase potential features considered
            top_features = importances.head(top_n_features).index.tolist()

            # Identify which top features correspond to original numeric columns for value sampling
            top_features_in_original_numeric = [f for f in top_features if f in self.numerical_cols]

            if not top_features_in_original_numeric:
                 self.log("Top important features are not directly sampleable numeric columns (e.g., one-hot encoded). Skipping value-based importance sampling.", level='warning')
                 # Consider alternative: sample based on top encoded features if needed
                 return

            self.log(f"Top {len(top_features_in_original_numeric)} important numeric features for value sampling: {top_features_in_original_numeric}")
            total_fi_samples = 0
            # Sample very few based on top values per feature (e.g. 1% of target size)
            samples_per_feature = max(1, self.target_sample_size // 100)

            for col in top_features_in_original_numeric:
                 # Sample points with highest values in this important feature
                 # Use original data ('self.data') for sampling values
                 # Ensure sampling from valid indices where target was not NaN
                 valid_data_for_col = self.data.loc[valid_target_indices, col]
                 top_vals_indices = valid_data_for_col.nlargest(samples_per_feature).index
                 if not top_vals_indices.empty:
                     top_val_samples = self.data.loc[top_vals_indices]
                     self._add_to_pool(top_val_samples)
                     total_fi_samples += len(top_val_samples)
                 # Potential Enhancement: sample low values or quantiles as well

            self.log(f"Feature importance sampling added {total_fi_samples} samples based on top values of numeric features.")

        except Exception as e:
            # Catch specific errors if needed, e.g., ValueError if target is continuous for Classifier
            self.log(f"Error during feature importance sampling: {e}", level='warning')


    def _adaptive_clustering_sampler(self):
        """Performs clustering (DBSCAN with KMeans fallback) and samples from clusters."""
        self.log("Attempting adaptive clustering (DBSCAN/KMeans)...")
        ml_data = self._get_numeric_data_for_ml()

        # Need sufficient data and features for clustering
        min_samples_for_cluster = max(10, self.kmeans_k) # Min samples needed
        if ml_data.empty or len(ml_data) < min_samples_for_cluster or ml_data.shape[1] == 0:
            self.log(f"Skipping clustering: Not enough numerical data ({len(ml_data)}x{ml_data.shape[1]}), samples, or features. Min required: {min_samples_for_cluster}", level='warning')
            return

        try:
            # Scale and apply PCA
            scaler = StandardScaler()
            scaled_data_np = scaler.fit_transform(ml_data.to_numpy())

            # Ensure n_components is valid
            n_features = scaled_data_np.shape[1]
            n_components = min(10, n_features) # Use up to 10 components or max features
            if n_components <= 0:
                self.log("Skipping clustering: No components for PCA.", level='warning')
                return

            pca = PCA(n_components=n_components, random_state=42)
            reduced_data = pca.fit_transform(scaled_data_np)
            explained_var = pca.explained_variance_ratio_.sum()
            self.log(f"PCA applied reducing to {n_components} components. Explained variance: {explained_var:.2f}")

            # Try DBSCAN - adjust min_samples based on dimensionality
            dbscan_min_samples = max(5, n_components * 2) # Heuristic: double the dimensions
            # Estimate eps using k-NN distance (optional but recommended for robustness)
            # from sklearn.neighbors import NearestNeighbors
            # k_for_eps = dbscan_min_samples
            # nn = NearestNeighbors(n_neighbors=k_for_eps)
            # nn.fit(reduced_data)
            # distances, indices = nn.kneighbors(reduced_data)
            # sorted_distances = np.sort(distances[:, k_for_eps-1], axis=0)
            # # Find the knee/elbow - requires more complex logic, use fixed for now
            # estimated_eps = 0.5 # Default, replace with knee calculation if implemented

            dbscan = DBSCAN(eps=0.5, min_samples=dbscan_min_samples, n_jobs=-1).fit(reduced_data)
            labels = dbscan.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0) # Number of actual clusters
            noise_ratio = np.sum(labels == -1) / len(labels) if len(labels) > 0 else 0

            # Fallback to KMeans if DBSCAN finds <= 1 cluster OR if noise is extremely high (e.g., >95%)
            if n_clusters <= 1 or noise_ratio > 0.95 :
                self.log(f"DBSCAN found {n_clusters} clusters (noise {noise_ratio:.2f}). Falling back to KMeans (k={self.kmeans_k})...")
                # Ensure k is valid given number of samples
                actual_k = min(self.kmeans_k, len(reduced_data))
                if actual_k < 2: # Need at least 2 clusters for KMeans to be meaningful
                    self.log(f"Cannot run KMeans: effective k ({actual_k}) is less than 2. Skipping clustering.", level='warning')
                    return

                kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10) # Set n_init explicitly
                kmeans.fit(reduced_data)
                labels = kmeans.labels_
                cluster_algo = "KMeans"
            else:
                 self.log(f"DBSCAN clustering successful. Found {n_clusters} clusters (noise {noise_ratio:.2f}).")
                 cluster_algo = "DBSCAN"


            # Sample from clusters (using original data indices)
            cluster_labels_series = pd.Series(labels, index=ml_data.index)
            total_cluster_samples = 0
            n_found_clusters = len(np.unique(labels))

            # Distribute a portion of target size across clusters
            samples_per_cluster = max(1, (self.target_sample_size // 5) // n_found_clusters) if n_found_clusters > 0 else 1

            for label in np.unique(labels):
                 cluster_indices = cluster_labels_series[cluster_labels_series == label].index
                 cluster_data = self.data.loc[cluster_indices] # Sample from original data
                 n = min(samples_per_cluster, len(cluster_data))
                 if n > 0:
                     sample = cluster_data.sample(n=n, random_state=42)
                     self._add_to_pool(sample)
                     total_cluster_samples += len(sample)
                     label_desc = f"cluster {label}" if label != -1 else "noise (outliers)"
                     # self.log(f"Sampled {n} instances from {label_desc} (using {cluster_algo})")

            self.log(f"Clustering ({cluster_algo}) added {total_cluster_samples} samples across {n_found_clusters} unique labels.")

        except Exception as e:
            self.log(f"Error during adaptive clustering: {e}", level='warning')

    def _skew_sampler(self):
        """Samples from highly skewed numerical features using PPS based on absolute value."""
        if not self.numerical_cols:
             self.log("Skipping skew sampling: No numerical columns.", level='warning')
             return

        self.log("Sampling from skewed numerical features (abs(skew) > 1.0)...")
        total_skew_samples = 0
        # Sample very few per column (e.g., 1% of target size)
        samples_per_skewed_col = max(1, self.target_sample_size // 100)

        for col in self.numerical_cols:
             # Check if stats were computed and skew exists and is significant
             if col in self.col_stats and 'skew' in self.col_stats[col] and not pd.isna(self.col_stats[col]['skew']) and abs(self.col_stats[col]['skew']) > 1.0:
                self.log(f"Column '{col}' identified as skewed (skew={self.col_stats[col]['skew']:.2f}). Applying PPS sampling.")
                try:
                    # Using absolute value as weight - simple PPS approach
                    weights = self.data[col].abs()
                    # Handle potential NaNs or zeros introduced during processing
                    weights = weights.fillna(0)

                    sum_weights = weights.sum()
                    if sum_weights == 0 or not np.isfinite(sum_weights):
                         self.log(f"Weights for skewed column '{col}' sum to zero or non-finite. Skipping.", level='warning')
                         continue

                    # Normalize weights, add epsilon for stability
                    weights = (weights + 1e-9) / (sum_weights + 1e-6)

                    # Ensure we don't try to sample more than available valid weights/rows
                    valid_indices = weights[weights > 0].index
                    n_available = len(valid_indices)
                    if n_available == 0:
                        self.log(f"No valid positive weights for skewed column '{col}'. Skipping.", level='warning')
                        continue

                    n_to_sample = min(samples_per_skewed_col, n_available)

                    # Sample only from rows with positive weights
                    selected = self.data.loc[valid_indices].sample(
                        n=n_to_sample,
                        weights=weights.loc[valid_indices], # Use aligned weights
                        replace=False, # Sample without replacement if possible
                        random_state=42
                    )
                    self._add_to_pool(selected)
                    total_skew_samples += len(selected)

                except Exception as e:
                     self.log(f"Error during skew sampling for column '{col}': {e}", level='warning')

        if total_skew_samples > 0:
             self.log(f"Skew sampling added {total_skew_samples} samples.")
        else:
             self.log("No skewed columns met criteria or no samples added.")


    def _finalize_sample(self):
        """Finalizes the sample by deduplicating and resampling the pool."""
        self.log("Finalizing sample...")
        if self.sample_pool is None or self.sample_pool.empty:
            self.log("Sample pool is empty. No final sample generated.", level='warning')
            # Ensure an empty DataFrame with correct columns is available
            cols = self.original_data.columns if self.original_data is not None else []
            self.sampled_data = pd.DataFrame(columns=cols)
            return

        n_pool_before_dedup = len(self.sample_pool)
        # Deduplicate based on all columns
        # Keep='first' is arbitrary but consistent
        self.sample_pool.drop_duplicates(inplace=True, keep='first')
        n_pool_after_dedup = len(self.sample_pool)
        self.log(f"Sample pool size: {n_pool_before_dedup} before deduplication, {n_pool_after_dedup} after.")

        if n_pool_after_dedup == 0:
             self.log("Sample pool empty after deduplication.", level='warning')
             self.sampled_data = pd.DataFrame(columns=self.original_data.columns)
             return

        # Resample from the unique pool to meet the target size
        final_size = min(self.target_sample_size, n_pool_after_dedup)
        self.log(f"Resampling unique pool to final target size: {final_size}")

        # Use resampling without replacement
        self.sampled_data = resample(self.sample_pool,
                                     n_samples=final_size,
                                     replace=False, # Should always be possible now
                                     random_state=42)

        # Reset index for a clean final output
        self.sampled_data.reset_index(drop=True, inplace=True)
        # Ensure final data has same columns as original (in case some were dropped/added)
        self.sampled_data = self.sampled_data.reindex(columns=self.original_data.columns)


    def get_sample(self):
        """
        Returns the generated sample DataFrame.

        Returns:
            pd.DataFrame: The sampled data, or an empty DataFrame if sampling failed or hasn't run.
        """
        if self.sampled_data is None:
             self.log("No sample generated yet. Call fit() first.", level='warning')
             # Return empty DataFrame with original columns if possible
             cols = self.original_data.columns if self.original_data is not None else []
             return pd.DataFrame(columns=cols)
        return self.sampled_data

    def get_explanation(self):
        """Returns the list of explanations logged during the sampling process."""
        return self.explanations

# --- Example Usage ---
if __name__ == '__main__':
    # Create dummy data for demonstration
    data = {
        'ID': range(1000),
        'Category': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'] + ['X', 'Y'] * 2 + [np.nan]*50, 1000,
                                     p=[0.3,.2,.15,.1,.05,.03,.02] + [0.005]*2*2 + [0.15/50]*50), # Added NaN and rare cats 'X', 'Y'
        'Value': np.random.gamma(1, 15, 1000) * np.random.choice([-1, 1], 1000) + 50, # Skewed data
        'Timestamp': pd.to_datetime(np.random.randint(1672531200, 1704067200, size=1000), unit='s'), # ~2023-2024
        'region': np.random.choice(['North', 'South', 'East', 'West', np.nan], 1000, p=[0.3,0.3,0.2,0.15,0.05]), # Add NaN to region
        'household_id': [f"HH_{r}_{np.random.randint(1,20)}" if pd.notna(r) else None for r in np.random.choice(['North', 'South', 'East', 'West', np.nan], 1000)], # Link HH ID to region, allow None
        'AnotherValue': np.random.randn(1000) * 5 + 20,
        'BoolFeature': np.random.choice([True, False, None], 1000, p=[0.45, 0.45, 0.1]), # Boolean with None
        'target': np.random.choice([0, 1, 0], 1000, p=[0.75, 0.20, 0.05]) # Target column, added some 0s
    }
    your_large_dataframe = pd.DataFrame(data)
    # Add more NaNs
    your_large_dataframe.loc[your_large_dataframe.sample(frac=0.05, random_state=1).index, 'Value'] = np.nan
    your_large_dataframe.loc[your_large_dataframe.sample(frac=0.03, random_state=2).index, 'AnotherValue'] = np.nan
    # Make one category very rare
    your_large_dataframe.loc[0, 'Category'] = 'Z' # Very rare category

    print("--- Original DataFrame Info ---")
    your_large_dataframe.info()
    print("\n--- Original Value Counts (Category) ---")
    print(your_large_dataframe['Category'].value_counts(dropna=False))
    print("\n--- Original Skew (Value) ---")
    print(your_large_dataframe['Value'].skew())


    print("\n--- Initializing Sampler ---")
    sampler = DynamicHybridSampler(
        target_sample_size=75,          # Desired sample size
        imputation_strategy='median',   # Use median for numeric imputation
        stratification_col='region',    # Stratify by region column
        kmeans_k=4,                     # Use k=4 for KMeans fallback
        use_outlier_sampler=True,       # Enable outlier sampling step
        use_onehot=False                # Use Label Encoding for categoricals
    )

    print("\n--- Running Sampler Fit ---")
    sampler.fit(your_large_dataframe)

    print("\n--- === Sampling Explanation Log === ---")
    for line in sampler.get_explanation():
        print(line)
    print("--- === End Explanation Log === ---")


    print("\n--- Final Sample ---")
    final_sample = sampler.get_sample()
    if not final_sample.empty:
        print(final_sample.head())
        final_sample.info()

        print("\n--- Final Sample Value Counts (Category) ---")
        print(final_sample['Category'].value_counts(dropna=False))
        print("\n--- Final Sample Skew (Value) ---")
        print(final_sample['Value'].skew())

        # Verify stratification column distribution (approximate)
        print("\n--- Original Stratification Column Distribution ---")
        print(your_large_dataframe['region'].value_counts(normalize=True, dropna=False))
        print("\n--- Final Sample Stratification Column Distribution ---")
        print(final_sample['region'].value_counts(normalize=True, dropna=False))
    else:
        print("Final sample is empty.")

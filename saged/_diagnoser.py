import pandas as pd
from tqdm import tqdm
import warnings

from scipy.stats import zscore

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

import itertools
from scipy import stats
from collections import defaultdict

tqdm.pandas()

import copy
from ._utility import check_benchmark

class DisparityDiagnoser:
    def __init__(self, benchmark, features: list[str] or str = None, target_groups: list[str] or str = None,
                 generations: list[str] or str = None, baseline='baseline', group_type='domain'):

        # Initial placeholders for later use
        self.summary_df_dict = {}
        self.summary_df_dict_with_p_values = {}
        self.disparity_df = {}
        self.specifications = ['category']
        self.full_specification_columns = ['category', 'domain', 'source_tag']

        # Validate the benchmark DataFrame
        check_benchmark(benchmark)

        # If classification_features, generations, or target_groups are not specified, use all available in benchmark
        self.features = [features] if isinstance(features, str) else features if features is not None else \
            list(set([col.split('_', 1)[1] for col in benchmark.columns if f'{baseline}_' in col]))
        self.generations = [generations] if isinstance(generations,
                                                       str) else generations if generations is not None else \
            [col for col in benchmark.columns if
             col not in self.full_specification_columns and not col.startswith(baseline)]
        self.target_groups = [target_groups] if isinstance(target_groups,
                                                           str) else target_groups if target_groups is not None else \
            benchmark[group_type].unique().tolist()

        self.benchmark = benchmark
        self.baseline = baseline

        # Ensure baseline and feature columns exist in the benchmark, if applicable
        self._validate_columns()

        # Modify the benchmark DataFrame based on the specified group_type and target_groups
        self._modify_benchmark(group_type)

    def _identify_and_assign_generations_features(self, benchmark: pd.DataFrame, features=None, generations=None):
        """
        Identifies the generations and classification_features from the DataFrame columns.
        If classification_features or generations are None, the function assigns them based on the DataFrame columns.

        Parameters:
            benchmark (pd.DataFrame): The DataFrame containing the data with columns named in the format 'generation_feature'.
            features (list or str, optional): The list of classification_features to filter. If None, all classification_features are identified.
            generations (list or str, optional): The list of generations to filter. If None, all generations are identified.

        Returns:
            assigned_generations (list): A list of identified or assigned generation prefixes.
            assigned_features (list): A list of identified or assigned classification_features.
        """
        # Step 1: Identify potential generations by grouping columns with the same prefix
        potential_generations = defaultdict(list)

        for col in benchmark.columns:
            if '_' in col:
                prefix = col.rsplit('_', 1)[0]  # Take everything before the last underscore as the generation
                potential_generations[prefix].append(col)

        # Step 2: Filter to keep only those prefixes that correspond to multiple columns (indicating a valid generation)
        assigned_generations = [generations] if isinstance(generations,
                                                           str) else generations if generations is not None else \
            [gen for gen, cols in potential_generations.items() if len(cols) > 1]

        # Step 3: Identify classification_features associated with the confirmed generations
        assigned_features = [features] if isinstance(features, str) else features if features is not None else \
            list(set([col.rsplit('_', 1)[1] for col in benchmark.columns if
                      any(col.startswith(gen + '_') for gen in assigned_generations)]))

        # Final validation to ensure generation-feature pairs exist in the DataFrame
        assigned_generations = [gen for gen in assigned_generations if
                                any(f'{gen}_{feature}' in benchmark.columns for feature in assigned_features)]

        return assigned_generations, assigned_features

    def _validate_columns(self):
        """Validate that all necessary columns exist in the benchmark DataFrame."""

        for generation in self.generations:
            assert generation in self.benchmark.columns, f"Column '{generation}' not found in benchmark"
            for feature in self.features:
                assert f'{generation}_{feature}' in self.benchmark.columns, \
                    f"Generation feature '{generation}_{feature}' not found in benchmark"

    def _modify_benchmark(self, group_type):
        """Modify the benchmark DataFrame by retaining relevant columns based on specified classification_features and generations."""

        self.benchmark.drop(columns=['prompts', 'keyword'], inplace=True)

        assert group_type in ['domain', 'category'], "Please use 'domain' or 'category' as the group_type."

        # If target_groups is specified, filter rows based on the group_type
        if self.target_groups:
            if group_type == 'domain':
                self.benchmark = self.benchmark[self.benchmark['domain'].isin(self.target_groups)]
            elif group_type == 'category':
                self.benchmark = self.benchmark[self.benchmark['category'].isin(self.target_groups)]

        # Escape special characters in baseline, generations, and features
        baseline_pattern = f"{re.escape(self.baseline)}_({'|'.join(map(re.escape, self.features))})"
        generations_pattern = f"({'|'.join(map(re.escape, self.generations))})_({'|'.join(map(re.escape, self.features))})"

        # Define the regex patterns to match the columns
        columns_to_keep = self.full_specification_columns + \
                          [col for col in self.benchmark.columns if re.match(baseline_pattern, col)] + \
                          [col for col in self.benchmark.columns if re.match(generations_pattern, col)] + \
                          self.generations

        # Keep only the relevant columns
        self.benchmark = self.benchmark[columns_to_keep]

        self.value_columns = []
        for feature in self.features:
            for generation in self.generations:
                self.value_columns += [f'{generation}_{feature}']


    def _summary_statistics(self, summary_function, permutation_test=False, custom_agg=False,
                            num_permutations=1000, **kwargs):
        elements = self.specifications.copy()
        df = self.benchmark.copy()

        # Generate all combinations (Cartesian product) of the elements
        combinations = list(itertools.product(elements, repeat=len(elements)))
        sorted_combinations = sorted(combinations,
                                     key=lambda combo: (len(set(combo)), [elements.index(item) for item in combo]))

        value_columns = self.value_columns
        summary_df = pd.DataFrame()


        summary_df_with_p_values = pd.DataFrame()

        def perform_permutation_test(df, group, combo, value_col, summary_function, num_permutations=1000):
            """
            Perform a permutation test on a given group of data within a DataFrame.
            """
            # Get the observed statistic for the current group
            original_stat = group[value_col]

            # Filter the DataFrame according to the specifications in the group
            filtered_data = df.copy()
            for col in combo:
                filtered_data = filtered_data[filtered_data[col] == group[col]]

            perm_stats = []

            for _ in range(num_permutations):
                # Permute the rows of the filtered DataFrame
                shuffled_data = filtered_data.sample(frac=1, replace=False).reset_index(drop=True)

                # Apply the summary function directly on the permuted DataFrame
                warnings.simplefilter(action='ignore', category=FutureWarning)
                perm_stat = summary_function(shuffled_data)
                warnings.simplefilter(action='default', category=FutureWarning)

                # Store the permuted statistic
                perm_stats.append(perm_stat)

            # Calculate p-value by comparing permuted statistics with the observed statistic
            p_value = np.mean(np.array(perm_stats) >= original_stat)

            return p_value

        if summary_function: #make a copy of the summary function
            copy_summary_function = copy.deepcopy(summary_function)

        # Iterate over sorted combinations
        for combo in sorted_combinations:
            combo = list(set(combo))
            remainder = [item for item in self.full_specification_columns if item not in combo]
            process_columns = combo + value_columns


            if summary_function:
                # print('============================='
                #       '=============================')
                # print(f'Processing {combo}...')
                # print(list(df.columns))
                df_2 = df[process_columns].copy()
                df_2 = df_2.loc[:, ~df_2.columns.duplicated()]
                if ('standard_assign' in kwargs) and kwargs['standard_assign'] and 'sd_extract_fn' in kwargs:
                    sd_extract_fn = kwargs['sd_extract_fn']  # extract the standard
                    standard_dict = sd_extract_fn(df_2.drop(columns = combo))
                    # print(standard_dict)
                    def summary_function(x): return copy_summary_function(x, standard_dict)

                grouped = df_2.groupby(combo)
                # grouped = grouped.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtypes == 'object' else col)
                if custom_agg:
                    new_df = grouped.apply(summary_function)
                else:
                    # print(grouped.head())
                    new_df = grouped.agg(summary_function)

                new_df_2 = new_df.copy()
                new_df = new_df_2.reset_index(level=combo)

                for item in remainder:
                    new_df[item] = 'ALL'

                summary_df = pd.concat([summary_df, new_df], axis=0)
                new_order = self.specifications + [col for col in summary_df.columns if col not in self.specifications]
                summary_df = summary_df[new_order].copy()
                summary_df = summary_df.copy().reset_index(drop=True)

                if permutation_test:
                    for value_col in value_columns:
                        # Add a column for p-values
                        p_values = []
                        for _, group in new_df.iterrows():
                            p_value = perform_permutation_test(
                                df=df,
                                group=group,
                                combo=combo,
                                value_col=value_col,
                                summary_function=summary_function,
                                num_permutations=num_permutations
                            )
                            p_values.append(p_value)

                        new_df[f'{value_col}_p_value'] = p_values

                    summary_df_with_p_values = summary_df_with_p_values.reset_index(drop=True)
                    new_df = new_df.reset_index(drop=True)
                    summary_df_with_p_values = pd.concat([summary_df_with_p_values, new_df], axis=0).reset_index(drop=True)



        return summary_df, summary_df_with_p_values

    def statistics_disparity(self):
        """
        disparity by ratio and differences and other things
        """

        def _dixon_q_test(data):
            # Dixon Q critical values tables
            dixon_critical_values = {
                3: {0.1: 0.941, 0.05: 0.970, 0.04: 0.976, 0.02: 0.988, 0.01: 0.994},
                4: {0.1: 0.765, 0.05: 0.829, 0.04: 0.846, 0.02: 0.889, 0.01: 0.926},
                5: {0.1: 0.642, 0.05: 0.710, 0.04: 0.729, 0.02: 0.780, 0.01: 0.821},
                6: {0.1: 0.560, 0.05: 0.625, 0.04: 0.644, 0.02: 0.698, 0.01: 0.740},
                7: {0.1: 0.507, 0.05: 0.568, 0.04: 0.586, 0.02: 0.637, 0.01: 0.680},
                8: {0.1: 0.468, 0.05: 0.526, 0.04: 0.543, 0.02: 0.590, 0.01: 0.634},
                9: {0.1: 0.437, 0.05: 0.493, 0.04: 0.510, 0.02: 0.555, 0.01: 0.598},
                10: {0.1: 0.412, 0.05: 0.466, 0.04: 0.483, 0.02: 0.527, 0.01: 0.568},
            }

            dixon_critical_values_r11 = {
                8: {0.001: 0.799, 0.002: 0.769, 0.005: 0.724, 0.01: 0.682, 0.02: 0.633, 0.05: 0.554, 0.1: 0.480,
                    0.2: 0.386},
                9: {0.001: 0.750, 0.002: 0.720, 0.005: 0.675, 0.01: 0.634, 0.02: 0.586, 0.05: 0.512, 0.1: 0.441,
                    0.2: 0.352},
                10: {0.001: 0.713, 0.002: 0.683, 0.005: 0.637, 0.01: 0.597, 0.02: 0.551, 0.05: 0.477, 0.1: 0.409,
                     0.2: 0.325},
            }

            dixon_critical_values_r21 = {
                11: {0.001: 0.770, 0.002: 0.746, 0.005: 0.708, 0.01: 0.674, 0.02: 0.636, 0.05: 0.575, 0.1: 0.518,
                     0.2: 0.445},
                12: {0.001: 0.739, 0.002: 0.714, 0.005: 0.676, 0.01: 0.643, 0.02: 0.605, 0.05: 0.546, 0.1: 0.489,
                     0.2: 0.420},
                13: {0.001: 0.713, 0.002: 0.687, 0.005: 0.649, 0.01: 0.617, 0.02: 0.580, 0.05: 0.522, 0.1: 0.467,
                     0.2: 0.399},
            }

            dixon_critical_values_r22 = {
                14: {0.001: 0.732, 0.002: 0.708, 0.005: 0.672, 0.01: 0.640, 0.02: 0.603, 0.05: 0.546, 0.1: 0.491,
                     0.2: 0.422},
                15: {0.001: 0.708, 0.002: 0.685, 0.005: 0.648, 0.01: 0.617, 0.02: 0.582, 0.05: 0.524, 0.1: 0.470,
                     0.2: 0.403},
                16: {0.001: 0.691, 0.002: 0.667, 0.005: 0.630, 0.01: 0.598, 0.02: 0.562, 0.05: 0.505, 0.1: 0.453,
                     0.2: 0.386},
                17: {0.001: 0.671, 0.002: 0.647, 0.005: 0.611, 0.01: 0.580, 0.02: 0.545, 0.05: 0.489, 0.1: 0.437,
                     0.2: 0.373},
                18: {0.001: 0.652, 0.002: 0.628, 0.005: 0.594, 0.01: 0.564, 0.02: 0.529, 0.05: 0.475, 0.1: 0.424,
                     0.2: 0.361},
                19: {0.001: 0.640, 0.002: 0.617, 0.005: 0.581, 0.01: 0.551, 0.02: 0.517, 0.05: 0.462, 0.1: 0.412,
                     0.2: 0.349},
                20: {0.001: 0.627, 0.002: 0.604, 0.005: 0.568, 0.01: 0.538, 0.02: 0.503, 0.05: 0.450, 0.1: 0.401,
                     0.2: 0.339},
                25: {0.001: 0.574, 0.002: 0.550, 0.005: 0.517, 0.01: 0.489, 0.02: 0.457, 0.05: 0.406, 0.1: 0.359,
                     0.2: 0.302},
                30: {0.001: 0.539, 0.002: 0.517, 0.005: 0.484, 0.01: 0.456, 0.02: 0.425, 0.05: 0.376, 0.1: 0.332,
                     0.2: 0.278},
            }

            n = len(data)

            if n < 3 or n > 30:
                return "not applicable due to sample size"

            data_sorted = sorted(data)

            error = 0.000000001

            if 4 <= n <= 7:
                # Use r10
                denominator = data_sorted[-1] - data_sorted[0]
                Q_exp = (data_sorted[0] - data_sorted[0]) / denominator if denominator != 0 else (data_sorted[1] - data_sorted[0]) / (denominator + error)
                critical_values = dixon_critical_values.get(n, {})
            elif 8 <= n <= 10:
                # Use r11
                denominator = data_sorted[-1] - data_sorted[0]
                Q_exp = (data_sorted[1] - data_sorted[0]) / denominator if denominator != 0 else (data_sorted[1] - data_sorted[0]) / (denominator + error)
                critical_values = dixon_critical_values_r11.get(n, {})
            elif 11 <= n <= 13:
                # Use r21
                denominator = data_sorted[-1] - data_sorted[0]
                Q_exp = (data_sorted[2] - data_sorted[0]) / denominator if denominator != 0 else (data_sorted[2] - data_sorted[0]) / (denominator + error)
                critical_values = dixon_critical_values_r21.get(n, {})
            elif 14 <= n <= 30:
                # Use r22
                denominator = data_sorted[-1] - data_sorted[0]
                Q_exp = (data_sorted[2] - data_sorted[0]) / denominator if denominator != 0 else (data_sorted[2] - data_sorted[0]) / (denominator + error)
                critical_values = dixon_critical_values_r22.get(n, {})
            else:
                critical_values = dixon_critical_values_r22.get(5, {})
                return "not applicable due to sample size"

            # Find the lowest alpha where Q_exp > Q_critical
            for alpha in sorted(critical_values.keys()):
                Q_critical = critical_values[alpha]
                if Q_exp > Q_critical:
                    return f'significance at {alpha}'

            return "not significance"

        def calculate_disparities_by_column(df, value_columns):
            """
            Calculate disparity metrics (max/min Ratio, Difference, Standard Deviation, Max, Min, Z-Score, Dixon's Q Test)
            for each individual column and return them in the specified format.

            Parameters:
            df (pd.DataFrame): The input DataFrame containing the values.
            value_columns (list): A list of column names representing the values to be used for disparity calculation.

            Returns:
            pd.DataFrame: DataFrame with columns 'disparity_metric', 'value_1', 'value_2', 'value_3'.
            """
            disparity_dict = {'disparity_metric': []}
            disparity_dict['disparity_metric'].extend(
                ['Max', 'Min', 'Min/Max', 'Max-Min', 'Avg', 'Std', 'Max Z-Score', 'Dixon Q'])

            for col in value_columns:
                values = df[col].dropna()  # Drop NaN values for accurate calculations
                disparity_dict[col] = []

                # Calculate the max/min ratio
                max_min_ratio = values.min() / values.max() if values.max() != 0 else np.nan  # Handle division by zero
                if max_min_ratio == np.nan:
                    print(f"max_min_ratio is nan for column '{col}', because the max value is 0.")
                difference = values.max() - values.min()
                average = values.mean()
                std_dev = values.std()

                # Calculate the Z-Score for the maximum value
                z_scores = zscore(values)
                z_scores = z_scores[~np.isnan(z_scores)]  # Remove NaN values
                if len(z_scores) > 0:
                    max_z_score = np.max(z_scores)
                else:
                    max_z_score = np.nan
                    print(f"No valid z-scores for column {col}.")

                # Calculate Dixon's Q test statistic for potential outliers
                try:
                    dixon_q = _dixon_q_test(values)
                except ValueError as e:
                    print(f"Dixon's Q test not applicable for column '{col}' because {e}")
                    dixon_q = np.nan

                # Find the max and min values with their corresponding category, domain, source_tag
                if not values.empty:
                    max_row = df.loc[values.idxmax()]
                    min_row = df.loc[values.idxmin()]
                    max_info = f"({max_row['category']},{max_row['domain']},{max_row['source_tag']}: {max_row[col]})"
                    min_info = f"({min_row['category']},{min_row['domain']},{min_row['source_tag']}: {min_row[col]})"
                else:
                    max_info = "No data available for max value."
                    min_info = "No data available for min value."

                # Append results for each disparity metric
                disparity_dict[col].extend(
                    [max_info, min_info, max_min_ratio, difference, average, std_dev, max_z_score, dixon_q])

            # Convert the dictionary to a DataFrame
            disparity_df = pd.DataFrame(disparity_dict)
            return disparity_df

        value_columns = self.value_columns
        result_df = pd.DataFrame()
        for statistics, summary_df in self.summary_df_dict.items():
            disparity_df = calculate_disparities_by_column(summary_df, value_columns)
            disparity_df['statistics'] = statistics
            result_df = pd.concat([result_df.copy(), disparity_df], axis=0)

        # reorder it
        column = result_df.pop('statistics')
        result_df.insert(0, 'statistics', column)
        result_df = result_df.copy().reset_index(drop=True)

        self.disparity_df = result_df

        return result_df

    def customized_statistics(self, customized_function, customized_name ='customized', custom_agg = False, test = False, **kwargs):
        summary_function = customized_function
        summary_df, summary_df_with_p_values = self._summary_statistics(summary_function, custom_agg = custom_agg, permutation_test=test, **kwargs)
        self.summary_df_dict[customized_name] = summary_df
        self.summary_df_dict_with_p_values[customized_name] = summary_df_with_p_values
        if test:
            return summary_df_with_p_values
        else:
            return summary_df

    def mean(self, **kwargs):
        summary_function = lambda x: np.mean(x.dropna()) if not x.dropna().empty else np.nan
        # summary_function = lambda x: np.mean(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) > 0 else np.nan
        function_name = 'mean'
        return self.customized_statistics(summary_function, function_name, **kwargs)

    def median(self, **kwargs):
        summary_function = lambda x: np.median(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) > 0 else np.nan
        function_name = 'median'
        return self.customized_statistics(summary_function, function_name, **kwargs)

    def mode(self, bin_width=0.5, **kwargs):
        def _binning_average(data, bin_width=0.5):
            bins = np.arange(np.nanmin(data), np.nanmax(data) + bin_width, bin_width)
            binned_data = np.digitize(data, bins)
            mode_bin = stats.mode(binned_data)[0]
            mode_average = bins[mode_bin - 1] + bin_width / 2  # Midpoint of the bin
            return mode_average

        summary_function = lambda x: _binning_average(x[~np.isnan(x)], bin_width) if len(x[~np.isnan(x)]) > 0 else np.nan
        function_name = 'mode'
        return self.customized_statistics(summary_function, function_name, **kwargs)

    def variance(self, **kwargs):
        summary_function = lambda x: np.var(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) > 0 else np.nan
        function_name = 'variance'
        return self.customized_statistics(summary_function, function_name, **kwargs)

    def standard_deviation(self, **kwargs):
        summary_function = lambda x: np.std(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) > 0 else np.nan
        function_name = 'standard_deviation'
        return self.customized_statistics(summary_function, function_name, **kwargs)

    def skewness(self, **kwargs):
        summary_function = lambda x: stats.skew(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) > 0 else np.nan
        function_name = 'skewness'
        return self.customized_statistics(summary_function, function_name, **kwargs)

    def kurtosis(self, **kwargs):
        summary_function = lambda x: stats.kurtosis(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) > 0 else np.nan
        function_name = 'kurtosis'
        return self.customized_statistics(summary_function, function_name, **kwargs)

    def range(self, **kwargs):
        summary_function = lambda x: np.nanmax(x[~np.isnan(x)]) - np.nanmin(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) > 0 else np.nan
        function_name = 'range'
        return self.customized_statistics(summary_function, function_name, **kwargs)

    def quantile_range(self, lower=0.25, upper=0.75, **kwargs):
        summary_function = lambda x: np.quantile(x[~np.isnan(x)], upper) - np.quantile(x[~np.isnan(x)], lower) if len(x[~np.isnan(x)]) > 0 else np.nan
        function_name = f'quantile_{lower}_{upper}'
        return self.customized_statistics(summary_function, function_name, **kwargs)

    def percentile_range(self, lower=25, upper=75, **kwargs):
        summary_function = lambda x: np.percentile(x[~np.isnan(x)], upper) - np.percentile(x[~np.isnan(x)], lower) if len(x[~np.isnan(x)]) > 0 else np.nan
        function_name = f'percentile_{lower}_{upper}'
        return self.customized_statistics(summary_function, function_name, **kwargs)

    def kl_divergence(self, baseline = None, **kwargs):

        if baseline is None:
            baseline = self.baseline

        if 'bins' in kwargs and isinstance(kwargs['bins'], int):
            bins = kwargs['bins']
        else:
            bins = 10

        def _convert_to_distribution(data, bins=bins, range=None):
            # Calculate the histogram
            hist, bin_edges = np.histogram(data, bins=bins, range=range, density=True)
            # Normalize to ensure it sums to 1 (probability distribution)
            hist = hist / np.sum(hist)
            return hist

        def pair_kl_divergence(p, q, bins=10, range=None):
            # Remove NaNs from input
            p = p[~np.isnan(p)]
            q = q[~np.isnan(q)]

            # Check if either p or q is empty after removing NaNs
            if p.size == 0 or q.size == 0:
                return np.nan

            # Convert continuous data to probability distributions
            p_dist = _convert_to_distribution(p, bins=bins, range=range)
            q_dist = _convert_to_distribution(q, bins=bins, range=range)

            # Prevent division by zero and log of zero
            p_dist = np.where(p_dist != 0, p_dist, 1e-10)
            q_dist = np.where(q_dist != 0, q_dist, 1e-10)

            # KL divergence calculation
            return np.sum(p_dist * np.log(p_dist / q_dist))

        def summary_custom_agg(group):
            summary = {}

            # Calculate KL divergence for all columns against baseline
            for col in group.columns:
                feature_candidate = [feature for feature in self.features if feature in col]
                if feature_candidate != []:
                    feature = feature_candidate[0]
                    baseline_col = f'{baseline}_{feature}'
                    # Filter out NaN values
                    non_nan_mask = ~np.isnan(group[col])
                    if non_nan_mask.sum() == 0:  # Check if all values are NaN
                        summary[col] = np.nan
                        continue

                    if baseline_col in group.columns:
                        divergence = pair_kl_divergence(group[col], group[f'{baseline}_{feature}'])
                    else:
                        warnings.warn(f'Baseline feature {baseline_col} not found. Defaulting baseline to 0.')
                        divergence = pair_kl_divergence(group[col], np.zeros(len(group[col])))
                    summary[col] = divergence

            return pd.Series(summary)

        function_name = f'kl_divergence_wrt_{baseline}_bin_{bins}'
        return self.customized_statistics(summary_custom_agg, function_name, custom_agg=True, **kwargs)

    def precision(self, baseline=None, tolerance=0, **kwargs):
        if baseline is None:
            baseline = self.baseline

        def summary_custom_agg(group):
            summary = {}
            for col in group.columns:
                feature_candidate = [feature for feature in self.features if feature in col]
                if feature_candidate:
                    feature = feature_candidate[0]
                    baseline_col = f'{baseline}_{feature}'

                    # Filter out NaN values
                    non_nan_mask = ~np.isnan(group[col])

                    if non_nan_mask.sum() == 0:  # Check if all values are NaN
                        summary[col] = np.nan
                        continue

                    if baseline_col in group.columns:
                        within_tolerance = (group[col][non_nan_mask] - group[baseline_col][
                            non_nan_mask]).abs() <= tolerance
                    else:
                        warnings.warn(f'Baseline feature {baseline_col} not found. Defaulting baseline to 0.')
                        within_tolerance = (group[col][non_nan_mask] - 0).abs() <= tolerance

                    precision = within_tolerance.sum() / len(group[non_nan_mask])
                    summary[col] = precision
            return pd.Series(summary)

        function_name = f'precision_wrt_{baseline}_tolerance_{tolerance}'
        return self.customized_statistics(summary_custom_agg, function_name, custom_agg=True, **kwargs)

    def selection_rate(self, standard_by='mean', selection_method='larger', **kwargs):

        def _binning(data, bin_width=0.5):
            # data = data[~np.isnan(data)]
            # if len(data) == 0:  # If all values are NaN, return NaN
            #     return np.nan
            bins = np.arange(np.nanmin(data), np.nanmax(data) + bin_width, bin_width)
            binned_data = np.digitize(data, bins)
            mode_bin = stats.mode(binned_data)[0]
            mode_average = bins[mode_bin - 1] + bin_width / 2  # Midpoint of the bin
            return mode_average

        def statistical_measure(data):
            # data = data[~np.isnan(data)]
            # if len(data) == 0:  # If all values are NaN, return NaN
            #     return np.nan
            if standard_by == 'mean':
                return np.mean(data)
            elif standard_by == 'median':
                return np.median(data)
            elif standard_by.startswith('mode'):
                bin_width = float(standard_by.split('-')[1]) if len(standard_by.split('-')) == 2 else 0.1
                return _binning(data, bin_width)
            elif standard_by.startswith('quantile_range'):
                try:
                    q = float(standard_by.split('=')[1])
                    return np.quantile(data, q)
                except (IndexError, ValueError):
                    raise ValueError("For quantile_range, use the format 'quantile_range=0.25' for the 25th percentile")
            elif standard_by.startswith('percentile_range'):
                try:
                    p = float(standard_by.split('=')[1])
                    return float(np.percentile(data, p))
                except (IndexError, ValueError):
                    raise ValueError(
                        "For percentile_range, use the format 'percentile_range=25' for the 25th percentile")
            elif standard_by.startswith('fixed'):
                try:
                    p = float(standard_by.split('=')[1])
                    return p
                except (IndexError, ValueError):
                    raise ValueError(
                        "For percentile_range, use the format 'fixed=0.4' for the 0.4 distance")
            else:
                raise ValueError(
                    "Invalid measure specified. Use 'mean', 'median', 'mode', 'quantile_range=q', or 'percentile_range=p'.")

        def standard_extraction_function(df):
            standard_dict = {col: statistical_measure(df[col].tolist()) for col in df.columns}
            return standard_dict

        def summary_custom_agg(group, standard_dict):
            summary = {}
            for col in group.columns:
                try:
                    col_standard = standard_dict[col]

                    # Filter out NaN values
                    non_nan_mask = ~np.isnan(group[col])

                    if non_nan_mask.sum() == 0:  # Check if all values are NaN
                        summary[col] = np.nan
                        continue

                    if selection_method == 'larger':
                        sf = lambda x, standard: (x >= standard).mean()
                    elif selection_method == 'smaller':
                        sf = lambda x, standard: (x <= standard).mean()
                    elif selection_method.startswith('within-range'):
                        rang = float(selection_method.split('-')[2])
                        sf = lambda x, standard: (abs(x - standard) <= rang).mean()
                    elif selection_method.startswith('within-percentage'):
                        percentage = float(selection_method.split('-')[2])
                        sf = lambda x, standard: (abs(x - standard) <= percentage * standard).mean()
                    else:
                        raise ValueError(
                            "Invalid selection method specified. Use 'larger', 'smaller', 'within-range-e', 'within-percentage-p'.")

                    selection_rate = sf(group[col][non_nan_mask], col_standard)
                    summary[col] = selection_rate
                except:
                    pass
            return pd.Series(summary)

        function_name = f'sr_{selection_method}_sd_{standard_by}'
        return self.customized_statistics(summary_custom_agg, function_name, custom_agg=True, standard_assign=True,
                                          sd_extract_fn=standard_extraction_function, **kwargs)

    def correlation(self, baseline=None, method='pearson', **kwargs):
        if baseline is None:
            baseline = self.baseline

        def summary_custom_agg(group):
            summary = {}
            for col in group.columns:
                # Identify the feature based on self.features
                feature_candidate = [feature for feature in self.features if feature in col]
                if feature_candidate:
                    feature = feature_candidate[0]
                    baseline_col = f'{baseline}_{feature}'

                    # Filter out rows where either the column or baseline column has NaN
                    if baseline_col in group.columns:
                        valid_mask = ~group[[col, baseline_col]].isnull().any(axis=1)
                    else:
                        # If baseline column is missing, skip the correlation computation
                        warnings.warn(
                            f'Baseline feature {baseline_col} not found. Correlation will not be computed for {col}.')
                        summary[col] = np.nan
                        continue

                    if valid_mask.sum() == 0:  # Check if there are no valid rows
                        summary[col] = np.nan
                        continue

                    # Compute correlation using the specified method (e.g., 'pearson', 'spearman', 'kendall')
                    correlation_value = group[col][valid_mask].corr(group[baseline_col][valid_mask], method=method)
                    summary[col] = correlation_value

            return pd.Series(summary)

        # Define function name for identification
        function_name = f'correlation_wrt_{baseline}_method_{method}'
        return self.customized_statistics(summary_custom_agg, function_name, custom_agg=True, **kwargs)


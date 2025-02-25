import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Tuple
from statsmodels.tsa.seasonal import DecomposeResult as DecompResults
from abc import ABC, abstractmethod
from src.utils.generate_dataset import generate_windows_dataset
from src.utils.transformations import (
    manipulate_seasonal_component,
    manipulate_trend_component,
)
from src.utils.features import (
    trend_strength,
    trend_slope,
    trend_linearity,
    seasonal_strength,
)

from src.models.naive_correlation import CorrelationModel
from src.utils.genetic_algorithm import GeneticAlgorithm

# Set up logging
logging.basicConfig(level=logging.INFO)


# DATA
def get_mts_dataset(
    data_dir, time_series_to_use, context_length, step_size, backfill=True, index_col=0
):
    df = pd.read_csv(data_dir, index_col=index_col)
    df.index = pd.to_datetime(df.index)
    df = df[time_series_to_use]
    if backfill:
        df = df.bfill()
    mts_dataset = generate_windows_dataset(
        df, context_length, step_size, time_series_to_use
    )
    return mts_dataset


def get_transformed_uts_with_features_and_decomps(
    uts_decomp,
    trend_det_factor,
    trend_slope_factor,
    trend_lin_factor,
    seasonal_det_factor,
    m=0,  # FIXME: Not yet included in config. Other values than 0 have been problematic.
):
    transformed_uts_trend = manipulate_trend_component(
        trend_comp=uts_decomp.trend,
        f=trend_det_factor,
        g=trend_slope_factor,
        h=trend_lin_factor,
        m=m,
    )

    transformed_uts_seasonal = manipulate_seasonal_component(
        seasonal_comp=uts_decomp.seasonal, k=seasonal_det_factor
    )

    transformed_uts = (
        transformed_uts_trend + transformed_uts_seasonal + uts_decomp.resid
    )

    transformed_uts_decomp = {
        "trend": transformed_uts_trend,
        "seasonal": transformed_uts_seasonal,
        "resid": uts_decomp.resid,
    }

    transformed_uts_features = np.array(
        [
            trend_strength(transformed_uts, uts_decomp.trend),
            trend_slope(transformed_uts),
            trend_linearity(transformed_uts),
            seasonal_strength(transformed_uts, uts_decomp.seasonal),
        ]
    )

    return transformed_uts, transformed_uts_features, transformed_uts_decomp


# MODELS
class TimeSeriesTransformationModel(ABC):
    def __init__(
        self,
        model_type: str,
        model_params: dict,
        mts_dataset: pd.DataFrame,
        mts_features: pd.DataFrame,
        mts_decomp: List[DecompResults],
    ):
        self.model_type = model_type
        self.model_params = model_params
        self.mts_dataset = mts_dataset
        self.mts_features = mts_features
        self.mts_decomp = mts_decomp

    # NOTE: May want to send more params than necessary and only use relevant params for a given model
    @abstractmethod
    def fit(self):
        pass

    # NOTE: Same as above
    @abstractmethod
    def transform(self):
        pass


class CorrelationGeneticAlgorithmModel(TimeSeriesTransformationModel):
    def __init__(
        self,
        model_type: str,
        model_params: dict,
        mts_dataset: pd.DataFrame,
        mts_features: pd.DataFrame,
        mts_decomp: List[DecompResults],
        num_uts_in_mts: int,
        num_features_per_uts: int,
        manual_init_transform: bool = False,
    ):
        super().__init__(
            model_type, model_params, mts_dataset, mts_features, mts_decomp
        )
        self.model = CorrelationModel()
        self.num_features_per_uts = num_features_per_uts
        self.num_uts_in_mts = num_uts_in_mts
        self.manual_init_transform = manual_init_transform
        self.data_set_size = len(mts_dataset)

    def fit(self) -> None:
        self.model.train(self.mts_features)
        logging.info("Successfully fit correlation model to feature data")

    def predict_mts_feature_values(
        self, model_input, original_mts_index, target_mts_index, init_uts_index
    ):
        predicted_features = self.model.infer(model_input)
        return predicted_features

    def transform(
        self,
        model_input: pd.DataFrame,
        original_mts_index: int,
        target_mts_index: int,
        init_uts_index: int,
    ) -> Tuple[List, List, List, np.ndarray]:
        """Predict target features and transform MTS using genetic algorithm.

        Args:
            model_input (_type_): Dataframe containing model input for a singe MTS
            original_mts_index (_type_): Index of the MTS that is to be transformed
            target_mts_index (_type_): Index of the transformation target MTS
            init_uts_index (_type_): Index of the UTS in the original MTS that we base the transformation on

        Returns:
            Tuple[List, List, List, np.ndarray]: List of transformed MTS, List of transformed MTS features, List of factors used for transformation, Predicted features
        """
        predicted_features_df = self.predict_mts_feature_values(
            model_input, original_mts_index, target_mts_index, init_uts_index
        )
        logging.info("Successfully predicted features")

        # Get GA parameters
        num_GA_runs = self.model_params["genetic_algorithm_args"]["num_runs"]
        num_generations = self.model_params["genetic_algorithm_args"]["num_generations"]
        num_parents_mating = self.model_params["genetic_algorithm_args"][
            "num_parents_mating"
        ]
        solutions_per_pop = self.model_params["genetic_algorithm_args"][
            "solutions_per_population"
        ]
        init_range_low = self.model_params["genetic_algorithm_args"]["init_range_low"]
        init_range_high = self.model_params["genetic_algorithm_args"]["init_range_high"]
        parent_selection_type = self.model_params["genetic_algorithm_args"][
            "parent_selection_type"
        ]
        crossover_type = self.model_params["genetic_algorithm_args"]["crossover_type"]
        mutation_type = self.model_params["genetic_algorithm_args"]["mutation_type"]
        mutation_percent_genes = self.model_params["genetic_algorithm_args"][
            "mutation_percent_genes"
        ]
        trend_det_factor_low = self.model_params["genetic_algorithm_args"][
            "legal_values"
        ]["trend_det_factor"][0]
        trend_det_factor_high = self.model_params["genetic_algorithm_args"][
            "legal_values"
        ]["trend_det_factor"][1]
        trend_slope_factor_low = self.model_params["genetic_algorithm_args"][
            "legal_values"
        ]["trend_slope_factor"][0]
        trend_slope_factor_high = self.model_params["genetic_algorithm_args"][
            "legal_values"
        ]["trend_slope_factor"][1]
        trend_lin_factor_low = self.model_params["genetic_algorithm_args"][
            "legal_values"
        ]["trend_lin_factor"][0]
        trend_lin_factor_high = self.model_params["genetic_algorithm_args"][
            "legal_values"
        ]["trend_lin_factor"][1]
        seasonal_det_factor_low = self.model_params["genetic_algorithm_args"][
            "legal_values"
        ]["seasonal_det_factor"][0]
        seasonal_det_factor_high = self.model_params["genetic_algorithm_args"][
            "legal_values"
        ]["seasonal_det_factor"][1]

        num_genes = self.num_features_per_uts

        # Contstraint on GA solutions
        legal_factor_values = [
            np.linspace(trend_det_factor_low, trend_det_factor_high, 100),
            np.linspace(trend_slope_factor_low, trend_slope_factor_high, 100),
            np.linspace(trend_lin_factor_low, trend_lin_factor_high, 100),
            np.linspace(seasonal_det_factor_low, seasonal_det_factor_high, 100),
        ]

        # Get predicted feature values in right shape
        is_index_column = predicted_features_df.columns.str.contains("index")
        predicted_features = predicted_features_df.loc[:, ~is_index_column].to_numpy()
        predicted_features_reshape = predicted_features.reshape(3, 4)

        # NOTE May want to drop support for multiple GA runs?
        # Run genetic algorithm
        GA_runs_mts = []
        GA_runs_features = []
        GA_runs_factors = []

        logging.info("Starting genetic algorithm runs...")
        for _ in tqdm(range(num_GA_runs)):
            transformed_mts = []
            transformed_mts_features = []
            transformed_mts_factors = []

            for i in range(self.num_uts_in_mts):
                # NOTE: If manual transform is enabled, skip the GA run for the UTS that is manually transformed
                if i == init_uts_index and self.manual_init_transform:
                    assert (
                        (self.manual_transformed_uts is not None)
                        and (self.manual_transformed_uts_decomp is not None)
                        and (self.manual_transformed_uts_features is not None)
                    ), "Manual transform not done"
                    transformed_mts.append(self.manual_transformed_uts)
                    transformed_mts_features.append(
                        self.manual_transformed_uts_features
                    )
                    transformed_mts_factors.append(
                        [
                            self.manual_transform_factors[0],
                            self.manual_transform_factors[1],
                            self.manual_transform_factors[2],
                            self.manual_transform_factors[3],
                        ]
                    )
                    continue

                init_mts_decomps = self.mts_decomp[original_mts_index]
                univariate_decomps = init_mts_decomps[i]
                univariate_target_features = predicted_features_reshape[i]

                ga_instance = GeneticAlgorithm(
                    original_time_series_decomp=univariate_decomps,
                    target_features=univariate_target_features,
                    num_generations=num_generations,
                    num_parents_mating=num_parents_mating,
                    sol_per_pop=solutions_per_pop,
                    num_genes=num_genes,
                    gene_space=legal_factor_values,
                    init_range_low=init_range_low,
                    init_range_high=init_range_high,
                    parent_selection_type=parent_selection_type,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    mutation_percent_genes=mutation_percent_genes,
                )

                ga_instance.run_genetic_algorithm()

                # Use GA solution to modify the trend and seasonal components
                factors, _, _ = ga_instance.get_best_solution()

                transformed_trend = manipulate_trend_component(
                    univariate_decomps.trend, factors[0], factors[1], factors[2], m=0
                )
                transformed_seasonal = manipulate_seasonal_component(
                    univariate_decomps.seasonal, factors[3]
                )
                # Reconstruct the transformed time series
                transformed_ts = (
                    transformed_trend + transformed_seasonal + univariate_decomps.resid
                )
                transformed_mts.append(transformed_ts)
                # Calculate features for the transformed time series
                transformed_mts_features.append(
                    [
                        trend_strength(transformed_trend, univariate_decomps.resid),
                        trend_slope(transformed_trend),
                        trend_linearity(transformed_trend),
                        seasonal_strength(
                            transformed_seasonal, univariate_decomps.resid
                        ),
                    ]
                )

                transformed_mts_factors.append(factors)

            GA_runs_mts.append(transformed_mts)
            GA_runs_features.append(transformed_mts_features)
            GA_runs_factors.append(transformed_mts_factors)

        logging.info(
            f"Successfully transformed the other univariate time series in the MTS. {num_GA_runs} genetic algorithm runs completed."
        )

        # NOTE: Only return one set of predicted features in this function
        predicted_features = predicted_features[0]

        return GA_runs_mts, GA_runs_features, GA_runs_factors, predicted_features

    def manual_transform_uts(self, mts_index, uts_index):
        assert self.manual_init_transform, "Manual transform not enabled"
        self.manual_transform_factors = [
            self.model_params["manual_transform"]["trend_det_factor"],
            self.model_params["manual_transform"]["trend_slope_factor"],
            self.model_params["manual_transform"]["trend_lin_factor"],
            self.model_params["manual_transform"]["seasonal_det_factor"],
        ]
        uts_decomp = self.mts_decomp[mts_index][uts_index]
        transformed_uts, transformed_uts_features, transformed_uts_decomp = (
            get_transformed_uts_with_features_and_decomps(
                uts_decomp,
                self.manual_transform_factors[0],
                self.manual_transform_factors[1],
                self.manual_transform_factors[2],
                self.manual_transform_factors[3],
            )
        )
        self.manual_transformed_uts = transformed_uts
        self.manual_transformed_uts_features = transformed_uts_features
        self.manual_transformed_uts_decomp = transformed_uts_decomp


def get_model_by_type(
    model_type: str,
    model_params: dict,
    mts_dataset: pd.DataFrame,
    mts_features: pd.DataFrame,
    mts_decomp: List[DecompResults],
    num_uts_in_mts: int,
    num_features_per_uts: int,
    manual_init_transform: bool,
) -> TimeSeriesTransformationModel:
    if model_type == "correlation_GA":
        return CorrelationGeneticAlgorithmModel(
            model_type,
            model_params,
            mts_dataset,
            mts_features,
            mts_decomp,
            num_uts_in_mts,
            num_features_per_uts,
            manual_init_transform,
        )
    else:
        raise ValueError(f"Model type {model_type} not supported")


# PLOTTING

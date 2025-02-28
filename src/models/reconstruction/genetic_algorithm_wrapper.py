import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

from src.models.reconstruction.genetic_algorithm import GeneticAlgorithm
from src.models.timeseries_transformation_model import TimeSeriesTransformationModel
from src.models.naive_correlation import CorrelationModel
from statsmodels.tsa.seasonal import DecomposeResult as DecompResults
from src.utils.logging_config import logger

from src.utils.features import (
    trend_strength,
    trend_slope,
    trend_linearity,
    seasonal_strength,
)
from src.utils.transformations import (
    manipulate_trend_component,
    manipulate_seasonal_component,
)


class GeneticAlgorithmWrapper(TimeSeriesTransformationModel):
    def __init__(
        self,
        model_type: str,
        model_params: dict,
        mts_dataset: pd.DataFrame,
        mts_features: pd.DataFrame,
        mts_decomp: List[DecompResults],
        num_uts_in_mts: int,
        num_features_per_uts: int,
    ):
        super().__init__(
            model_type, model_params, mts_dataset, mts_features, mts_decomp
        )
        self.num_features_per_uts = num_features_per_uts
        self.num_uts_in_mts = num_uts_in_mts
        self.data_set_size = len(mts_dataset)

    def fit(self) -> None:
        pass

    def transform(
        self,
        predicted_features: pd.DataFrame,
        original_mts_index: int,
        target_mts_index: int,
    ) -> Tuple[List, List, List, np.ndarray]:
        """Transform MTS using genetic algorithm.

        Args:
            model_input (_type_): Dataframe containing model input for a singe MTS
            original_mts_index (_type_): Index of the MTS that is to be transformed
            target_mts_index (_type_): Index of the transformation target MTS
            init_uts_index (_type_): Index of the UTS in the original MTS that we base the transformation on

        Returns:
            Tuple[List, List, List, np.ndarray]: List of transformed MTS, List of transformed MTS features, List of factors used for transformation, Predicted features
        """
        predicted_features = predicted_features.copy()

        ####  Get Genetic Algorithm Parameters ##########
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
        is_index_column = predicted_features.columns.str.contains("index")
        predicted_features = predicted_features.loc[:, ~is_index_column].to_numpy()
        predicted_features_reshape = predicted_features.reshape(3, 4)

        # NOTE May want to drop support for multiple GA runs?
        # Run genetic algorithm
        GA_runs_mts = []
        GA_runs_features = []
        GA_runs_factors = []

        logger.info("Starting genetic algorithm runs...")
        for _ in tqdm(range(num_GA_runs)):
            transformed_mts = []
            transformed_mts_features = []
            transformed_mts_factors = []

            for i in range(self.num_uts_in_mts):
                original_mts_decomp = self.mts_decomp[original_mts_index]
                univariate_decomps = original_mts_decomp[i]
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

        logger.info(
            f"Successfully transformed the other univariate time series in the MTS. {num_GA_runs} genetic algorithm runs completed."
        )

        return GA_runs_mts, GA_runs_features, GA_runs_factors

from typing import List, Tuple, override

import numpy as np
from tqdm import tqdm

from src.models.reconstruction.genetic_algorithm import GeneticAlgorithm
from src.models.reconstruction.reconstruction_model import ReconstructionModel
from src.utils.features import (
    decomp_and_features,
    seasonal_strength,
    trend_linearity,
    trend_slope,
    trend_strength,
)
from src.utils.logging_config import logger
from src.utils.transformations import (
    manipulate_seasonal_component,
    manipulate_trend_component,
)


class GeneticAlgorithmWrapper(ReconstructionModel):
    @override
    def __init__(self, model_params: dict, config: dict):
        self.model_params = model_params
        self.config = config
        self.trained = False

    @override
    def train(self, mts_dataset, X, y) -> None:
        num_features_per_uts: int = self.config["dataset_args"]["num_features_per_uts"]
        num_uts_in_mts: int = len(self.config["dataset_args"]["timeseries_to_use"])
        seasonal_period: int = self.config["stl_args"]["series_periodicity"]

        self.mts_dataset = mts_dataset
        self.mts_decomp, _ = decomp_and_features(
            mts=self.mts_dataset,
            num_features_per_uts=num_features_per_uts,
            series_periodicity=seasonal_period,
            decomps_only=True,
        )
        self.num_features_per_uts = num_features_per_uts
        self.num_uts_in_mts = num_uts_in_mts
        self.data_set_size = len(mts_dataset)
        self.trained = True
        pass

    @override
    def transform(
        self,
        predicted_features: np.ndarray,
        original_mts_indices: np.ndarray,
    ) -> Tuple[List, List]:
        """Transform MTS using genetic algorithm.

        Args:
            predicted_features: Dataframe containing predicted features for multiple MTS
            original_mts_indices: List of indices for the original MTS

        Returns:
            Tuple[List, List, List, np.ndarray]: List of transformed MTS, List of transformed MTS features, List of factors used for transformation, Predicted features
        """
        assert (
            self.trained
        ), "Reconstruction model needs to be trained before transformation"
        predicted_features = predicted_features.copy()

        ####  Get Genetic Algorithm Parameters ##########
        num_generations = self.model_params["num_generations"]
        num_parents_mating = self.model_params["num_parents_mating"]
        solutions_per_pop = self.model_params["solutions_per_population"]
        init_range_low = self.model_params["init_range_low"]
        init_range_high = self.model_params["init_range_high"]
        parent_selection_type = self.model_params["parent_selection_type"]
        crossover_type = self.model_params["crossover_type"]
        mutation_type = self.model_params["mutation_type"]
        mutation_percent_genes = self.model_params["mutation_percent_genes"]
        trend_det_factor_low = self.model_params["legal_values"]["trend_det_factor"][0]
        trend_det_factor_high = self.model_params["legal_values"]["trend_det_factor"][1]
        trend_slope_factor_low = self.model_params["legal_values"][
            "trend_slope_factor"
        ][0]
        trend_slope_factor_high = self.model_params["legal_values"][
            "trend_slope_factor"
        ][1]
        trend_lin_factor_low = self.model_params["legal_values"]["trend_lin_factor"][0]
        trend_lin_factor_high = self.model_params["legal_values"]["trend_lin_factor"][1]
        seasonal_det_factor_low = self.model_params["legal_values"][
            "seasonal_det_factor"
        ][0]
        seasonal_det_factor_high = self.model_params["legal_values"][
            "seasonal_det_factor"
        ][1]

        num_genes = self.num_features_per_uts

        # Constraint on GA solutions
        legal_factor_values = [
            np.linspace(trend_det_factor_low, trend_det_factor_high, 100),
            np.linspace(trend_slope_factor_low, trend_slope_factor_high, 100),
            np.linspace(trend_lin_factor_low, trend_lin_factor_high, 100),
            np.linspace(seasonal_det_factor_low, seasonal_det_factor_high, 100),
        ]

        # Calculate the number of MTS we need to process
        number_of_mts = len(original_mts_indices)
        # Reshape predicted features to have the right shape for each MTS
        # (num_mts, num_uts_in_mts, num_features_per_uts)
        predicted_features_reshape = predicted_features.reshape(
            number_of_mts,
            self.num_uts_in_mts,
            self.num_features_per_uts,
        )

        all_mts_transformed = []
        all_mts_transformed_features = []
        all_mts_transformed_factors = []

        # For each original MTS index and its corresponding predicted features
        for mts_idx, original_mts_index in tqdm(
            enumerate(original_mts_indices), total=len(original_mts_indices)
        ):
            transformed_mts = []
            transformed_mts_features = []
            transformed_mts_factors = []

            # Process each univariate time series within the MTS
            for i in range(self.num_uts_in_mts):
                original_mts_decomp = self.mts_decomp[original_mts_index]
                univariate_decomps = original_mts_decomp[i]
                univariate_target_features = predicted_features_reshape[mts_idx][i]

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
                    univariate_decomps.trend,
                    factors[0],
                    factors[1],
                    factors[2],
                    m=0,
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

            all_mts_transformed.append(transformed_mts)
            all_mts_transformed_features.append(transformed_mts_features)
            all_mts_transformed_factors.append(transformed_mts_factors)

        logger.info(
            f"Successfully transformed {number_of_mts} multivariate time series."
        )

        return (
            all_mts_transformed,
            all_mts_transformed_features,
        )

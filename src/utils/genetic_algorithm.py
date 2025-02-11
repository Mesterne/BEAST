import numpy as np
from pygad import GA

from statsmodels.tsa.seasonal import DecomposeResult  # For type hinting

from .transformations import (
    manipulate_seasonal_component,
    manipulate_trend_component,
)
from .features import (
    trend_strength,
    trend_slope,
    trend_linearity,
    seasonal_strength,
)


class GeneticAlgorithm:

    def __init__(
        self,
        original_time_series_decomp: DecomposeResult,
        target_features: np.ndarray,
        num_generations: int,
        num_parents_mating: int,
        sol_per_pop: int,
        num_genes: int,
        gene_space: list,
        init_range_low: int,
        init_range_high: int,
        parent_selection_type: str = "sss",
        crossover_type: str = "single_point",
        mutation_type: str = "random",
        mutation_percent_genes: float = 25,
    ):
        self.original_time_series_decompositions = original_time_series_decomp
        self.target_features = target_features
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.sol_per_pop = sol_per_pop
        self.num_genes = num_genes
        self.gene_space = gene_space
        self.init_range_low = init_range_low
        self.init_range_high = init_range_high
        self.parent_selection_type = parent_selection_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.mutation_percent_genes = mutation_percent_genes

    def euclidean_distance(self, x, y):
        return np.linalg.norm(x - y)

    # NOTE: Based on this stackoverflow answer:
    # https://stackoverflow.com/questions/69544556/passing-arguments-to-pygad-fitness-function
    def fintness_function_factory(self):

        # PyGAD fitness function only accepts 3 arguments:
        # ga_instance, solution, solution_idx
        def fitness_func(ga_instance, solution, solution_idx):
            # NOTE: Assuming solution is the list of factors [f, g, h, k]

            original_trend = self.original_time_series_decompositions.trend
            original_seasonal = self.original_time_series_decompositions.seasonal
            original_residual = self.original_time_series_decompositions.resid

            # NOTE: The factor m is excluded from solution for now.
            # It caused some issues when included.
            solution_trend = manipulate_trend_component(
                original_trend, solution[0], solution[1], solution[2], m=0
            )
            solution_seasonal = manipulate_seasonal_component(
                original_seasonal, solution[3]
            )

            solution_features = np.array(
                [
                    trend_strength(solution_trend, original_residual),
                    trend_slope(solution_trend),
                    trend_linearity(solution_trend),
                    seasonal_strength(solution_seasonal, original_residual),
                ]
            )

            # NOTE: Multiplying by -1 because PyGAD maximizes the fitness function
            return -1 * self.euclidean_distance(
                solution_features, np.array(self.target_features)
            )

        return fitness_func

    def run_genetic_algorithm(self):
        self.ga_instance = GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            sol_per_pop=self.sol_per_pop,
            num_genes=self.num_genes,
            gene_space=self.gene_space,
            init_range_low=self.init_range_low,
            init_range_high=self.init_range_high,
            parent_selection_type=self.parent_selection_type,
            crossover_type=self.crossover_type,
            mutation_type=self.mutation_type,
            mutation_percent_genes=self.mutation_percent_genes,
            fitness_func=self.fintness_function_factory(),
            suppress_warnings=True,
        )

        self.ga_instance.run()

    def get_best_solution(self):
        if self.ga_instance is None:
            raise Exception("Genetic algorithm has not been run yet.")
        return self.ga_instance.best_solution()

    def plot_fitness(self):
        if self.ga_instance is None:
            raise Exception("Genetic algorithm has not been run yet.")
        self.ga_instance.plot_fitness()

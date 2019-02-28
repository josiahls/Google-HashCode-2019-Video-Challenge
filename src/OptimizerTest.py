"""
State:



"""
from bayes_opt import BayesianOptimization, JSONLogger, Events
from src.JosiahGeneticAlgTest import GeneticAlgorithm

if __name__ == '__main__':
    X = "Hello world!"

    # We setup the optimization function
    def maximization_function(number_of_children,
                              number_of_splits,
                              number_of_parents,
                              mutation_rate,
                              min_mutations,
                              max_mutations,
                              population_size,
                              loss_exp,
                              max_generations):
        alg = GeneticAlgorithm(number_of_children=number_of_children,
                               number_of_splits=number_of_splits,
                               number_of_parents=number_of_parents,
                               number_parents_to_keep=None,
                               mutation_rate=mutation_rate,
                               parents_generation_rate=0.2,
                               parent_keep_rate=0.2,
                               min_mutations=min_mutations,
                               max_mutations=max_mutations,
                               population_size=population_size,
                               iterations=20,
                               loss_exp=loss_exp,
                               max_generations=max_generations,
                               max_fitness=len(X),
                               thresh_fitness=len(X),
                               size=len(X),
                               target=X)
        try:
            fitness = alg.search()
        except ValueError:
            return 0
        print(f'Fitness strength is: {fitness}')
        return fitness

    # Setup the hyper-parameters and the bayesian optimization model
    p_bounds = {'number_of_children': (1, len(X)),
                'number_of_splits': (1, len(X)),
                'number_of_parents': (1, len(X)),
                'mutation_rate': (0, 1),
                'min_mutations': (0, len(X)),
                'max_mutations': (0, len(X)),
                'population_size': (0, 100 * len(X)),
                'loss_exp': (1, len(X)),
                'max_generations': (1, len(X))}

    optimizer = BayesianOptimization(
        f=maximization_function,
        pbounds=p_bounds,
        verbose=2,
        random_state=1,
    )

    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )













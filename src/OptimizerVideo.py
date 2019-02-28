"""
State:



"""
from bayes_opt import BayesianOptimization, JSONLogger, Events
from src.JosiahGeneticAlgVideo import GeneticAlgorithm

if __name__ == '__main__':

    alg = GeneticAlgorithm(number_of_children=2,
                           number_of_splits=4,
                           number_of_parents=3,
                           number_parents_to_keep=None,
                           mutation_chance=0.2,
                           parents_generation_rate=0.2,
                           parent_keep_rate=0.2,
                           min_mutations=1,
                           max_mutations=1,
                           population_size=400,
                           iterations=2,
                           loss_exp=2.0,
                           max_generations=100,
                           thresh_fitness=10 ** 100,
                           bi_max_mutations=0.1,
                           bi_min_mutations=0.5,
                           max_inner_splits=5,
                           max_swap_num=5,
                           out_file_name_prefix='me')

    # We setup the optimization function
    def maximization_function(number_of_children, number_of_splits, number_of_parents, number_parents_to_keep,
                              mutation_chance, parents_generation_rate, parent_keep_rate, min_mutations,
                              max_mutations, population_size, loss_exp, max_generations,
                              bi_max_mutations, bi_min_mutations, max_inner_splits, max_swap_num,
                              new_parents_per_generation):

        alg.reset_params(number_of_children=number_of_children,
                         number_of_splits=number_of_splits,
                         number_of_parents=number_of_parents,
                         number_parents_to_keep=number_parents_to_keep,
                         mutation_chance=mutation_chance,
                         parents_generation_rate=parents_generation_rate,
                         parent_keep_rate=parent_keep_rate,
                         min_mutations=min_mutations,
                         max_mutations=max_mutations,
                         population_size=population_size,
                         loss_exp=loss_exp,
                         max_generations=max_generations,
                         bi_max_mutations=bi_max_mutations,
                         bi_min_mutations=bi_min_mutations,
                         max_inner_splits=max_inner_splits,
                         max_swap_num=max_swap_num,
                         new_parents_per_generation=new_parents_per_generation)

        try:
            fitness = alg.search(time_limit=5)
        except ValueError:
            return 1
        print(f'Fitness strength is: {fitness}')
        return fitness


    # Setup the hyper-parameters and the bayesian optimization model
    p_bounds = {'number_of_children': (2, alg.v),
                'number_of_splits': (1, alg.v),
                'number_of_parents': (2, alg.v),
                'number_parents_to_keep': (0, alg.v),
                'mutation_chance': (0, 1),
                'parents_generation_rate': (0, 1),
                'parent_keep_rate': (0.1, 1),
                'min_mutations': (0, alg.v),
                'max_mutations': (0, alg.v),
                'population_size': (0, 100 * alg.v),
                'loss_exp': (1, alg.v),
                'max_generations': (1, alg.v),
                'bi_max_mutations': (1, alg.v),
                'bi_min_mutations': (1, alg.v),
                'max_inner_splits': (1, alg.v),
                'max_swap_num': (1, alg.v),
                'new_parents_per_generation': (1, alg.v)}

    optimizer = BayesianOptimization(
        f=maximization_function,
        pbounds=p_bounds,
        verbose=2,
        random_state=1,
    )

    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=20,
        n_iter=50,
    )

    print('The max is: ')
    print(optimizer.max)

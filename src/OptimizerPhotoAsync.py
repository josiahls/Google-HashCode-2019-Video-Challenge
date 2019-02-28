from datetime import datetime
import sys
import threading

from bayes_opt import BayesianOptimization, JSONLogger, Events
from AlgPhotos import GeneticAlgorithm

results = []
optimizer_id = 0
optimizer_id_queue = 0


def run_optimization():
    global optimizer_id, optimizer_id_queue
    optimizer_id += 1
    local_id = optimizer_id
    print(f'Optimizer {optimizer_id} Starting')

    while local_id != 1 and local_id != optimizer_id_queue + 1:
        pass

    # Let the thread sleep
    sys.stdin = open('../photo_input/a_example.txt')  # Needs

    alg = GeneticAlgorithm(number_of_children=2,
                           number_of_splits=4,
                           number_of_parents=3,
                           mutation_chance=0.2,
                           parents_generation_rate=0.2,
                           parent_keep_rate=0.2,
                           min_mutations=1,
                           max_mutations=1,
                           population_size=400,
                           iterations=2,
                           loss_exp=1.0,
                           max_generations=100,
                           bin_max_mutations=0.1,
                           bin_min_mutations=0.5,
                           max_inner_splits=5,
                           max_swap_num=5,
                           out_file_name_prefix='me_'+str(local_id))
    sys.stdin.close()

    optimizer_id_queue += 1

    # We setup the optimization function
    def maximization_function(number_of_children, number_of_splits, number_of_parents,
                              mutation_chance, parents_generation_rate, parent_keep_rate, min_mutations,
                              max_mutations, population_size, max_generations,
                              bin_max_mutations, bin_min_mutations, max_inner_splits, max_swap_num):

        alg.reset_params(number_of_children=number_of_children,
                         number_of_splits=number_of_splits,
                         number_of_parents=number_of_parents,
                         mutation_chance=mutation_chance,
                         parents_generation_rate=parents_generation_rate,
                         parent_keep_rate=parent_keep_rate,
                         min_mutations=min_mutations,
                         max_mutations=max_mutations,
                         population_size=population_size,
                         max_generations=max_generations,
                         bin_max_mutations=bin_max_mutations,
                         bin_min_mutations=bin_min_mutations,
                         max_inner_splits=max_inner_splits,
                         max_swap_num=max_swap_num)

        try:
            fitness = alg.search(time_limit=5)
        except ValueError:
            return 1
        print(f'Fitness strength is: {fitness}')
        return fitness

    p_bounds = {'number_of_children': (2, 3),
                'number_of_splits': (1, 3),
                'number_of_parents': (2, 3),
                'mutation_chance': (0, 1),
                'parents_generation_rate': (0, 1),
                'parent_keep_rate': (0.1, 1),
                'min_mutations': (1, 2),
                'max_mutations': (1, 2),
                'population_size': (10, 100 * 2),
                'max_generations': (1, 100),
                'bin_max_mutations': (1, 4),
                'bin_min_mutations': (1, 2),
                'max_inner_splits': (1, 3),
                'max_swap_num': (1, 3)}

    optimizer = BayesianOptimization(
        f=maximization_function,
        pbounds=p_bounds,
        verbose=2,
        random_state=1,
    )

    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTMIZATION_END, logger)

    optimizer.maximize(
        init_points=25,
        n_iter=50,
    )

    print('The max is: ')
    print(optimizer.max)
    global results
    results.append((optimizer.max['target'], optimizer.max['params']))
    print(f'Optimizer {optimizer_id} Ending with target {optimizer.max["target"]}')

if __name__ == '__main__':
    results = []
    optimizer_id = 0

    targets = (
        run_optimization,
        run_optimization,
        run_optimization,
        run_optimization,
        run_optimization,
        run_optimization,
        run_optimization,
        run_optimization,
        # run_optimization
    )
    optimizer_threads = []
    for target in targets:
        optimizer_threads.append(threading.Thread(target=target))
        optimizer_threads[-1].daemon = True
        optimizer_threads[-1].start()

    results = []
    for optimizer_thread in optimizer_threads:
        optimizer_thread.join()

    for result in results:
        print(result[0], "found a maximum value of: {}".format(result[1]))

    now = datetime.now()
    fout = open("logs/hyper_params" + now.strftime("%Y%m%d-%H%M%S.%f") + ".out", 'w')

    towrite = "\n"
    for result in results:
        towrite += str(result) + '\n'
    fout.write(towrite)
    fout.close()

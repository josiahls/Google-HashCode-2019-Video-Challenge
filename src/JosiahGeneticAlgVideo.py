import random
import sys
import time
import math
from datetime import datetime


class GeneticAlgorithm:
    def __init__(self, number_of_children,
                 number_of_splits,
                 number_of_parents,
                 number_parents_to_keep,
                 mutation_chance,
                 parents_generation_rate,
                 parent_keep_rate,
                 bi_min_mutations,
                 bi_max_mutations,
                 min_mutations,
                 max_mutations,
                 max_inner_splits,
                 max_swap_num,
                 population_size,
                 iterations,
                 loss_exp,
                 thresh_fitness,
                 max_generations,
                 print_generation=1,
                 out_file_name_prefix='a',
                 new_parents_per_generation=None):
        """

        :param number_of_children:
        :param number_of_splits:
        :param number_of_parents:
        :param number_parents_to_keep:
        :param mutation_chance:
        :param parents_generation_rate:
        :param parent_keep_rate:
        :param bi_min_mutations:
        :param bi_max_mutations:
        :param min_mutations:
        :param max_mutations:
        :param max_inner_splits:
        :param max_swap_num:
        :param population_size:
        :param iterations:
        :param loss_exp:
        :param thresh_fitness:
        :param max_generations:
        :param print_generation:
        :param out_file_name_prefix:
        :param new_parents_per_generation:
        """

        random.seed(1)
        """ Testing Params """
        self.print_generation = print_generation  # show progress by generation with winning one
        self.out_file_name_prefix = out_file_name_prefix

        """ Load Program Data """
        self.v, self.e, self.r, self.c, self.x = list(map(int, input().strip().split()))
        self.video_sizes = list(map(int, input().strip().split()))  # they start at 0 so list with 0 index is ok
        assert len(self.video_sizes) == self.v
        self.e_cache_times = [dict() for _ in range(self.e)]
        self.cd_latencies = [None for _ in range(self.e)]

        for i in range(self.e):
            latency, num_caches = list(map(int, input().strip().split()))
            self.cd_latencies[i] = latency

            for j in range(num_caches):
                current_cache_node, current_cache_latency = list(map(int, input().strip().split()))
                self.e_cache_times[i][current_cache_node] = current_cache_latency

        self.requests = []
        self.min_requests = 0
        for i in range(self.r):
            request_list = list(map(int, input().strip().split()))
            self.requests.append(request_list)
            self.min_requests += request_list[2]

        self.geneset_videos = range(self.v)

        """ Changeable Params """
        self.max_generations = max_generations
        self.iterations = iterations
        self.loss_exp = loss_exp

        self.bi_min_mutations = bi_min_mutations  # on average, when a bin is mutated, remove 20% of the videos (regardless of capacity)
        self.bi_max_mutations = bi_max_mutations  # change this lower for large values

        self.max_swap_num = max_swap_num
        self.max_inner_splits = max_inner_splits
        self.max_mutations = max_mutations
        self.min_mutations = min_mutations
        
        self.number_of_splits = number_of_splits
        self.number_of_children = number_of_children
        self.number_of_parents = number_of_parents

        self.mutation_chance = mutation_chance
        self.population_size = population_size

        self.parent_keep_rate = parent_keep_rate
        self.parents_generation_rate = parents_generation_rate

        self.thresh_fitness = thresh_fitness

        """ Some of these parameters are better set automatically """
        _auto_size = int(self.parent_keep_rate * self.population_size)
        self.number_parents_to_keep = _auto_size if number_parents_to_keep is None else number_parents_to_keep
        _auto_size = int(self.parents_generation_rate * self.population_size)
        self.new_parents_per_generation = _auto_size if number_parents_to_keep is None else new_parents_per_generation

    def reset_params(self, max_generations, loss_exp, bi_min_mutations,
                     bi_max_mutations, max_swap_num, max_inner_splits, max_mutations,
                     min_mutations, number_of_splits, number_of_children,
                     number_of_parents, mutation_chance, population_size, 
                     parent_keep_rate, parents_generation_rate,
                     new_parents_per_generation, number_parents_to_keep=0, thresh_fitness=None, iterations=None):
        
        self.max_generations = int(max_generations)
        if iterations is not None:
            self.iterations = int(iterations)
        self.loss_exp = int(loss_exp)

        self.bi_min_mutations = int(bi_min_mutations)  # on average, when a bin is mutated, remove 20% of the videos (regardless of capacity)
        self.bi_max_mutations = int(bi_max_mutations)  # change this lower for large values

        self.max_swap_num = int(max_swap_num)
        self.max_inner_splits = int(max_inner_splits)
        self.max_mutations = int(max_mutations)
        self.min_mutations = int(min_mutations)

        self.number_of_splits = int(number_of_splits)
        self.number_of_children = int(number_of_children)
        self.number_of_parents = int(number_of_parents)

        self.mutation_chance = mutation_chance
        self.population_size = int(population_size)

        self.parent_keep_rate = parent_keep_rate
        self.parents_generation_rate = parents_generation_rate

        self.thresh_fitness = int(thresh_fitness) if thresh_fitness is not None else self.thresh_fitness

        number_parents_to_keep = None if number_parents_to_keep == 0 else int(number_parents_to_keep)
        """ Some of these parameters are better set automatically """
        _auto_size = int(self.parent_keep_rate * self.population_size)
        self.number_parents_to_keep = _auto_size if number_parents_to_keep is None else int(number_parents_to_keep)
        _auto_size = int(self.parents_generation_rate * self.population_size)
        self.new_parents_per_generation = _auto_size if number_parents_to_keep is None else int(new_parents_per_generation)

    def fitness(self, state):
        mssaved = 0
        for req in self.requests:
            vid, endpt, nreqs = req

            dcreqtime = self.cd_latencies[endpt]
            minreqtime = dcreqtime
            for ccacheid in self.e_cache_times[endpt]:
                if vid in state[ccacheid]:
                    minreqtime = min(self.e_cache_times[endpt][ccacheid], minreqtime)

            mssaved += (dcreqtime - minreqtime) * nreqs

        return mssaved * 1000 / self.min_requests

    def generate_parent(self):
        parent = []

        for i in range(self.c):
            csize = 0
            cset = set()
            while 1:
                video = random.choice(self.geneset_videos)
                newsize = self.video_sizes[video]

                if csize + newsize <= self.x:
                    cset.add(video)
                    csize += newsize
                else:
                    break  # don't try to pack it at this stage

            parent.append(cset)

        assert self.is_valid_element(parent)
        return parent

    def is_valid_element(self, state):
        return len(state) == self.c and all(sum(self.video_sizes[e] for e in binset) <= self.x for binset in state)

    def breed(self, states):
        P = len(states)
        assert P == self.number_of_parents

        children = []

        N = len(states[0])  # or any other in the array
        assert N == self.c

        for c in range(self.number_of_children):

            # Crossover
            splitlocs = [0] + sorted([random.choice(range(N)) for i in range(self.number_of_splits - 1)]) + [
                N]  # doesn't guarantee unique

            splitchoices = [random.choice(range(P))]
            for i in range(1, self.number_of_splits):
                # random excluding prior choice:
                choice = random.choice(range(P - 1))
                if choice <= splitchoices[i - 1]: choice += 1
                splitchoices.append(choice)

            # print("making child: ", states, P, splitlocs, splitchoices)

            assert len(splitlocs) == self.number_of_splits + 1, f'This locs {len(splitlocs)} != {self.number_of_splits + 1}'
            assert len(splitchoices) == self.number_of_splits, f'This choices {len(splitchoices)} != {self.number_of_splits}'
            # cchild = ''.join(strs[splitchoices[i]][splitlocs[i]:splitlocs[i+1]] for i in range(self.number_of_splits))
            cchild = []
            for i in range(self.number_of_splits):
                if not isinstance(states[splitchoices[i]], list) and (isinstance(states[splitchoices[i]][j], set) for j
                                                                      in range(len(states[splitchoices[i]]))):
                    assert 0
                    # print(strs[splitchoices[i]], strs)
                cchild += states[splitchoices[i]][splitlocs[i]:splitlocs[i + 1]]

            # do again:
            cchild = []

            # Crossover at deeper level
            for h in range(self.c):
                csize = 0
                cset = set()

                numinnersplits = random.randint(1, self.max_inner_splits)

                splitchoices = [random.choice(range(P))]
                for i in range(1, numinnersplits):
                    # random excluding prior choice:
                    choice = random.choice(range(P - 1))
                    if choice <= splitchoices[i - 1]: choice += 1
                    splitchoices.append(choice)

                for i in range(numinnersplits):  # random this 1 to max
                    curchoices = list(states[splitchoices[i]][h])
                    if len(curchoices) > 0:
                        while 1:
                            video = random.choice(curchoices)
                            newsize = self.video_sizes[video]

                            if csize + newsize <= (i / float(
                                    numinnersplits)) * self.x:  # yeah, there is a slight bias the first one sends less but this evens out as the selection is random
                                cset.add(video)
                                csize += newsize
                            else:
                                break  # don't try to pack it at this stage

                cchild.append(cset)

            # Swaps
            '''
            swaps = random.randint(0, self.max_swap_num)

            for i in range(self.max_swap_num):
                participants = random.sample(NUMSWAPPARTICIPANTRATE)
            '''

            # Mutate
            if random.random() < self.mutation_chance:
                cchild = self.mutate(cchild)

            # print(cchild)
            assert self.is_valid_element(cchild), 'This child is invalid'

            children.append(cchild)

        return children

    def mutate_bin(self, binset, binmutrate):
        # return a mutated bin; helper for mutate()
        numtoremove = math.floor(binmutrate * len(binset))
        toremove = random.sample(binset, numtoremove)
        for elremove in toremove:  # use the right variable
            binset.remove(elremove)

        csize = sum(self.video_sizes[e] for e in binset)
        while 1:
            video = random.choice(self.geneset_videos)
            newsize = self.video_sizes[video]

            if csize + newsize <= self.x:
                binset.add(video)
                csize += newsize
            else:
                break  # don't try to pack it at this stage

        return binset

    def mutate(self, state):
        nummutations = random.randint(self.min_mutations, self.max_mutations + 1)

        inds = random.sample(range(len(state)), nummutations)

        for ind in inds:
            binmutrate = random.uniform(self.bi_min_mutations, self.bi_max_mutations)
            state[ind] = self.mutate_bin(state[ind], binmutrate)

        swaps = random.randint(0, self.max_swap_num)  # this could be a hyperparam function or distribution

        for i in range(swaps):
            bins = random.sample(range(self.e), 2)
            state[bins[0]], state[bins[1]] = state[bins[1]], state[bins[0]]

        return state  # to be consistent for other problems, we return reference to the state

    def lateral_transfer(self, parents):
        return parents

    def select_children(self, children):
        return children[:int(self.population_size * 0.8)] + children[
                                               int(self.population_size * 0.5): int(self.population_size * 0.5) + self.population_size - int(self.population_size * 0.8)]

    def search(self, time_limit=None):
        totaltime = 0
        numconverged = 0

        print("Testing parameters...")
        current_iter = 0
        for f in range(self.iterations):
            current_iter = f
            starttime = time.time()

            parents = [self.generate_parent() for i in range(100)]
            # print(parents)

            for g in range(self.max_generations):

                children = []
                # children = [breed([random.choice(parents) for _ in range(self.number_of_parents)]) for d in range(2 * self.population_size)]
                for d in range(2 * self.population_size):
                    children += self.breed([random.choice(parents) for _ in range(self.number_of_parents)])

                children += parents[:self.number_parents_to_keep]
                children += [self.generate_parent() for i in range(self.new_parents_per_generation)]
                children = sorted(children, key=lambda child: self.fitness(child), reverse=True)

                # we should keep the ones that are more different and still ok, but for now, just pull from top 80% and bottom 20%
                children = self.select_children(children)

                maxfitness = self.fitness(children[0])

                print("Gen " + str(g) + " max fitness " + str(maxfitness) + " with " + str(children[0]))
                parents = children

                if self.fitness(children[0]) >= self.thresh_fitness:
                    numconverged += 1
                    if self.print_generation:
                        print("Finish gen " + str(g) + " max fitness " + str(maxfitness) + " with " + str(children[0]))
                    break

            else:
                numconverged += 0

            # print(parents)

            endtime = time.time()

            elapsedtime = endtime - starttime
            totaltime += elapsedtime
            if time_limit is not None and time_limit < elapsedtime:
                print('Reached time limit')
                current_iter = current_iter if current_iter != 0 else 1
                break

        convergedrate = numconverged / current_iter
        avgtime = totaltime / current_iter

        strength = avgtime * (current_iter ** self.loss_exp)

        print(strength)

        now = datetime.now()
        fout = open("logs/ans_" + self.out_file_name_prefix + now.strftime("%Y%m%d-%H%M%S.%f") + ".out", 'w')

        beststate = children[0]
        towrite = str(self.c) + "\n"
        for i in range(self.c):
            towrite += str(i) + " " + " ".join(str(e) for e in list(beststate[i]))
        fout.write(towrite)
        fout.close()

        return 1 / strength  # change this to return the fitness after 5 seconds


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
                           max_generations=2,
                           thresh_fitness=10 ** 100,
                           bi_max_mutations=0.1,
                           bi_min_mutations=0.5,
                           max_inner_splits=5,
                           max_swap_num=5,
                           out_file_name_prefix='me_'+str(0))

    param_dict = {}

    alg.reset_params(**param_dict)


    print(f'Fitness strength is: {alg.search()}')

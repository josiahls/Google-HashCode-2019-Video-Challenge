import random
import time
import math
from datetime import datetime
import sys



FNAME_PREFIX = '../photo_input/a_example.txt'


class GeneticAlgorithm:
    def __init__(self, number_of_children,
                 number_of_splits,
                 number_of_parents,
                 mutation_chance,
                 parents_generation_rate,
                 parent_keep_rate,
                 bin_min_mutations,
                 bin_max_mutations,
                 min_mutations,
                 max_mutations,
                 max_inner_splits,
                 max_swap_num,
                 population_size,
                 iterations,
                 max_generations,
                 loss_exp=1,
                 print_generation=1,
                 out_file_name_prefix='photoinput/a.in'):
        """

        :param number_of_children:
        :param number_of_splits:
        :param number_of_parents:
        :param mutation_chance:
        :param parents_generation_rate:
        :param parent_keep_rate:
        :param bin_min_mutations:
        :param bin_max_mutations:
        :param min_mutations:
        :param max_mutations:
        :param max_inner_splits:
        :param max_swap_num:
        :param population_size:
        :param iterations:
        :param loss_exp:
        :param max_generations:
        :param print_generation:
        :param out_file_name_prefix:
        """

        random.seed(1)

        """ Testing Params """
        self.print_generation = print_generation  # show progress by generation with winning one
        self.out_file_name_prefix = out_file_name_prefix

        """ Load Program Data """
        self.N = list(map(int, input().strip().split()))[0]
        self.TAGS_S = []
        self.TYPE = []
        self.NUMV = 0
        self.NUMH = 0
        self.HINDS = []
        self.VINDS = []

        self.TAGSET = set()
        TAGSET2ID = dict()
        for i in range(self.N):
            lline = list(map(str, input().strip().split()))
            self.TAGS_S.append(lline[2:])
            for e in lline[2:]:
                self.TAGSET.add(e)
            for tag in self.TAGS_S[i]:
                tag = 1
            self.TYPE.append(lline[0])
            if lline[0] == 'H':
                self.NUMH += 1
                self.HINDS.append(i)
            elif lline[0] == 'V':
                self.NUMV += 1
                self.VINDS.append(i)

        assert self.NUMV + self.NUMH == self.N

        x = 0
        for tag in self.TAGSET:
            TAGSET2ID[tag] = x
            x += 1

        self.TAGS = []
        for i in range(len(self.TAGS_S)):
            self.TAGS.append(set(TAGSET2ID[t] for t in self.TAGS_S[i]))

        print("Tags", self.TAGS)

        '''
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
        '''

        """ Changeable Params """
        self.max_generations = max_generations
        self.iterations = iterations
        self.loss_exp = loss_exp

        self.bin_min_mutations = bin_min_mutations  # on average, when a bin is mutated, remove 20% of the videos (regardless of capacity)
        self.bin_max_mutations = bin_max_mutations  # change this lower for large values

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

        self.thresh_fitness = 10**100

        """ Some of these parameters are better set automatically """
        self.number_parents_to_keep = int(self.parent_keep_rate * self.population_size)
        self.new_parents_per_generation = int(self.parents_generation_rate * self.population_size)


    def reset_params(self, max_generations, bin_min_mutations,
                     bin_max_mutations, max_swap_num, max_inner_splits, max_mutations,
                     min_mutations, number_of_splits, number_of_children,
                     number_of_parents, mutation_chance, population_size,
                     parent_keep_rate, parents_generation_rate,  iterations=None):

        self.max_generations = int(max_generations)
        if iterations is not None:
            self.iterations = int(iterations)

        self.bin_min_mutations = int(bin_min_mutations)  # on average, when a bin is mutated, remove 20% of the videos (regardless of capacity)
        self.bin_max_mutations = int(bin_max_mutations)  # change this lower for large values

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

        """ Some of these parameters are better set automatically """
        self.number_parents_to_keep = int(self.parent_keep_rate * self.population_size)
        self.new_parents_per_generation = int(self.parents_generation_rate * self.population_size)


    def fithelper(self, twoimgs):
        #print("t",twoimgs)
        tagsl = set()
        for e in twoimgs[0]:
            tagsl.add(e)
        tagsr = set()
        for e in twoimgs[1]:
            tagsr.add(e)
        tagsboth = tagsl & tagsr

        return min(len(tagsboth), len(tagsl.difference(tagsr)), len(tagsr.difference(tagsl)))

    def fitness(self, state):
        totalfitness = 0
        for i in range(len(state)-1):
            totalfitness += self.fithelper(state[i:i+2])

        return totalfitness

    def help_of_element(self, state, i):
        return (self.fitness(state[i-1:i+1]) + self.fitness(state[i:i+2]))/2

    def generate_parent(self):
        parent = []
        used = set()
        for i in range(self.N):
            rand = random.random()
            if rand < (self.NUMH/(self.NUMH + int(self.NUMV/2))):

                touse = random.choice(self.HINDS)
                works = 0
                count = 0

                while count < self.N * 8:

                    count += 1
                    touse = random.choice(self.HINDS)
                    if touse not in used:
                        works = 1
                        break
                if not works:
                    break
                used.add(touse)
                parent.append( (touse, ) )
            else:
                touse = random.sample(self.VINDS, 2)

                works = 0
                count = 0
                while count < self.N * 14:
                    touse = random.sample(self.VINDS, 2)
                    count += 1

                    if not any(e in used for e in touse):
                        works = 1
                        break

                if not works:
                    break
                for touse_e in touse:
                    used.add(touse_e)
                parent.append( tuple(touse) )
        #print("made ",parent)

        assert self.is_valid_element(parent)

        for e in parent:
            assert isinstance(e, tuple)
        return parent

    def is_valid_element(self, state):
        #return 1
        flatlist = [item for sublist in state for item in sublist]
        N = self.N
        return len(state) <= self.N and all(len(state[i]) == 1 and all(self.TYPE[e] == 'H' for e in state[i]) or len(state[i]) == 2 and all(self.TYPE[e] == 'V' for e in state[i]) for i in range(len(state))) and len(flatlist) == len(set(flatlist))
        # also could check no element exceeds N-1
        #return len(state) == self.c and all(sum(self.video_sizes[e] for e in binset) <= self.x for binset in state)

    def breed(self, states, helpdictavg, helpdictmax):
        #print(states)
        P = len(states)
        assert P == self.number_of_parents

        children = []

        N = self.N # for this function

        parentsrev = []
        for i in range(P):
            cset = dict()
            for j in range(len(states[i])):
                cset[j] = states[i][j]
            parentsrev.append(cset)

        for c in range(self.number_of_children):

            cchild = []
            used = [0 for i in range(N)]

            # Crossover
            # splitlocs1 = [0] + sorted([random.choice(range(N)) for i in range(self.number_of_splits - 1)]) + [N]  # doesn't guarantee unique
            # streaklengths = [splitlocs1[i] - splitlocs1[i-1] for i in range(1,N)]

            curparent = random.choice(states)
            #print("parent", curparent)
            curindex = random.choice(range(0, len(curparent)))

            while True:

                CHANCETOSWITCHPARENTS = 0.1
                doswitch = random.random() > CHANCETOSWITCHPARENTS
                tries = 0
                worked = False

                curparent = random.choice(states)  # same 'macro' as the other two lines
                curindex = random.choice(range(len(curparent)))

                while tries < len(states) * 8:
                    tries += 1

                    if curindex < len(curparent) and not any(used[e] for e in curparent[curindex]):
                        #print("r")
                        worked = 1
                        break
                    else:
                        curparent = random.choice(states) # same 'macro' as the other two lines
                        curindex = random.choice(range(len(curparent)))
                    doswitch = False

                if tries < len(states) * 8:
                    worked = True

                if not worked:
                    break

                for e in curparent[curindex]:
                    used[e] = 1
                cchild.append(curparent[curindex])
                #print(cchild)

            # Try as a alternative marching forward and just removing the picture
            '''
            splitchoices = [random.choice(range(P))]
            for i in range(1, self.number_of_splits):
                # random excluding prior choice:
                choice = random.choice(range(P - 1))
                if choice <= splitchoices[i - 1]: choice += 1
                splitchoices.append(choice)
            '''

                # If the next value is already used, we must switch to a different chain.  Take a weighted probability of where that element appears on other chains.
                # after we can pull no more elements, stop and return this child


            # print("making child: ", states, P, splitlocs, splitchoices)

            # assert len(splitlocs) == self.number_of_splits + 1, f'This locs {len(splitlocs)} != {self.number_of_splits + 1}'
            # assert len(splitchoices) == self.number_of_splits, f'This choices {len(splitchoices)} != {self.number_of_splits}'
            # cchild = ''.join(strs[splitchoices[i]][splitlocs[i]:splitlocs[i+1]] for i in range(self.number_of_splits))
            # # find weak transitions.  First find all the transitions.  Then take from these
            '''
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
            '''


            # Swaps
            '''
            swaps = random.randint(0, self.max_swap_num)

            for i in range(self.max_swap_num):
                participants = random.sample(NUMSWAPPARTICIPANTRATE)
            '''

            # Mutate
            #if random.random() < self.mutation_chance:
            #    cchild = self.mutate(cchild)

            # print(cchild)
            assert self.is_valid_element(cchild), str(cchild)#'This child is invalid'

            children.append(cchild)

        #print(children)
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

        '''

        :param state:
        :return:
        '''
        '''
        nummutations = random.randint(self.min_mutations, self.max_mutations + 1)

        inds = random.sample(range(len(state)), nummutations)

        for ind in inds:
            binmutrate = random.uniform(self.bin_min_mutations, self.bin_max_mutations)
            state[ind] = self.mutate_bin(state[ind], binmutrate)

        swaps = random.randint(0, self.max_swap_num)  # this could be a hyperparam function or distribution

        for i in range(swaps):
            bins = random.sample(range(self.e), 2)
            state[bins[0]], state[bins[1]] = state[bins[1]], state[bins[0]]
        '''

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
            print("iter",f)
            current_iter = f
            starttime = time.time()

            parents = [self.generate_parent() for i in range(self.population_size)]
            # print(parents)

            for g in range(self.max_generations):

                children = []
                # children = [breed([random.choice(parents) for _ in range(self.number_of_parents)]) for d in range(2 * self.population_size)]

                helpdict = dict() # this is not yet great for vertical pics, as keeps them together
                for parent in parents:
                    for i in range(len(parent)):
                        element = parent[i]
                        val = self.help_of_element(parent, i)

                        if not element in helpdict:
                            helpdict[element] = set()
                        helpdict[element].add(val)

                helpdictavg = dict()
                for element in helpdict:
                    helpdictavg[element] = sum(helpdict[element]) / len(helpdict[element])

                helpdictmax = dict()
                for element in helpdict:
                    helpdictmax[element] = max(helpdict[element])

                for d in range(2 * self.population_size):
                    children += self.breed([random.choice(parents) for _ in range(self.number_of_parents)], helpdictavg, helpdictmax)

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
        fout = open(now.strftime("%Y%m%d-%H%M%S.%f") + ".out", 'w')

        beststate = children[0]
        towrite = str(len(children[0])) + "\n"
        for i in range(len(children[0])):
            towrite += " ".join(str(e) for e in children[0][i])
        fout.write(towrite)
        fout.close()

        return 1 / strength  # change this to return the fitness after 5 seconds


if __name__ == '__main__':
    sys.stdin = open(FNAME_PREFIX)

    alg = GeneticAlgorithm(number_of_children=2,
                           number_of_splits=4,
                           number_of_parents=3,
                           mutation_chance=0.2,
                           parents_generation_rate=0.2,
                           parent_keep_rate=0.2,
                           min_mutations=1,
                           max_mutations=1,
                           population_size=20,
                           iterations=2,
                           max_generations=10,
                           bin_max_mutations=0.1,
                           bin_min_mutations=0.5,
                           max_inner_splits=5,
                           max_swap_num=5)
    print(f'Fitness strength is: {alg.search()}')

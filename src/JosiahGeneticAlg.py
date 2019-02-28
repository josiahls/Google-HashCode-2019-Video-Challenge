import string
import random
import time


class GeneticAlgorithm:
    def __init__(self, number_of_children, number_of_splits, number_of_parents, number_parents_to_keep,
                 mutation_rate, parents_generation_rate, parent_keep_rate, loss_exp,
                 min_mutations, max_mutations, max_generations, population_size,
                 max_fitness, thresh_fitness, iterations, size, target, new_parents_per_generation=None):

        self.thresh_fitness = thresh_fitness
        self.max_fitness = max_fitness
        self.max_generations = int(max_generations)
        self.loss_exp = loss_exp
        self.parent_keep_rate = parent_keep_rate
        self.parents_generation_rate = parents_generation_rate
        self.number_of_parents = int(number_of_parents)
        self.iterations = int(iterations)
        self.population_size = int(population_size)
        self.max_mutations = int(max_mutations)
        self.min_mutations = int(min_mutations)
        self.mutation_rate = mutation_rate
        self.number_of_splits = int(number_of_splits)
        self.number_of_children = int(number_of_children)

        """ Some of these parameters are better set automatically """
        _auto_size = int(self.parent_keep_rate * self.population_size)
        self.number_parents_to_keep = _auto_size if number_parents_to_keep is None else number_parents_to_keep
        _auto_size = int(self.parents_generation_rate * self.population_size)
        self.new_parents_per_generation = _auto_size if number_parents_to_keep is None else new_parents_per_generation

        """ Set the generator / dataset sampling """
        self.parent_generator = string.printable

        """ State space size definition """
        self.size = size

        """ Target """
        self.target = target

    def fitness(self, x):
        return sum(e[0] == e[1] for e in zip(self.target, x))

    def generate_parent(self):
        return "".join(random.choice(self.parent_generator) for i in range(self.size))

    def is_valid_element(self, child):
        return len(child) == self.size

    def breed(self, strs):
        P = len(strs)

        children = []

        for c in range(self.number_of_children):
            splitlocs = [0] + sorted([random.choice(range(self.size)) for i in range(self.number_of_splits - 1)]) + [
                self.size]  # doesn't guarantee unique

            splitchoices = [random.choice(range(P))]
            for i in range(1, self.number_of_splits):

                # random excluding prior choice:
                choice = random.choice(range(P - 1))
                if choice <= splitchoices[i - 1]: choice += 1

                splitchoices.append(choice)

            assert len(splitlocs) == self.number_of_splits + 1
            assert len(splitchoices) == self.number_of_splits
            # cchild = ''.join(strs[splitchoices[i]][splitlocs[i]:splitlocs[i+1]] for i in range(self.number_of_splits))
            cchild = ''
            for i in range(self.number_of_splits):
                if not isinstance(strs[splitchoices[i]], str):
                    print(strs[splitchoices[i]], strs)
                cchild += strs[splitchoices[i]][splitlocs[i]:splitlocs[i + 1]]

            # self.mutate
            if random.random() < self.mutation_rate:
                cchild = self.mutate(cchild)

            assert self.is_valid_element(cchild)

            children.append(cchild)

        return children

    def mutate(self, strc):
        nummutations = random.randint(self.min_mutations, self.max_mutations + 1)

        inds = random.sample(range(len(strc)), nummutations)
        stra = [e for e in strc]
        for ind in inds:
            stra[ind] = random.choice(self.parent_generator)  # choose randomly

        ans = ''.join(stra)
        return ans

    def select_children(self, children):
        return children[:int(self.population_size * 0.8)] + children[
                                                            int(self.population_size * 0.5): int(
                                                                self.population_size * 0.5) + self.population_size - int(
                                                                self.population_size * 0.8)]

    def search(self):
        totaltime = 0
        numconverged = 0

        print("Testing parameters...")
        for f in range(self.iterations):
            starttime = time.time()

            parents = [self.generate_parent() for i in range(100)]
            # print(parents)

            for g in range(self.max_generations):

                children = []
                # children = [self.breed([random.choice(parents) for _ in range(self.number_of_parents)]) for d in range(2 * self.population_size)]
                for d in range(2 * self.population_size):
                    children += self.breed([random.choice(parents) for _ in range(self.number_of_parents)])

                children += parents[:self.number_parents_to_keep]
                children += [self.generate_parent() for i in range(self.new_parents_per_generation)]
                children = sorted(children, key=lambda child: self.fitness(child), reverse=True)

                # we should keep the ones that are more different, but for now, just pull from top 80% and bottom 20%
                children = self.select_children(children)

                self.max_fitness = self.fitness(children[0])

                print("Gen " + str(g) + " max self.fitness " + str(self.max_fitness) + " with " + children[0])
                parents = children

                if self.fitness(children[0]) >= self.thresh_fitness:
                    numconverged += 1
                    print(
                        "Finish gen " + str(g) + " max self.fitness " + str(self.max_fitness) + " with " + children[0])
                    break

            else:
                numconverged += 0

            # print(parents)

            endtime = time.time()

            elapsedtime = endtime - starttime
            totaltime += elapsedtime

        convergedrate = numconverged / self.iterations
        avgtime = totaltime / self.iterations

        strength = avgtime * (self.iterations ** self.loss_exp)

        print(strength)
        return strength


if __name__ == '__main__':
    X = "Hello world!"

    alg = GeneticAlgorithm(number_of_children=2,
                           number_of_splits=4,
                           number_of_parents=3,
                           number_parents_to_keep=None,
                           mutation_rate=0.2,
                           parents_generation_rate=0.2,
                           parent_keep_rate=0.2,
                           min_mutations=1,
                           max_mutations=1,
                           population_size=400,
                           iterations=2,
                           loss_exp=2.0,
                           max_generations=100,
                           max_fitness=len(X),
                           thresh_fitness=len(X),
                           size=len(X),
                           target=X)
    print(f'Fitness strength is: {alg.search()}')

import string
import random
import time
random.seed(1)

#N = 1000
#M = 1000

target = "Hello world!"
N = len(target)

MAXFITNESS = len(target)
THRESHFITNESS = MAXFITNESS # for benchmarking only


geneset = string.printable

MAXGENERATIONS = 100
TESTSPERSETUP = 2 # 10
TESTLOSSEXPONENT = 2.0

def fitness(x):
    return sum(e[0] == e[1] for e in zip(target, x))

print( fitness(target) )

def genParent():
    return "".join(random.choice(geneset) for i in range(N))

def isValidElement(child):
    return len(child) == N

def breed(strs):
    P = len(strs)

    children = []

    for c in range(NUMCHILDREN):
        splitlocs = [0] + sorted([random.choice(range(N)) for i in range(NUMSPLITS-1)]) + [N] # doesn't guarantee unique

        splitchoices = [random.choice(range(P))]
        for i in range(1, NUMSPLITS):

            # random excluding prior choice:
            choice = random.choice(range(P-1))
            if choice <= splitchoices[i-1]: choice += 1

            splitchoices.append(choice)

        assert len(splitlocs) == NUMSPLITS+1
        assert len(splitchoices) == NUMSPLITS
        #cchild = ''.join(strs[splitchoices[i]][splitlocs[i]:splitlocs[i+1]] for i in range(NUMSPLITS))
        cchild = ''
        for i in range(NUMSPLITS):
            if not isinstance(strs[splitchoices[i]], str):
                print(strs[splitchoices[i]], strs)
            cchild += strs[splitchoices[i]][splitlocs[i]:splitlocs[i+1]]


        # Mutate
        if random.random() < MUTCHANCE:
            cchild = mutate(cchild)
        
        assert isValidElement(cchild)
        
        children.append(cchild)

    return children
    
def mutate(strc):
    nummutations = random.randint(MINMUTATIONS, MAXMUTATIONS+1)

    inds = random.sample(range(len(strc)), nummutations)
    stra = [e for e in strc]
    for ind in inds:
        stra[ind] = random.choice(geneset)  # choose randomly

    ans = ''.join(stra)
    return ans
    
def select_children(children):
    return children[:int(POPSIZE * 0.8)] + children[int(POPSIZE * 0.5): int(POPSIZE * 0.5) + POPSIZE-int(POPSIZE * 0.8)]

def ga(NUMSPLITS, NUMCHILDREN, NUMPARENTS, POPSIZE, MINMUTATIONS, MAXMUTATIONS, NEWPARENTSPERGENRATE, PARENTSKEPTRATE, MUTCHANCE):

    totaltime = 0
    numconverged = 0

    print("Testing parameters...")
    for f in range(TESTSPERSETUP):
        starttime = time.time()


        parents = [genParent() for i in range(100)]
        #print(parents)


        for g in range(MAXGENERATIONS):

            children = []
            #children = [breed([random.choice(parents) for _ in range(NUMPARENTS)]) for d in range(2 * POPSIZE)]
            for d in range(2 * POPSIZE):
                children += breed([random.choice(parents) for _ in range(NUMPARENTS)])

            children += parents[:PARENTSKEPT]
            children += [genParent() for i in range(NEWPARENTSPERGEN)]
            children = sorted(children, key=lambda child: fitness(child), reverse=True)

            # we should keep the ones that are more different, but for now, just pull from top 80% and bottom 20%
            children = select_children(children)

            maxfitness = fitness(children[0])

            print("Gen "+str(g)+" max fitness "+str(maxfitness)+" with "+children[0])
            parents = children

            if fitness(children[0]) >= THRESHFITNESS:
                numconverged += 1
                print("Finish gen "+str(g)+" max fitness "+str(maxfitness)+" with "+children[0])
                break

        else:
            numconverged += 0

        # print(parents)

        endtime = time.time()

        elapsedtime = endtime - starttime
        totaltime += elapsedtime

    convergedrate = numconverged / TESTSPERSETUP
    avgtime = totaltime / TESTSPERSETUP


    strength = avgtime * (TESTSPERSETUP**TESTLOSSEXPONENT)

    print(strength)
    return strength


if __name__=='__main__':


    # hyperparameters
    NUMSPLITS = 4
    NUMCHILDREN = 2
    NUMPARENTS = 3

    POPSIZE = 400
    MINMUTATIONS = 1
    MAXMUTATIONS = 1
    NEWPARENTSPERGENRATE = 0.2

    PARENTSKEPTRATE = 0.2

    NEWPARENTSPERGEN = int(NEWPARENTSPERGENRATE * POPSIZE)
    PARENTSKEPT = int(PARENTSKEPTRATE * POPSIZE)



    MUTCHANCE = 0.2

    print(ga(NUMSPLITS, NUMCHILDREN, NUMPARENTS, POPSIZE, MINMUTATIONS, MAXMUTATIONS, NEWPARENTSPERGENRATE, PARENTSKEPTRATE, MUTCHANCE))

import string
import random
import time
import math
random.seed(1)

#MAXFITNESS = len(target)
THRESHFITNESS = 10**100 #MAXFITNESS # for benchmarking only


# read input
V, E, R, C, X = list(map(int, input().strip().split()))
VSIZES = list(map(int, input().strip().split()))
assert len(VSIZES) == V
ECACHETIMES = [dict() for i in range(E)]
DCLATENCIES = []

for i in range(E):
    latency, numcaches = list(map(int, input().strip().split()))
    DCLATENCIES[i] = latency

    for j in range(numcaches):
        curcachenode, curcachelatency = list(map(int, input().strip().split()))
        ECACHETIMES[i][curcachenode] = curcachelatency

REQS = []
NUMINDREQS = 0
for i in range(R):
    reqlist = list(map(int, input().strip().split()))
    reqs.append(reqlist)
    NUMINDREQS += reqlist[2]
    

geneset_videos = range(V)

MAXGENERATIONS = 100
TESTSPERSETUP = 2 # 10
TESTLOSSEXPONENT = 2.0

BINMUTRATEMIN = 0.1 # on average, when a bin is mutated, remove 20% of the videos (regardless of capacity)
BINMUTRATEMAX = 0.5 # change this lower for large values


#MAXSWAPRATE = 0.1
MAXSWAPNUM = 5
MAXSWAPS = MAXSWAPNUM
#NUMSWAPPARTICIPANTRATE = 1.0 # 0.0 for no swaps to 1.0
#NUMSWAPPARTICIPANT = math.ceil(NUMPARENTS * NUMSWAPPARTICIPANTRATE)
# implemented this in mutation instead for now


def fitness(state):
    mssaved = 0
    for req in REQS:
        vid, endpt, nreqs = req

        dcreqtime = DCLATENCIES[endpt]
        minreqtime = dcreqtime
        for ccacheid in ECACHETIMES[endpt]:
            if state[ccacheid].contains(vid):
                minreqtime = ECACHETIMES[endpt][ccacheid]
                
        mssaved = (dcreqtime - minreqtime) * nreqs

    return mssaved / NUMINDREQS * 1000


#print( fitness(target) )

def genParent():

    parent = []

    for i in range(V):
        csize = 0
        cset = set()
        while 1:
            video = random.choice(geneset_videos)

            if csize + newsize <= X:
                cset.add(video)
            else:
                break # don't try to pack it at this stage
            
        parent.append(cset)

    return parent


def isValidElement(child):
    return sum(VSIZES[e] for e in binset) <= X

def breed(states):

    P = len(states)

    children = []

    for c in range(NUMCHILDREN):

        # Crossover
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

        # Swaps
        '''
        swaps = random.randint(0, MAXSWAPS)

        for i in range(MAXSWAPS):
            participants = random.sample(NUMSWAPPARTICIPANTRATE)
        '''
            
        
        # Mutate
        if random.random() < MUTCHANCE:
            cchild = mutate(cchild)
        
        assert isValidElement(cchild)
        
        children.append(cchild)

        

    return children


def mutate_bin(binset):
    # return a mutated bin; helper for mutate()
    numtoremove = math.floor(binmutrate * len(binset))
    toremove = random.sample(binset, numtoremove)
    for elremove in binset:
        binset.remove(elremove)

    csize = sum(VSIZES[e] for e in binset)
    while 1:
        video = random.choice(geneset_videos)

        if csize + newsize <= X:
            cset.add(video)
        else:
            break # don't try to pack it at this stage
    
    return binset
    

def mutate(state):
    nummutations = random.randint(MINMUTATIONS, MAXMUTATIONS+1)

    inds = random.sample(range(len(strc)), nummutations)

    for ind in inds:
        binmutrate = random.random(BINMUTRATEMIN, BINMUTRATEMAX)
        state[ind] = mutate_bin(state[ind], binmutrate)

    swaps = random.randint(0, MAXSWAPS) # this could be a hyperparam function or distribution

    for i in range(swaps):
        bins = random.sample(range(E),2)
        state[bins[0]], state[bins[1]] = state[bins[1]], state[bins[0]]

    return state # to be consistent for other problems, we return reference to the state
    

def lateral_transfer(parents):
    return parents

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
    return 1/strength  #change this to return the fitness after 5 seconds

    now = datetime.now()
    fout = open("ans_"+FNAME_PREFIX+"+now.strftime("%Y%m%d-%H%M%S.%f")+".out")


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


    
                                         


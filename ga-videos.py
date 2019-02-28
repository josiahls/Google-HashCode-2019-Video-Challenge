import string
import random
import time
import math
from datetime import datetime
random.seed(1)

#MAXFITNESS = len(target)
THRESHFITNESS = 10**100 #MAXFITNESS # for benchmarking only

DOPRINTGEN = 1 # show progress by generation with winning one

FNAME_PREFIX = 'a'

# read input
V, E, R, C, X = list(map(int, input().strip().split()))
VSIZES = list(map(int, input().strip().split())) # they start at 0 so list with 0 index is ok
assert len(VSIZES) == V
ECACHETIMES = [dict() for i in range(E)]
DCLATENCIES = [None for i in range(E)]

for i in range(E):
    latency, numcaches = list(map(int, input().strip().split()))
    DCLATENCIES[i] = latency

    for j in range(numcaches):
        curcachenode, curcachelatency = list(map(int, input().strip().split()))
        ECACHETIMES[i][curcachenode] = curcachelatency
#print(ECACHETIMES)
REQS = []
NUMINDREQS = 0
for i in range(R):
    reqlist = list(map(int, input().strip().split()))
    REQS.append(reqlist)
    NUMINDREQS += reqlist[2]
#print(REQS)
#print(NUMINDREQS)

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


def fitness(state): # be sure that higher is better
    mssaved = 0
    for req in REQS:
        vid, endpt, nreqs = req
        #print(vid,endpt,nreqs)

        dcreqtime = DCLATENCIES[endpt]
        minreqtime = dcreqtime
        for ccacheid in ECACHETIMES[endpt]:
            if vid in state[ccacheid]:
                minreqtime = min(ECACHETIMES[endpt][ccacheid], minreqtime)
                #minreqsvr = state[ccacheid]
                
        mssaved += (dcreqtime - minreqtime) * nreqs
        #print("at endpt", endpt, "saved", mssaved, "with", (dcreqtime - minreqtime), "off", nreqs, "reqs using", minreqsvr)

    return mssaved * 1000 / NUMINDREQS
'''
print(fitness([set(),set(),set()]))
print(fitness([set(),{2},{1}]))
print(fitness([{2},{3,1},{0,1}]))
print(fitness([{1},{2},{}]))
'''

#print( fitness(target) )

def genParent():

    parent = []

    for i in range(C):
        csize = 0
        cset = set()
        while 1:
            video = random.choice(geneset_videos)
            newsize = VSIZES[video]

            if csize + newsize <= X:
                cset.add(video)
                csize += newsize
            else:
                break # don't try to pack it at this stage
            
        parent.append(cset)

    assert isValidElement(parent)
    return parent


def isValidElement(state):
    # print (X, state, list(sum(VSIZES[e] for e in binset) for binset in state))
    return len(state) == C and all(sum(VSIZES[e] for e in binset) <= X for binset in state)

def breed(states):

    P = len(states)
    assert P == NUMPARENTS

    children = []

    N = len(states[0]) # or any other in the array
    assert N == C

    for c in range(NUMCHILDREN):

        # Crossover
        splitlocs = [0] + sorted([random.choice(range(N)) for i in range(NUMSPLITS-1)]) + [N] # doesn't guarantee unique

        splitchoices = [random.choice(range(P))]
        for i in range(1, NUMSPLITS):
            # random excluding prior choice:
            choice = random.choice(range(P-1))
            if choice <= splitchoices[i-1]: choice += 1
            splitchoices.append(choice)
        
        # print("making child: ", states, P, splitlocs, splitchoices)
        

        assert len(splitlocs) == NUMSPLITS+1
        assert len(splitchoices) == NUMSPLITS
        #cchild = ''.join(strs[splitchoices[i]][splitlocs[i]:splitlocs[i+1]] for i in range(NUMSPLITS))
        cchild = []
        for i in range(NUMSPLITS):
            if not isinstance(states[splitchoices[i]], list) and (isinstance(states[splitchoices[i]][j], set) for j in range(len(states[splitchoices[i]]))):
                assert 0
                # print(strs[splitchoices[i]], strs)
            cchild += states[splitchoices[i]][splitlocs[i]:splitlocs[i+1]]

        # Swaps
        '''
        swaps = random.randint(0, MAXSWAPS)

        for i in range(MAXSWAPS):
            participants = random.sample(NUMSWAPPARTICIPANTRATE)
        '''
            
        
        # Mutate
        if random.random() < MUTCHANCE:
            cchild = mutate(cchild)

        # print(cchild)
        assert isValidElement(cchild)
        
        children.append(cchild)

        

    return children


def mutate_bin(binset, binmutrate):
    # return a mutated bin; helper for mutate()
    numtoremove = math.floor(binmutrate * len(binset))
    toremove = random.sample(binset, numtoremove)
    for elremove in toremove: # use the right variable
        binset.remove(elremove)

    csize = sum(VSIZES[e] for e in binset)
    while 1:
        video = random.choice(geneset_videos)
        newsize = VSIZES[video]

        if csize + newsize <= X:
            binset.add(video)
            csize += newsize
        else:
            break # don't try to pack it at this stage
    
    return binset
    

def mutate(state):
    nummutations = random.randint(MINMUTATIONS, MAXMUTATIONS+1)

    inds = random.sample(range(len(state)), nummutations)

    for ind in inds:
        binmutrate = random.uniform(BINMUTRATEMIN, BINMUTRATEMAX)
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

            # we should keep the ones that are more different and still ok, but for now, just pull from top 80% and bottom 20%
            children = select_children(children)

            maxfitness = fitness(children[0])
            
            print("Gen "+str(g)+" max fitness "+str(maxfitness)+" with "+str(children[0]))
            parents = children

            if fitness(children[0]) >= THRESHFITNESS:
                numconverged += 1
                if DOPRINTGEN:
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

    now = datetime.now()
    fout = open("ans_"+FNAME_PREFIX+now.strftime("%Y%m%d-%H%M%S.%f")+".out", 'w')

    beststate = children[0]
    towrite = str(C)+"\n"
    for i in range(C):
        towrite += str(i)+" "+" ".join(str(e) for e in list(beststate[i]))
    fout.write(towrite)
    fout.close()

    return 1/strength  #change this to return the fitness after 5 seconds
                


if __name__=='__main__':


    # hyperparameters
    NUMSPLITS = 2
    NUMCHILDREN = 4
    NUMPARENTS = 2

    POPSIZE = 15
    MINMUTATIONS = 1
    MAXMUTATIONS = 1
    NEWPARENTSPERGENRATE = 0.2

    PARENTSKEPTRATE = 0.2

    NEWPARENTSPERGEN = int(NEWPARENTSPERGENRATE * POPSIZE)
    PARENTSKEPT = int(PARENTSKEPTRATE * POPSIZE)

    

    MUTCHANCE = 0.2

    print(ga(NUMSPLITS, NUMCHILDREN, NUMPARENTS, POPSIZE, MINMUTATIONS, MAXMUTATIONS, NEWPARENTSPERGENRATE, PARENTSKEPTRATE, MUTCHANCE))


    
                                         


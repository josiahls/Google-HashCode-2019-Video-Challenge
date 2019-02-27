"""
State:



"""

class Optimizer:
    def __init__(self, params: dict):
        self.params = params

    def optimize(self, function):
        """
        We want to check the fitness after a fixed amount of time

        :param function:
        :return:
        """
        params = {'NUMSPLITS', 'NUMCHILDREN', 'NUMPARENTS', 'POPSIZE', 'MINMUTATIONS', 'MAXMUTATIONS',
                  'NEWPARENTSPERGENRATE', 'PARENTSKEPTRATE', 'MUTCHANCE'}
        pass

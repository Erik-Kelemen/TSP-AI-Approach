import sys
from collections import deque

import tsp
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming

from utils import memoize, PriorityQueue, is_in, Graph, argmax_random_tie, probability, weighted_sampler
import time
import random
import glob


class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)

def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                # f1=open("output.txt", "a")
                print(str(len(explored)) + " NODES EXPANDED", end="")
                # f1.close()
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None

def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)

 
def uniform_cost_search(problem, display=False):
    """[Figure 3.14]"""
    return best_first_graph_search(problem, lambda node: node.path_cost, display)   

# ______________________________________________________________________________

class TravelingSalesmanProblem(Problem):
    """The problem of finding a minimum cost Hamiltonian cycle
    from a random start city with minimum cost. The graph is represented 
    by an NxN adjacency matrix that is symmetrical about its diagonal (undirected).
    A state is represented as a set of visited cities, and the most recently visited city.
    The set of possible actions is to pick and add an unvisited city to the set of cities,
    and make this new city the most recently visited city."""

    "state: (set visited_cities, int curr_city)"
    def __init__(self, n, adj_mat):
        self.cities = frozenset(range(0,n)) #{0, 1, 2, 3}
        self.graph = adj_mat
        state = (frozenset({0}), 0)
        super().__init__(state)
        self.n = n

    def actions(self, state):
        #"""If we have visited all cities and curr city is start city again"""
        if state[0] == self.cities and state[1] == 0:
            return []  
        elif state[0] == self.cities and state[1] != 0:   # If we have visited all cities but need to return to our start city
            return [0]
        else:
            actions = []
            #Get difference between two sets, yields UNVISITED cities
            unvisited = self.cities - state[0]
            for city in unvisited:
                actions.append(city)
            return actions

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        return (state[0].union({action}), action)

    def goal_test(self, state):
        """Check if state[0] == state[n], and (unvisited set is null OR visited set is size N)."""
        if state[0] != self.cities or state[1] != 0:
            return False
        return True

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        # cost = c + self.graph[state1[1]][action]
        # print(cost)
        return c + self.graph[state1[1]][action]

    def random_heuristic(self, node):
        successors = self.actions(node.state)
        if not successors:
            return 0
        rand_edge = self.graph[node.state[1]][random.choice(successors)]
        return rand_edge

    def cheapest_remaining_heuristic(self, node):
        successors = self.actions(node.state)
        if not successors:
            return 0
        lowest = self.graph[node.state[1]][successors[0]]
        for elem in successors[1:]:
            if self.graph[node.state[1]][elem] < lowest:
                lowest = self.graph[node.state[1]][elem]
        return lowest

    def MST_heuristic(self, node):
        t_unvisited = self.cities - node.state[0] 
        unvisited = t_unvisited.union(frozenset({node.state[1]}))
        n = len(unvisited)
        G = Graph(n) 
        adj_mat = np.zeros((n,n), dtype=int)
        r = 0
        for i in range(self.n):
            c = 0
            for j in range(self.n):
                if i in unvisited and j in unvisited:
                    adj_mat[r][c] = self.graph[i][j]
                    c+=1
            if i in unvisited:
                r += 1
        G.graph = adj_mat
        MST = G.primMST()
        sum = 0
        for i in range(1, n):
            sum += adj_mat[i][ MST[i] ]
        return sum
    
# ______________________________________________________________________________

def hill_climbing(problem):
    """
    [Figure 4.2]
    From the initial node, keep choosing the neighbor with highest value,
    stopping when no neighbor is better.
    """
    current = Node(problem.initial)
    while True:
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = argmax_random_tie(neighbors, key=lambda node: problem.value(node.state))
        if problem.value(neighbor.state) <= problem.value(current.state):
            break
        current = neighbor
    return current.state


def exp_schedule(k=20, lam=0.005, limit=100):
    """One possible schedule function for simulated annealing"""
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)


def simulated_annealing(problem, schedule=exp_schedule()):
    """[Figure 4.5] CAUTION: This differs from the pseudocode as it
    returns a state instead of a Node."""
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return current.state
        neighbors = current.expand(problem)
        if not neighbors:
            return current.state
        next_choice = random.choice(neighbors)
        delta_e = problem.value(next_choice.state) - problem.value(current.state)
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            current = next_choice

# ______________________________________________________________________________

def genetic_search(problem, ngen=10000, pmut=0.1, n=20):
    """Call genetic_algorithm on the appropriate parts of a problem.
    This requires the problem to have states that can mate and mutate,
    plus a value method that scores states."""

    # NOTE: This is not tested and might not work.
    # TODO: Use this function to make Problems work with genetic_algorithm.
    s = problem.initial_state
    states = [problem.result(s, a) for a in problem.actions(s)]
    random.shuffle(states)
    return genetic_algorithm(states[:n], problem.valueGA, range(n), None, ngen, pmut)


def genetic_algorithm(population, fitness_fn, gene_pool=[0, 1], f_thres=None, ngen=1000, pmut=0.1):
    """[Figure 4.8]"""
    for i in range(ngen):
        population = [mutate(recombine(*select(2, population, fitness_fn)), gene_pool, pmut)
                      for i in range(len(population))]

        fittest_individual = fitness_threshold(fitness_fn, f_thres, population)
        if fittest_individual:
            return fittest_individual

    return max(population, key=fitness_fn)


def fitness_threshold(fitness_fn, f_thres, population):
    if not f_thres:
        return None

    fittest_individual = max(population, key=fitness_fn)
    if fitness_fn(fittest_individual) >= f_thres:
        return fittest_individual

    return None


def init_population(pop_number, gene_pool, state_length):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    state_length:  The length of each individual"""
    g = len(gene_pool)
    population = []
    for i in range(pop_number):
        new_individual = [gene_pool[random.randrange(0, g)] for j in range(state_length)]
        population.append(new_individual)
    return population


def select(r, population, fitness_fn):
    fitnesses = map(fitness_fn, population)
    sampler = weighted_sampler(population, fitnesses)
    return [sampler() for i in range(r)]

# (1,2,3,0,4,5,6,7,8,9) || (0,1,5,2,4,3,6,7,8,9)
# (1,2) + (2,4) + (4,5) + (5,2) + (2, 3) + (3,6) + (6,7) + (7,8) + (8,9)
# (1,2,4,5,2,3,6,7,8,9)

def recombine(x, y):
    n=len(x)
    edge = random.randrange(0, n-1)

    new_tup = (x[edge], x[edge+1])
    from_edge = edge+1
    alt = 0
    ins = (0,)
    unvisited = set(range(n))
    unvisited.remove(x[edge])
    unvisited.remove(x[edge+1])
    while len(new_tup) < n:
        if alt % 2 == 0:
            from_edge = y.index(from_edge) + 1
            if from_edge < n and y[from_edge] not in new_tup:
                ins = (y[from_edge],)
            else:
                ins = (random.choice(tuple(unvisited)),)
        else:
            from_edge = x.index(from_edge) + 1
            if from_edge < n and x[from_edge] not in new_tup:
                ins = (x[from_edge],)
            else:
                ins = (random.choice(tuple(unvisited)),)
        new_tup = new_tup + ins
        alt += 1
        from_edge = ins[0]
        unvisited.remove(ins[0])
    return new_tup


def mutate(x, gene_pool, pmut):
    if random.uniform(0, 1) >= pmut:
        return x
    #print(gene_pool)

    n = len(x)
    c = random.randrange(0, n)
    r = random.randrange(0, n - 1)
    new_gene = x[c]

    new_tup = x[:c] + x[(c + 1):]
    return new_tup[:r+1] + (new_gene,) + new_tup[r + 1:]
    

# ______________________________________________________________________________

class TSPLocalSearch(Problem):
    """The problem of finding a minimum cost Hamiltonian cycle
    from a random start city with minimum cost. The graph is represented 
    by an NxN adjacency matrix that is symmetrical about its diagonal (undirected).
    A state is represented as a set of visited cities, and the most recently visited city.
    The set of possible actions is to pick and add an unvisited city to the set of cities,
    and make this new city the most recently visited city."""

    "state: tuple of ordered cities (0,1,2,3,...,n-1)"
    def __init__(self, n, adj_mat):
        start = tuple(range(n))
        super().__init__(start)
        self.initial_state = start
        self.n = n
        self.graph = adj_mat

    def actions(self, state):
        actions = []
        for elem in range(self.n-1):
            new_tup = state
            sto = new_tup[elem]
            new_tup = new_tup[:elem] + (new_tup[elem+1],) + (sto,) + new_tup[(elem+2):]
            actions.append(new_tup)
        return actions

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        return action
    #Will need to change following for HC and SA
    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        prev = state[0]
        total = 0
        for var in state[1:]:
            total += self.graph[prev][var]
            prev = var
        total *= -1
        return total
    def valueGA(self, state):
        prev = state[0]
        total = 0
        for var in state[1:]:
            total += self.graph[prev][var]
            prev = var
        #total *= -1
        return 7000 - total

def test_heuristic(n, adj_mat, h, prob, fi):
    result = astar_search(prob, h, display=False)
    
    print(result)
    fn = memoize(result, 'f')

    out = fn(result)
    print(str(out) + ",", end="", file=fi)
    path = result.path()
    # print(path)
    # s_path = ""

    # for elem in path:
    #     s_path += str(elem.state[1]) + " "
    # print("Path: ", end='')
    # print(s_path)

def main():
    f = open("output.txt", "a")
    
    lines = sys.stdin.readlines()
    start_time = time.time()

    #lines = ['10\n', '0 332 41 682 471 695 407 635 595 533\n', '332 0 293 451 360 457 407 493 353 239\n', '41 293 0 643 457 655 405 621 555 492\n', '682 451 643 0 791 19 857 890 98 233\n', '471 360 457 791 0 793 138 164 695 559\n', '695 457 655 19 793 0 861 888 105 233\n', '407 407 405 857 138 861 0 267 758 633\n', '635 493 621 890 164 888 267 0 798 657\n', '595 353 555 98 695 105 758 798 0 142\n', '533 239 492 233 559 233 633 657 142 0\n']
    #lines = ['4\n', '0 20 42 35\n', '20 0 30 34\n', '42 30 0 12\n', '35 34 12 0']
    
    # print(lines)
    n, adj_mat = read_format(lines)
    
    permutation, dist = independent_tsp_solver(adj_mat)
    #print(permutation)
    #print(dist)

    prob = TravelingSalesmanProblem(n, adj_mat)
    # print("Correct solution:")
    # permutation, dist = independent_tsp_solver(adj_mat)
    
    cpu_time = time.process_time()
    # print("UCS Test:")
    # test_heuristic(n, adj_mat, lambda u: 0, prob)

    # print("Random Edge Test:")
    # test_heuristic(n, adj_mat, prob.random_heuristic, prob)
    
    #print("Cheapest Edge Test:")
    test_heuristic(n, adj_mat, prob.cheapest_remaining_heuristic, prob, f)

    #print("MST Test:")
    test_heuristic(n, adj_mat, prob.MST_heuristic, prob, f)
    

    #print(ucs_result)
    #fn = memoize(ucs_result, 'f')
    #prob = TSPLocalSearch(n, adj_mat)
    # hc = hill_climbing(prob)

    # sa = simulated_annealing(prob)
    # tot_cost = 0
    # prev = hc[0]
    
    # #print(prev)
    # for city in hc[1:]:
    #     tot_cost += adj_mat[prev][city]
    #     prev = city
    # tot_cost += adj_mat[prev][0]

    # print("HILL_CLIMBING TOTAL COST OF PATH: " + str(tot_cost))

    # tot_cost = 0
    # prev = sa[0]
    
    # #print(prev)
    # for city in sa[1:]:
    #     tot_cost += adj_mat[prev][city]
    #     prev = city
    # tot_cost += adj_mat[prev][0]

    # print("SA TOTAL COST OF PATH: " + str(tot_cost))
    
    # ret = genetic_search(prob, 500*n, 0.1, n)
    
    # #print("GA with 1000 generations: " + str(ret))
    # ret = genetic_search(prob, 500*n, 0.1, n)
    # prev = ret[0]
    # total = 0
    # for var in ret[1:]:
    #     total += adj_mat[prev][var]
    #     prev = var
    # total += adj_mat[prev][0]
    #print("Distance: " + str(total))

    #"""FILE FORMAT:
    #(NODES_EXPANDED),(COST OF PATH),(COST_OPTIMAL),(CPU_TIME),(WALL_TIME)"""

    # print("%s," % (total), end="", file=f)
    # print("%s," % (dist), end="", file=f)
    print("%s," % (time.process_time() - cpu_time), end="", file=f)
    print("%s," % (time.time() - start_time), end="", file=f)
    print("", file=f)


    #(ret)
    
    f.close()

#File Reader:
def read_format(lines: list):
    n = int(lines[0])
    adj_m = [[0]*n for i in range(n)]
    c = 0
    for line in lines[1:]:
        j = 0
        for word in line.split(" "):
            adj_m[c][j] = int(word)
            j+=1 
        c += 1
    return (n, adj_m)
#Independent travelling salesman problem solver I found on the internet
#From: https://pypi.org/project/python-tsp/
def independent_tsp_solver(adj_mat):
    dist_mat = np.array(adj_mat)
    permutation, dist = solve_tsp_dynamic_programming(dist_mat)
    print("Dist: " + str(dist))
    print("Perm: " + str(permutation))
    return permutation, dist


if __name__ == "__main__":
    main()
    
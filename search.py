# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """

    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    """

    # # OUTPUT: Start State: (5, 5)
    # print("Start State:", problem.getStartState())

    # # OUTPUT: Is the start a goal? False
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))

    # # OUTPUT: Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))


    # DFS uses a stack; LIFO
    frontier_stack = util.Stack()
    explored_set = set()

    # Initialize by adding start state to Frontier Stack
    frontier_stack.push((problem.getStartState(), []))

    # Keep popping from stack till empty
    while not frontier_stack.isEmpty():
        # Pop the first element (last in on stack)
        temp_tuple = frontier_stack.pop()
        current_state = temp_tuple[0]
        actions = temp_tuple[1]

        # Check if we've already visited this state
        if current_state in explored_set:
            continue
        # Mark explored state
        explored_set.add(current_state)

        # Check if the state which was POPPED is the goal state;
        if problem.isGoalState(current_state):
            return actions

        # Else, explore successive states and add to stack
        for next_state, next_direction, _ in problem.getSuccessors(current_state):
            if next_state not in explored_set:
                frontier_stack.push((next_state, actions + [next_direction]))

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""

    # Command: python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs | score: 442.0
    # Command: python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5

    frontier_queue = util.Queue()
    explored_set = set()

    # Initialize by adding start state to Frontier Stack
    frontier_queue.push((problem.getStartState(), []))

    while not frontier_queue.isEmpty():
        # Pop the first element (first in queue)
        temp_tuple = frontier_queue.pop()
        current_state = temp_tuple[0]
        actions = temp_tuple[1]

        # Check if we've already visited this state
        if current_state in explored_set:
            continue
        # Mark explored state
        explored_set.add(current_state)

        # Check if the state which was POPPED is the goal state;
        if problem.isGoalState(current_state):
            return actions
        
        # Enque successive states level by level
        for next_state, next_direction, _ in problem.getSuccessors(current_state):
            if next_state not in explored_set:
                frontier_queue.push((next_state, actions + [next_direction]))

    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    
    frontier_pq = util.PriorityQueue()
    best_cost = {}

    start = problem.getStartState()
    frontier_pq.push((start, [], 0), 0)  # (state, actions, cost_so_far), priority=g(n)

    while not frontier_pq.isEmpty():
        current_state, actions, cost_so_far = frontier_pq.pop()

        # If we've already found a cheaper way to this state, skip this one
        if current_state in best_cost and cost_so_far > best_cost[current_state]:
            continue
        best_cost[current_state] = cost_so_far

        # Check for goal state only when we pop from frontier
        if problem.isGoalState(current_state):
            return actions

        for next_state, next_action, stepCost in problem.getSuccessors(current_state):
            new_cost = cost_so_far + stepCost

            # Push if better than any we've seen for next_state
            if next_state not in best_cost or new_cost < best_cost[next_state]:
                frontier_pq.push(
                    (next_state, actions + [next_action], new_cost),
                    new_cost
                )

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """

    # This keeps h(n) == 0
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
     # Order priority queue by lowest cost; explore these nodes first

    #  PQ order contingent on both g(n) and h(n)
    frontier_pq = util.PriorityQueue()

    # track the running lowest cost to get to each state
    best_cost = {}
    start = problem.getStartState()

    # call the heuristic function
    frontier_pq.push((start, [], 0), heuristic(start, problem))

    while not frontier_pq.isEmpty():
        # first pop the node w/ the smallest overall cost
        current_state, actions, cost_so_far = frontier_pq.pop()
        if current_state in best_cost and cost_so_far > best_cost[current_state]:
            continue

        # iterative update of best cost per state
        best_cost[current_state] = cost_so_far

        if problem.isGoalState(current_state):
            return actions

        for next_state, next_direction, stepCost in problem.getSuccessors(current_state):
            # compute the new g(n)
            new_cost = cost_so_far + stepCost

            # f(n) = g(n) + h(n)
            priority = new_cost + heuristic(next_state, problem)

            # only considered if it improves current best known cost
            if next_state not in best_cost or new_cost < best_cost[next_state]:
                frontier_pq.push(
                    (next_state, actions + [next_direction], new_cost),
                    priority
                )
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

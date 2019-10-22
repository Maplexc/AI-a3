# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp
import util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        "*** YOUR CODE HERE ***"

        for i in range(iterations):
            V = self.values.copy()
            for state in mdp.getStates():
                action_values = util.Counter()
                for action in mdp.getPossibleActions(state):
                    print("init")
                    for next_state, prob in mdp.getTransitionStatesAndProbs(state, action):
                        action_values[action] += (mdp.getReward(
                            state, action, next_state) + prob * discount * V[next_state])
                self.values[state] = action_values[action_values.argMax()]

        # working version
        # for i in range(iterations):
        #     V = self.values.copy()
        #     print("V: ", V)
        #     for state in mdp.getStates():
        #         action_values = util.Counter()
        #         print("action_values (init): ", action_values)
        #         for action in mdp.getPossibleActions(state):
        #             for transition_state, prob in mdp.getTransitionStatesAndProbs(state, action):
        #                 print("V (init): ", V)
        #                 action_values[action] += (mdp.getReward(
        #                     state, action, transition_state) + prob * discount * V[transition_state])
        #                 print("reward (init): ", mdp.getReward(
        #                     state, action, transition_state), state, action, transition_state)
        #                 print(self.values[state])
        #                 print("action_values (init): ", action_values)
        #         print("values (init): ", self.values)
        #         self.values[state] = action_values[action_values.argMax()]

        # self.states = mdp.getStates()
        # print("all states", self.states)
        # for state in self.states:
        #     print("current state", state)
        #     print("terminal or not? ", mdp.isTerminal(state))
        #     print(len(mdp.getPossibleActions(state)))
        #     for action in mdp.getPossibleActions(state):
        #         print("action", action)
        #         print("transition prob",
        #               mdp.getTransitionStatesAndProbs(state, action))
        #         for next_state_with_prob in mdp.getTransitionStatesAndProbs(state, action):
        #             next_state = next_state_with_prob[0]
        #             print("reward", mdp.getReward(state, action, next_state))
        # self.qValues = dict()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getQValue(self, state, action):
        """
          The q-value of the state action pair
          (after the indicated number of value iteration
          passes).  Note that value iteration does not
          necessarily create this quantity and you may have
          to derive it on the fly.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        Q_value = self.values[state]
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            Q_value += self.discount * prob * self.values[next_state]
            print(Q_value)
        return Q_value

    def getPolicy(self, state):
        """
          The policy is the best action in the given state
          according to the values computed by value iteration.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # the case at the terminal state, return None
        if self.mdp.isTerminal(state):
            return None
        # if the state only have one action, return that action as policy
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 1:
            return actions[0]
        else:
            action_values = util.Counter()
            for action in actions:
                print(action)
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    action_values[action] += (self.mdp.getReward(
                        state, action, next_state) + prob * self.discount * self.values[next_state])
            return action_values.argMax()

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)

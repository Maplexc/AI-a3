# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random
import util
import math


class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """

  def __init__(self, **args):
    "You can initialize Q-values here..."
    
    ReinforcementAgent.__init__(self, **args)
    "*** YOUR CODE HERE ***"
    self.q_values = util.Counter()
    # q_values[(state, action)] = q_value
    

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # print("here")
    if (state,action) in self.q_values:
      # legal_actions = self.getLegalActions(state)
      # print("legal_actions", legal_actions)
      # if action in self.q_values[state]:
      return self.q_values[(state,action)]
    else:
      return 0.0
    # if (state, action) in self.q_values:
    #   # legal_actions = self.getLegalActions(state)
    #   return self.q_values[(state, action)]
    # else:
    #   return 0.0

  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    legal_actions = self.getLegalActions(state)
    if len(legal_actions) == 0:
      return 0.0
    else:
      # print("legal_actions", legal_actions)
      action_values = []
      for legal_action in legal_actions:
        action_values.append(self.q_values[(state, legal_action)])
      return max(action_values)

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    legal_actions=self.getLegalActions(state)
    # print("legal!!!", legal_actions)
    if len(legal_actions) == 0:
      return None
    else:
      best_value=self.getValue(state)
      best_actions=[]
      action_values = util.Counter()
      for legal_action in legal_actions:
        q_value = self.getQValue(state, legal_action)
        action_values[legal_action] = q_value
        if q_value == best_value:
          best_actions.append(legal_action)
          # print("best action: ", best_actions)
      if len(best_actions) != 0:
        return random.choice(best_actions)
      else:
        return action_values.argMax()
      # best = random.choice(best_actions)
      # print("=================== ", best)
      # return best
      
      # action_values = util.Counter()
      # for legal_action in legal_actions:
      #   action_values[legal_action] = self.getQValue(state, legal_action)
      # return action_values.argMax()


  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions=self.getLegalActions(state)
    action=None
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # return self.getPolicy()
    if len(legalActions) == 0:
      return action
    if (util.flipCoin(self.epsilon)):
      action = random.choice(legalActions)
    else:
      action = self.getPolicy(state)
    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # q_value = self.q_values[state][action]
    # q_next_value = self.q_values[nextState][action]
    # q_update = q_value + self.alpha*(reward + self.gamma*q_next_value - q_value)
    # q_update=(1-self.alpha)*self.q_values[(state, action)] + \
    #           self.alpha*(reward + self.gamma*self.getValue(nextState))
    q_value = self.q_values[(state, action)]
    q_update = q_value + self.alpha*(reward+self.gamma*self.getValue(nextState)-q_value)
    self.q_values[(state, action)]=q_update

    # self.q_table[(state, action)] = (1 - self.alpha)*self.q_table[(state, action)] + \
                                    # self.alpha*( reward + self.gamma*self.getValue(nextState))

    # actionMax = self.getPolicy(state)
    # q_approx = self.getQValue(state, action)
    # target = reward + self.gamma*self.getQValue(nextState, actionMax)
    # # loss = abs(q_approx - target)
    # q_update = (1 - self.alpha)*q_approx + self.alpha* target
    # print(q_update)
    # self.q_values[(state, action)] = (1 - self.alpha)*q_approx + self.alpha* target


class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon']=epsilon
    args['gamma']=gamma
    args['alpha']=alpha
    args['numTraining']=numTraining
    self.index=0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action=QLearningAgent.getAction(self, state)
    self.doAction(state, action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor=util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.weights = util.Counter()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # print("weight: ", self.weights)
    features = self.featExtractor.getFeatures(state, action)
    # print("features: ======", features)
    # q_value = 0
    q_value = self.weights * features
    # print("Q_value!!!!!!!!!!!", q_value)
    # for feature in features:
    #   q_value += self.weights[feature] * features[feature]
    # print("Q_value!!!!!!!!!!!", q_value)
    return q_value



  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    correction = reward + self.gamma * self.getValue(nextState) - self.getQValue(state, action)
    features = self.featExtractor.getFeatures(state, action)
    for feature in features:
      # print(self.weights[feature])
      # print(self.weights)
      # print(correction)
      # print(feature)
      # self.weights[feature] += 0.0
      self.weights[feature] += self.alpha * correction * features[feature]
    

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass

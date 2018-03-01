# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        actual_score = successorGameState.getScore()
        
        #Distance to the nearest ghost
        nearest_ghost_pos=newGhostStates[0].getPosition()
        nearest_ghost = manhattanDistance(newPos, nearest_ghost_pos)
        
        scared_time=newScaredTimes[0]
        
        #Increase score when ghost scared
        if nearest_ghost > 0:
            if scared_time>0:
                actual_score = actual_score+(20/nearest_ghost)
            else:
                actual_score = actual_score-(10/nearest_ghost)
        
        #increase score when food closer        
        newFood_positions=newFood.asList()
        if len(newFood_positions)!=0: 
            nearest_food = manhattanDistance(newPos, newFood_positions[0])
            for food in newFood_positions:
                temp_food = manhattanDistance(newPos, food)
                if temp_food<nearest_food:
                    nearest_food = temp_food
            actual_score = actual_score + (10/nearest_food)

        return actual_score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def checkTerminal(state,agent,depth):
            return state.isWin() or state.isLose() or (state.getLegalActions(agent)==0) or (depth==self.depth)
        
        def minimax(stateValues):
            
            state=stateValues[0]
            agent=stateValues[1]
            depth=stateValues[2] 
            
             #call pacman again and increase depth
            if agent==state.getNumAgents():
                depth=depth+1
                return minimax((state,0,depth))
            
            if checkTerminal(state,agent,depth):
                return self.evaluationFunction(state)
            
            children_nodes = []
            legalActions = state.getLegalActions(agent)
            for action in legalActions:
                temp=minimax((state.generateSuccessor(agent, action),agent+1,depth))
                children_nodes.append(temp)

            #if ghost, get minimum
            #else get maximum
            if agent % state.getNumAgents()!=0:
                return min(children_nodes)
            
            return max(children_nodes)
        
        legalActions = gameState.getLegalActions(0) 
        max_utility,max_index = float('-inf'),0
    	
        for index,action in enumerate(legalActions): 
            temp_utility = minimax((gameState.generateSuccessor(0, action), 1, 0))
            if temp_utility>max_utility:
                max_utility = temp_utility
                max_index = index
    	
        return legalActions[max_index]
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def checkTerminal(state,agent,depth):
            return state.isWin() or state.isLose() or (state.getLegalActions(agent)==0) or (depth==self.depth)
            
        def alphaBetaPruning(stateValues,alpha,beta):
            
            state=stateValues[0]
            agent=stateValues[1]
            depth=stateValues[2] 
            best_action=None
            
            if agent == state.getNumAgents():
                agent = 0
                depth = depth + 1 
            
            if checkTerminal(state,agent,depth):
                return [self.evaluationFunction(state),best_action]
            else:
                if agent % state.getNumAgents()==0:
                    return maxValue((state,agent,depth), alpha, beta)
                else:
                    return minValue((state,agent,depth), alpha, beta)

        def minValue(stateValues,alpha,beta):
            
            v=float('inf')
            state=stateValues[0]
            agent=stateValues[1]
            depth=stateValues[2] 
            best_action= None

            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                value,_ = alphaBetaPruning((successor, agent + 1,depth), alpha, beta)
                v,best_action = min((v, best_action), (value, action))
                
                #Different cases for pacman and ghosts
                if agent % state.getNumAgents()==0:
                    if v > beta:
                        return v,best_action
                    alpha = min(alpha,v)
                else:
                    if v < alpha:
                        return v,best_action
                    beta = min(beta,v)

            return [v,best_action]

        def maxValue(stateValues,alpha,beta):
            
            v=float('-inf')
            state=stateValues[0]
            agent=stateValues[1]
            depth=stateValues[2] 
            best_action = None

            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                value,_ = alphaBetaPruning((successor, agent + 1, depth),alpha, beta)
                v,best_action = max((v, best_action), (value, action))
                
                #Different cases for pacman and ghosts
                if agent % state.getNumAgents()==0:
                    if v > beta:
                        return v,best_action
                    alpha = max(alpha,v)
                else:
                    if v < alpha:
                        return v,best_action
                    beta = max(beta,v)
            return [v,best_action]
        
        alpha=float('-inf')
        beta=float('inf')
        _,best_action = alphaBetaPruning((gameState,0,0),alpha,beta)
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def checkTerminal(state,agent,depth):
            return state.isWin() or state.isLose() or (state.getLegalActions(agent)==0) or (depth==self.depth)
            
        def expectiMax(stateValues):
            state=stateValues[0]
            agent=stateValues[1]
            depth=stateValues[2] 
            
            if agent == state.getNumAgents():
                depth=depth + 1
                return expectiMax((state, 0,depth)) 

            if checkTerminal(state,agent,depth):
                return self.evaluationFunction(state) 

            children_nodes = []
            for action in state.getLegalActions(agent): 
                children_nodes.append(expectiMax((state.generateSuccessor(agent, action), agent + 1,depth)))
            
            if agent % state.getNumAgents() == 0:
                return max(children_nodes)
            else:
                return sum(children_nodes)*1.0/len(children_nodes)

        legalActions = gameState.getLegalActions(0) 
        max_utility = float('-inf')
        max_index = 0
        
        for index, action in enumerate(legalActions): 
            temp_utility = expectiMax((gameState.generateSuccessor(0, action), 1, 0))
            if temp_utility > max_utility:
                max_utility = temp_utility
                max_index = index
        
        return gameState.getLegalActions()[max_index]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: 
            1) Add amount of food within a radius of 3 to the predicted score
		   2) Nearest ghost: If the ghosts are scared increase the predicted score  or else decrease the score
 		   3) Nearest food: Increase the predicted score.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    pred_score = currentGameState.getScore()
    nearest_ghost_distance = manhattanDistance(newPos, newGhostStates[0].getPosition())
    
    def countSurroundingFood(newPos, newFood):
        count = 0
        for x in range(newPos[0]-3, newPos[0]+4):
            for y in range(newPos[1]-3, newPos[1]+4):
                if x>=0 and y>=0 and (x < len(list(newFood)) and y < len(list(newFood[1]))):
                        if newFood[x][y]:
                            count += 1
        return count
    
    def distanceToNearestFood(newPos,newFood):
        if(len(newFood.asList())):
            nearest_food = manhattanDistance(newPos, newFood.asList()[0])
            for food in newFood.asList():
                temp_food = manhattanDistance(newPos, food)
                if temp_food<nearest_food:
                    nearest_food = temp_food
            return nearest_food
        else:
            return 0
            
    total_surrounding_food_count=countSurroundingFood(newPos,newFood)
    pred_score = pred_score + total_surrounding_food_count

    if nearest_ghost_distance > 0: 
        if newScaredTimes[0]>0:
            pred_score = pred_score+(100/nearest_ghost_distance)
        else:
            pred_score = pred_score-(10/nearest_ghost_distance)

    distance_to_nearest_food=distanceToNearestFood(newPos,newFood)
    if distance_to_nearest_food!=0:
        pred_score = pred_score + (10/distance_to_nearest_food)

    return pred_score

# Abbreviation
better = betterEvaluationFunction


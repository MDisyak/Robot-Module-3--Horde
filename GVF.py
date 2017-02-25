#Written by Michael Disyak - Feb 2017
#This class represents a general value function and has the ability to
# recieve updates to relevant incoming data streams and learn using TD(lambda) and GTD(lambda) algorithms
#  This class is designed to be used in conjunction with the Plotter class and a control file that implements a behaviour policy

import numpy as np
import time
import math
class GenValFunc:


    prediction = 0.0
    cumulant = 0.0
    delta = 0.0
    predictions = []
    postPrediction = 0.0
    postReturn = 0.0
    timeDiff = 0
    gammaNext = 0
    gammaCurrent = 0
    lamb = 0
    row = 0
    action = 0
    numberOfLearningSteps = 0
    averageError = 0.0
    offPol = False
    targetPol = None
    rowFunc = None
    rupee = 0.0
    ude = 0.0
    alpha = 0.1
    betaNotRupee = (1 - lamb) * (alpha/30)

    def __init__(self, numTilings, numTilesTotal, alpha = 0.1, gamma = 0.9, lamb = 0.9, offPol = False, targetPolicy = None, beta = 0.01, rowFunc = None):
        self.offPol = offPol
        self.gammaCurrent = gamma
        self.lamb = lamb
        self.numTilings = numTilings
        self.numTilesTotal = numTilesTotal
        self.currentState = np.zeros(self.numTilesTotal)
        self.nextState = np.zeros(self.numTilesTotal)
        self.weightVect = np.zeros(self.numTilesTotal)
        self.hWeightVect = np.zeros(self.numTilesTotal)
        self.alpha = alpha
        self.beta = beta
   #     self.postTimeSteps = int(round(1.0/(1.0-self.gammaCurrent)))*5
        self.recordedCumulant = np.array([])
        self.recordedPrediction = np.array([])
        self.recordedGammas = np.array([])
        self.recordedError = np.array([])
        self.recordedUDE = np.array([])
        self.recordedRupee = np.array([])
        self.eligTrace = np.zeros(self.numTilesTotal)
        self.targetPol = targetPolicy
        self.rowFunc = rowFunc

        #Vars for RUPEE
        self.alphaRupee = self.alpha*5
        self.hRupee = np.zeros(self.numTilesTotal)
        self.deltaRupee = np.zeros(self.numTilesTotal)
        self.taoRupee = 0.0
        self.betaNotRupee = (1 - self.lamb) * (self.alpha/30)
        self.betaRupee = 0.0

        #Vars for UDE
        self.deltaUDE = 0.0
        self.taoUDE = 0.0
        self.betaNotUDE = self.alpha * 10
        self.betaUDE = 0.0
        self.varUDE = 0.0
        self.nUDE = 0
        self.deltaMean = 0.0
        self.oldDeltaMean = 0.0
        self.deltaM2 = 0.0
        self.epsilonUDE = 0.0001
    #Updates all values that can change after an action occurs
    def update(self, nextState, currentState, cumulant, lamb, gamma, action):
        self.nextState = nextState
        self.currentState = currentState
        self.cumulant = cumulant
        self.lamb = lamb
        self.gammaNext = gamma
        self.action = action
        if self.offPol:
            self.calcRow()

#Initiates the learning step for off and on-policy gvfs
    def learn(self):
        if self.offPol: #If on-policy do TD(lamb) else GTD(lamb)
            self.learnGTD()
        else:
            self.learnOnPol()
        if not self.pavlovianControl is None:
            self.pavlovianControl()
        self.calcRupee()
        self.calcUDE()

#Performs the learning step for on-policy GVFs using TD(lambda)
    def learnOnPol(self): #args, stoppingEvent):
        startTime = time.time()

        #TD ERROR BELOW
        self.currentStateValue = np.dot(self.weightVect, self.currentState)
        self.nextStateValue = np.dot(self.weightVect, self.nextState)
        self.delta = self.cumulant + ((self.gammaNext * self.nextStateValue) - self.currentStateValue)

        self.eligTrace = (self.lamb * self.gammaCurrent * self.eligTrace) + self.currentState
        self.weightVect = self.weightVect + (self.alpha * self.delta * self.eligTrace)

        self.prediction = self.currentStateValue
        self.predictions.append(self.prediction)

        self.numberOfLearningSteps += 1
        self.gammaCurrent = self.gammaNext
        self.timeDiff = round(time.time()-startTime,6)

#Performs the learning step for off-policy GVFs using GTD(lambda)
    def learnGTD(self):
        startTime = time.time()
        alphaGTD = self.alpha #* (1-self.lamb)
       #TD ERROR BELOW
        self.currentStateValue = np.dot(self.weightVect, self.currentState)
        self.nextStateValue = np.dot(self.weightVect, self.nextState)
        self.delta = self.cumulant + ((self.gammaNext * self.nextStateValue) - self.currentStateValue)
        #End TD Error

        self.eligTrace = self.row * (self.currentState + (self.lamb * self.gammaCurrent * self.eligTrace))
        self.weightVect += alphaGTD * ((self.delta * self.eligTrace) - ((self.gammaNext * (1-self.lamb)) * np.dot(self.eligTrace, self.hWeightVect) * self.nextState))
        self.hWeightVect += self.beta * ((self.delta * self.eligTrace) - (np.dot(self.hWeightVect, self.currentState) * self.currentState))
        self.prediction = self.currentStateValue
        self.predictions.append(self.prediction)

        self.numberOfLearningSteps += 1
        self.gammaCurrent = self.gammaNext
        self.timeDiff = round(time.time()-startTime,6)

#TODO: be overwritten dynamically upon creation
#Determines if the action taken matches the target policy of the off-policy gvf for the purpose of importance sampling
    def calcRow(self):
        if self.action == self.targetPol:
            self.row = 1
        else:
            self.row = 0
        #self.rowFunction(self, self.targetPol, self.action)

#Calculates an approximation of the true return post-hoc
    def verifier(self):

        self.recordedPrediction = np.append(self.recordedPrediction, [self.prediction])
        self.predictions.append(self.prediction)

        self.recordedCumulant = np.append(self.recordedCumulant, [self.cumulant])
        self.recordedGammas = np.append(self.recordedGammas, [self.gammaCurrent])

        if np.size(self.recordedCumulant) == self.postTimeSteps + 1:
            currentPostPrediction = self.recordedPrediction[0]
            returnTotal = 0
            gammaTotal = 1
            self.recordedGammas[0] = 1

            for i in range(0,np.size(self.recordedCumulant)-1): #0 to length of your recorded cumulant

                currentCumulant = self.recordedCumulant[i]
                gammaTotal = gammaTotal * self.recordedGammas[i]
                returnTotal = returnTotal + (gammaTotal * currentCumulant)

            self.postReturn = returnTotal
            self.postPrediction = currentPostPrediction
            self.recordedError = np.append(self.recordedError, returnTotal - currentPostPrediction)
            if np.size(self.recordedError) == self.postTimeSteps+1:
                self.recordedError = np.delete(self.recordedError, 0)
            self.averageError = np.sum(self.recordedError)/self.postTimeSteps
            self.recordedCumulant = np.delete(self.recordedCumulant, 0)
            self.recordedPrediction = np.delete(self.recordedPrediction, 0)
            self.recordedGammas = np.delete(self.recordedGammas, 0)

# Calcualtes RUPEE for the GVF
    def calcRupee(self):
        self.hRupee = self.hRupee + (self.alphaRupee*((self.delta * self.eligTrace) - (np.dot(np.transpose(self.hRupee),self.currentState) * self.currentState)))
        self.taoRupee = ((1 - self.betaNotRupee) * self.taoRupee) + self.betaNotRupee
        self.betaRupee = self.betaNotRupee/self.taoRupee
        self.deltaRupee = ((1-self.betaRupee)*self.deltaRupee) + (self.betaRupee * self.delta * self.eligTrace)
        self.rupee = math.sqrt(abs(np.dot(np.transpose(self.hRupee), self.deltaRupee)))
        self.recordedRupee = np.append(self.recordedRupee, self.rupee)

#Calculates Unexpected Demon Error for the GVF
    def calcUDE(self):
        self.taoUDE = ((1.0 - self.betaNotUDE) * self.taoUDE) + self.betaNotUDE
        self.betaUDE = self.betaNotUDE / self.taoUDE
        self.deltaUDE = ((1.0 - self.betaUDE) * self.deltaUDE) + (self.betaUDE * self.delta)
        self.calcVariance()
        self.ude = abs(round(self.deltaUDE,4)/(math.sqrt(round(self.varUDE,4)) + self.epsilonUDE))

#This method was taken from the Online Algorithm section of "Algorithms for calculating variance" on Wikipedia, Feb 22, 2017
#https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
#Function calculates variance in an online and incremental way
    def calcVariance(self):
        self.nUDE += 1
        self.oldDeltaMean = self.deltaMean
        self.deltaMean = (1.0 - self.betaUDE) * self.deltaMean + self.betaUDE * self.delta
        self.varUDE = ((self.nUDE - 1) * self.varUDE + (self.delta - self.oldDeltaMean) * (self.delta - self.deltaMean))/self.nUDE

    def pavlovianControl(self):
        test = True





import GVF
import json
import datetime
class Horde:

    horde = []
    numDemons = 0

#defaults to 8x8 tiles with 8 tilings (assumes 2 dimensional data)
    def createHorde(self, alphaList, gammaList, lambList, numDemons=10, numTilings=8, numTilesTotal=8*8*8, offPolList = None, targetPolicyList = None, betaList = None, pavList = None, rowFuncList = None, ):
        self.numDemons = numDemons
        if pavList is None:
            pavList = [None] * numDemons
        if betaList is None:
            betaList = [None] *numDemons
        if offPolList is None:
            offPolList = [False]*numDemons
        if targetPolicyList is None:
            targetPolicyList = [None]*numDemons
        if rowFuncList is None:
            rowFuncList = [None]*numDemons
        for i in range(0, numDemons):
            self.horde.append(GVF.GenValFunc(numTilings, numTilesTotal, alphaList[i], gammaList[i], lambList[i], offPolList[i], targetPolicyList[i], betaList[i], rowFuncList[i]))

    def update(self, varDict):
        nextState = varDict['nextState']
        currentState = varDict['currentState']
        cumulantList = varDict['cumulantList']
        gammaList = varDict['gammaList']
        lambList = varDict['lambList']
        action = varDict['action']

        for i in range(0, self.numDemons):
            self.horde[i].update(nextState, currentState, cumulantList[i], lambList[i], gammaList[i], action)

    def learn(self):
        for i in range(0, self.numDemons):
            self.horde[i].learn()

    def getHorde(self):
        return self.horde

    def getPredictions(self):
        pred = []
        for i in range(0, self.numDemons):
            pred.append(self.horde[i].prediction)
        return pred

    def getGVF(self, numGVF):
        return self.horde[numGVF]

    def getRupee(self):
        rupee = []
        for i in range(0, self.numDemons):
            rupee.append(self.horde[i].rupee)
        return rupee

    def getUDE(self):
        ude = []
        for i in range(0, self.numDemons):
            ude.append(self.horde[i].ude)
        return ude

    def writePredictions(self):
        stringOut = "{"
        for i in range(0, self.numDemons):
            gvf = self.horde[i]
            stringOut += 'gvf' + str(i) + ' : ' + str(gvf.predictions) + ", "
        stringOut += "}"
        with open('predictions_' + str(datetime.datetime.now()) + '.json', 'w') as f:
            json.dump(stringOut, f)

import lib_robotis_hack
import myLib
import numpy as np
import RLtoolkit
import json
import datetime

class ObservationManager:
    s1 = None
    s2 = None
    currentAngle = 0.0
    currentAngleS2 = 0.0
    currentLoad = 0.0
    currentTemperature = 0.0
    currentVoltage = 0.0
    angles = []
    loads = []
    temperatures = []
    voltages = []
    actions = []
    currentAction = 0
    currentState = None
    nextState = None
    numTilings = 24
    obsIndex = 0
    numTiles = 24
    numTilesTotal = numTiles * numTiles * numTiles * numTilings

    def __init__(self, servos):
        if servos is not None:
            self.s1 = servos[0]
            self.s2 = servos[1]

    def getObs(self, action=0):
        self.currentAngle = self.s1.read_angle()
        self.currentAngleS2 = self.s2.read_angle()
        self.currentLoad = self.s1.read_load()
        self.currentTemperature = self.s1.read_temperature()
        self.currentVoltage = self.s1.read_voltage()
        self.angles.append(self.currentAngle)
        self.loads.append(self.currentLoad)
        self.temperatures.append(self.currentTemperature)
        self.voltages.append(self.currentVoltage)
        self.actions.append(action)
        self.currentAction = action


    def initStateFromFile(self):
        self.getObsFromIndex()
        currentLoad = myLib.normalizeLoad(self.currentLoad)
        #  print 'load is: ' + str(currentLoad)
        currentAngle = myLib.normalizeAngle(self.currentAngle)
        obs = [currentAngle * self.numTiles,
               currentLoad * self.numTiles, self.currentAction]  # myLib.normalizeLoad(float(s1.read_load()))*numTiles]  # , s2.read_encoder()/853/.1, s2.read_load()/1023/.1]
        currentStateSparse = RLtoolkit.tiles.tiles(self.numTilings, self.numTilesTotal, obs)  # tileIndices
        currentState = np.zeros(self.numTilesTotal)
        for index in currentStateSparse:  # Convert to full vector of 0s and 1s
            currentState[index] = 1
            #  nextState = np.append(nextState, 1)#bias bit
        self.currentState = currentState


    def getObsFromIndex(self):
        index = self.obsIndex
        self.currentAngle = self.angles[index]
        self.currentLoad = self.loads[index]
        self.currentTemperature = self.temperatures[index]
        self.currentVoltage = self.voltages[index]
        self.currentAction = self.actions[index]
        self.obsIndex += 1

    def initState(self):
        self.getObs()
        # Return state values scaled to unit length and then scaled into number of tiles
        currentLoad = myLib.normalizeLoad(self.currentLoad)
      #  print 'load is: ' + str(currentLoad)
        currentAngle = myLib.normalizeAngle(self.currentAngle)
        obs = [currentAngle * self.numTiles,
               currentLoad * self.numTiles, self.currentAction]  # myLib.normalizeLoad(float(s1.read_load()))*numTiles]  # , s2.read_encoder()/853/.1, s2.read_load()/1023/.1]
        currentStateSparse = RLtoolkit.tiles.tiles(self.numTilings, self.numTilesTotal, obs)  # tileIndices
        currentState = np.zeros(self.numTilesTotal)
        for index in currentStateSparse:  # Convert to full vector of 0s and 1s
            currentState[index] = 1
            #  nextState = np.append(nextState, 1)#bias bit
        self.currentState = currentState

    def getState(self):
        # Return state values scaled to unit length and then scaled into number of tiles
        currentLoad = myLib.normalizeLoad(self.currentLoad)
       # print 'load is: ' + str(currentLoad)
        currentAngle = myLib.normalizeAngle(self.currentAngle)
        obs = [currentAngle * self.numTiles,
               currentLoad * self.numTiles, self.currentAction]  # myLib.normalizeLoad(float(s1.read_load()))*numTiles]  # , s2.read_encoder()/853/.1, s2.read_load()/1023/.1]
        nextStateSparse = RLtoolkit.tiles.tiles(self.numTilings, self.numTilesTotal, obs)  # tileIndices
        nextState = np.zeros(self.numTilesTotal)
        for index in nextStateSparse:  # Convert to full vector of 0s and 1s
            nextState[index] = 1
            #  nextState = np.append(nextState, 1)#bias bit
        self.nextState = nextState
        return nextState

    def writeObs(self):
        data = {'angles' : self.angles, 'loads' : self.loads , 'temperatures' : self.temperatures, 'voltages' : self.voltages, 'actions' : self.actions}
        with open('obs_' + str(datetime.datetime.now()) + '.json', 'w') as f:
            json.dump(data, f)

    def readFromFile(self, fileName):
        with open(fileName, 'r') as f:
            data = json.load(f)
        self.angles = data['angles']
        self.loads = data['loads']
        self.temperatures = data['temperatures']
        self.voltages = data['voltages']
        self.actions = data['actions']
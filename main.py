#Author: Michael Disyak
# Implements a behaviour policy and a uses a general value function (GVF) to make predictions about various signals.
#This file also utilized Plotter class to plot the data coming from the GVF

from lib_robotis_hack import *
import myLib as myLib
import threading
import random
random.seed(0)
import GVF
import RLtoolkit.tiles as tile
import numpy as np
import copy
import Horde
import datetime
import ObservationManager
import HordePlotter
import time
import types


pavGVF = None
obsMan = None
s1 = None
s2 = None

def pavlovianControl():
    if pavGVF.prediction > 90:
        if s2 is not None:

            s2.move_angle(myLib.degToRad(obsMan.currentAngleS2 + 10))
            s2.move_angle(obsMan.currentAngleS2)
        else:
            print('PAVLOV CONTROL!!!!!!!!!!!!!!')

def startFromFile(fileName):

#Set up all values for the current run

    obsMan = ObservationManager.ObservationManager(None)

    numTilings = obsMan.numTilings
    numTilesTotal = obsMan.numTilesTotal


    hordeSize = 10
    stdAlpha = 0.1/numTilings
    alphaList = [stdAlpha] * hordeSize
    betaList = [stdAlpha/10]*hordeSize
    gammaList = [0, 0.5, 0.75, 0.9, 0.98, 0.99, 0.999, 1, 1, 1] #1 ts, 2 ts, 4ts, 10ts, 50 ts, 100ts, 1000ts, state dep, offpol, offpol
    lambList = [0.9] * hordeSize
    offPolList= [False, False, False, False, False, False, False, False, True, True]
    targetPolicyList = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0] #Defines the target action for Off-policy

   #Create horde object
    horde = Horde.Horde()
   #Create horde of GVFs
    horde.createHorde(alphaList, gammaList, lambList, hordeSize, numTilings, numTilesTotal, offPolList, targetPolicyList, betaList)

    plotter = HordePlotter.HordePlotter(horde, obsMan)

    obsMan.readFromFile(fileName)
    obsMan.initStateFromFile()
    plotter.initPlot()
    numberOfActions = 0

    while (obsMan.obsIndex < len(obsMan.angles)):
        startTime = time.time()
        currentPosRad = obsMan.currentAngle
        #   myPlotter.currentLoad = myLib.normalizeLoad(s1.read_load())
        currentPos = myLib.radToDeg(currentPosRad)

        #No action, just reading from next state
        obsMan.getObsFromIndex()  # GET AND STORE
        nextState = obsMan.getState()  # GET STATE APPROXIMATION VIA TILECODING

        #Set cumulant for GVFs
        cumulantList = [obsMan.currentLoad] * hordeSize
        cumulantList[7] = 1 #Cumulant that reflects counting timesteps

        #State dependancy for GVF
        if (obsMan.currentLoad > 90):
            gammaList[7] = 0
        else:
            gammaList[7] = 1

        #Update data for GVFS
        updateDict = {'nextState': obsMan.nextState, 'currentState': obsMan.currentState, 'cumulantList': cumulantList,
                      'gammaList': gammaList, 'lambList': lambList, 'action': obsMan.currentAction}
        horde.update(updateDict)

        # Conduct learning step for GVFs
        horde.learn()
        #Set current state to next state
        obsMan.currentState = copy.deepcopy(nextState)

        #Plotting
        plotter.controlTime = round(time.time() - startTime, 3)
        numberOfActions += 1
        plotter.numberOfActions = numberOfActions
        plotter.plot()

    #Write predictions to file and save figure
    horde.writePredictions()
    plotter.saveFigure()

def startFromRobot(args, realTimeStop):
    global gammaList, lambList, hordeSize, left, rowFuncList, pavGVF, s1, s2, obsMan

    D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QEMU", baudrate=1000000)
    s_list = find_servos(D)
    s1 = Robotis_Servo(D, s_list[0])
    s2 = Robotis_Servo(D, s_list[1])

    left = False  # flag for movement direction in control function
    action = 0

    obsMan = ObservationManager.ObservationManager([s1, s2])

    numTilings = obsMan.numTilings
    numTilesTotal = obsMan.numTilesTotal

    hordeSize = 10
    stdAlpha = 0.1/numTilings
    alphaList = [stdAlpha] * hordeSize
    betaList = [stdAlpha/10]*hordeSize
    gammaList = [0, 0.5, 0.75, 0.9, 0.98, 0.99, 0.999, 1, 1, 1] #1 ts, 2 ts, 4ts, 10ts, 50 ts, 100ts, 1000ts, state dep, offpol, offpol
    lambList = [0.9] * hordeSize
    offPolList= [False, False, False, False, False, False, False, False, True, True]
    targetPolicyList = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0] #Defines the target action for Off-policy

    # Create horde object
    horde = Horde.Horde()
    # Create horde of GVFs
    horde.createHorde(alphaList, gammaList, lambList, hordeSize, numTilings, numTilesTotal, offPolList,
                      targetPolicyList, betaList)

    pavGVF = horde.getGVF(9)
    pavGVF.pavlovianControl = pavlovianControl

    plotter = HordePlotter.HordePlotter(horde, obsMan)

    obsMan.initState()#set initial state
    plotter.initPlot()
    numberOfActions = 0

    timeEnd = time.time() + 60 * 15#mins
    while (time.time() < timeEnd):
        startTime= time.time()
        currentPosRad = obsMan.currentAngle
     #   myPlotter.currentLoad = myLib.normalizeLoad(s1.read_load())
        currentPos = myLib.radToDeg(currentPosRad)
        if left:
            s1.move_angle(myLib.degToRad(currentPos + 20))
            if currentPos >= 80:
                left = False
                action = 0
        else:
            s1.move_angle(myLib.degToRad(currentPos - 20))
            if currentPos <= -80:
                left = True
                action = 1

        obsMan.getObs(action) #GET AND STORE
        nextState = obsMan.getState() #GET STATE APPROXIMATION VIA TILECODING

        cumulantList = [obsMan.currentLoad]*hordeSize#myLib.normalizeLoad(s1.read_load())#myLib.radToDeg(s1.read_angle())
        cumulantList[7] = 1 #State dependant
        if (obsMan.currentAngle > myLib.degToRad(90)):
            gammaList[7] = 0
        else:
            gammaList[7] = 1
        #MAYBE STEP gvf.update(nextState, currentState, cumulant, gamma, maybe row for GTD) - Update values for GVF
        updateDict = {'nextState': obsMan.nextState, 'currentState': obsMan.currentState, 'cumulantList': cumulantList, 'gammaList': gammaList, 'lambList': lambList, 'action': action}
        horde.update(updateDict)
        #STEP 3 learn stuff
        horde.learn()

        obsMan.currentState = copy.deepcopy(nextState)

        plotter.controlTime = round(time.time()-startTime,3)
        numberOfActions += 1
        plotter.numberOfActions = numberOfActions
        plotter.plot()
    obsMan.writeObs() #write observations to file
    horde.writePredictions()
    realTimeStop.set()
    plotter.saveFigure()



realTimeStop = threading.Event()
args = None
startFromFile('obs.json')
#startFromRobot(args, realTimeStop)


#threading.Thread(target=controlFunctionOnpolicyConditionalGamma, args=(1, realTimeStop)).start()
#threading.Thread(target=gvf.learn, args=(1, realTimeStop)).start()
#myPlotter.plotGVF(realTimeStop)
#controlFunction(None, realTimeStop)

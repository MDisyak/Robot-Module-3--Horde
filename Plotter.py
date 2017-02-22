import myLib
import matplotlib.pyplot as plt
import datetime
import time
import csv
import threading
import numpy as np

class Plotter:
    controlTime = 0.0
    currentAngle = 0.0
    currentLoad = 0.0
    numberOfActions = 0
    def __init__(self, GVFinput):
        self.gvf = GVFinput
        #self.initPlotter()

    def saveFigure(self, fileName='Figure_from_%s.png' % datetime.datetime.now()):
        plt.savefig('figures/%s' % fileName)


    def initPlotGVF(self):
        plt.ion()  # Turn interactive on

        self.graphSpan = 100 # the width of the graph

        self.maxY =25
        self.minY = -10

        x = np.arange(0, self.graphSpan)

        # Init the fig

        self.fig, (self.plot1Ax, self.plot2Ax) = plt.subplots(2)

        #init the data to be plotted
        self.angle = [0] * self.graphSpan
        self.load = [0] * self.graphSpan
        self.cumulant = [0] * self.graphSpan
        self.prediction = [0] * self.graphSpan


        # initial plot
        (self.angleLine, self.loadLine, self.cumulantLine, self.predictionLine) = self.plot1Ax.plot(x, self.angle, 'b', x, self.load, 'y', x, self.cumulant, 'g', x, self.prediction, 'r', linewidth=3)

        self.plot1Ax.axes.set_xlim(0, self.graphSpan)
        self.plot1Ax.axes.set_ylim(self.minY, self.maxY)

        self.angleLine.set_label('Angle')
        self.loadLine.set_label('Load')
        self.cumulantLine.set_label('Cumulant')
        self.predictionLine.set_label('Prediction')

        # Initialize the second graph ------------------------------================-----------------------


        #init the data to be plotted
        self.error = [0] * self.graphSpan
        self.postReturn = [0] * self.graphSpan
        self.postPrediction = [0] * self.graphSpan

        # initial plot
        (self.errorLine, self.postReturnLine, self.postPredictionLine) = self.plot2Ax.plot(x, self.error, 'y', x, self.postReturn, 'g', x, self.postPrediction, 'r', linewidth=3)

        self.plot2Ax.axes.set_xlim(0, self.graphSpan)
        self.plot2Ax.axes.set_ylim(self.minY, self.maxY)

        self.errorLine.set_label("Error")
        self.postReturnLine.set_label('Post-hoc Prediction')
        self.postPredictionLine.set_label('Prediction')


        self.plot1Ax.axes.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        self.plot1Ax.axes.grid()

        self.plot2Ax.axes.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        self.plot2Ax.axes.grid()
        self.controlTimeText = plt.text(0,self.minY - 5, "Time Control: " + str(0.0))
        self.gvfTimeText = plt.text(75,self.minY - 5, "Time GVF: " + str(0.0))
        self.elapsedTime = plt.text(56,self.minY - 5, "Elapsed Time: " + str(0.0))
        self.errorText = plt.text(38,self.minY - 5, "Error: " + str(0.0))
        self.avgErrorText = plt.text(18,self.minY - 5, "Avg Error: " + str(0.0))

        self.numLearnText = plt.text(2,self.minY + 4 , "# Learn step: " + str(0.0))
        self.numActionsText = plt.text(32,self.minY +4, "# Actions: " + str(0.0))
        self.trueReturnText = plt.text(52,self.minY +4, "True Return: " + str(0.0)) #change to min +4 4
        self.predictionText = plt.text(78,self.minY +4, "Prediction: " + str(0.0))
        plt.pause(0.05)

    def runPlotGVF(self):
        # take measurements
        currentAngle = self.currentAngle#self.s1.read_angle()#myLib.radToDeg(self.s1.read_angle())
        currentCumulant = self.gvf.cumulant
        currentPrediction = self.gvf.prediction
        currentLoad = self.currentLoad

        # add the newest to end of arrays

        self.angle.append(currentAngle)
        self.load.append(currentLoad)
        self.cumulant.append(currentCumulant)
        self.prediction.append(currentPrediction)

        self.angleLine.set_ydata(self.angle[-self.graphSpan:])
        self.loadLine.set_ydata(self.load[-self.graphSpan:])
        self.cumulantLine.set_ydata(self.cumulant[-self.graphSpan:])
        self.predictionLine.set_ydata(self.prediction[-self.graphSpan:])

        #SECOND PLOT ---------======================-----------------------------------
        # take measurements
        currentPostPrediction = self.gvf.postPrediction
        currentPostReturn = self.gvf.postReturn
        currentError = abs(currentPostReturn - currentPostPrediction)
        # add the newest to end of arrays
        self.error.append(currentError)
        self.postReturn.append(currentPostReturn)
        self.postPrediction.append(currentPostPrediction)
        self.errorLine.set_ydata(self.error[-self.graphSpan:])
        self.postReturnLine.set_ydata(self.postReturn[-self.graphSpan:])
        self.postPredictionLine.set_ydata(self.postPrediction[-self.graphSpan:])

        self.controlTimeText.set_text("Time Control: " + str(self.controlTime))
        self.gvfTimeText.set_text("Time GVF: " + str(self.gvf.timeDiff))
        self.errorText.set_text("Error: " + str(round(currentError,4)))
        self.avgErrorText.set_text("Avg Error: " + str(round(self.gvf.averageError,4)))
        self.elapsedTime.set_text("Run Time: " + str(round((time.time() - self.startTimeRun), 3)))

        self.numLearnText.set_text("# Learn Steps: " + str(self.gvf.numberOfLearningSteps))
        self.numActionsText.set_text("# Actions: " + str(self.numberOfActions))
        self.trueReturnText.set_text("True Return: " + str(self.gvf.postReturn))
        self.predictionText.set_text("Prediction: " + str(self.gvf.postPrediction))
        plt.pause(0.05)

    def plotGVF(self, stoppingEvent):
        self.startTimeRun = time.time()
        self.initPlotGVF()
        while not stoppingEvent.is_set():

            self.runPlotGVF()
            #time.sleep(.5)
        self.saveFigure('GVF2__alpha_' + str(self.gvf.alpha) + '_tilings_' + str(self.gvf.numTilings) + '_gamma_' + str(self.gvf.gamma)+ '_lambda_' + str(self.gvf.lamb) + '.png')

    def initPlotter(self):
        self.startTimeRun = time.time()
        self.initPlotGVF()

    def plot(self):
        self.runPlotGVF()
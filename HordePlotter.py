import myLib
import matplotlib.pyplot as plt
import datetime
import time
import csv
import threading
import numpy as np

class HordePlotter:
    horde = None
    gvfList = []

    graphSpan = 100  # width of plot
    graphSpanL = 900  # width of large plot
    currentAngle = 0.0
    currentLoad = 0.0
    currentTemperature = 0.0
    currentVoltage = 0.0
    observationManager = None
    predMinY = -300#-500
    predMaxY = 300#500
    obsMinY = -200
    obsMaxY = 200
    rupMinY = 0
    rupMaxY = 100
    udeMinY = -500
    udeMaxY = 500


    def __init__(self, horde, obsMan):
            self.horde = horde
            self.observationManager = obsMan
            self.gvfList = horde.getHorde()
            self.angle = [0] * self.graphSpan
            self.load = [0] * self.graphSpan
            self.temperature = [0] * self.graphSpan
            self.voltage = [0] * self.graphSpan
            self.ude = []
            self.rupee = []
            self.pred = []
            for i in range(0, len(self.gvfList)):
                self.pred.append([0]*self.graphSpan)
                self.rupee.append([0]*self.graphSpanL)
                self.ude.append([0]*self.graphSpanL)


    def initPlot(self):
        plt.ion()
        self.fig, (self.predictionAx, self.obsAx, self.rupeeAx, self.udeAx) = plt.subplots(4)
        self.fig.subplots_adjust(left=0.05, right=.75, hspace=1)

        x = np.arange(0, self.graphSpan)#1 - dimensional x for smaller scale
        xL = np.arange(0, self.graphSpanL)
        self.x2 = []#2 dimensional x
        self.x2L = []#2 dimensional x for larger scale
        for i in range(0, len(self.gvfList)):
            self.x2.append(x)
            self.x2L.append(xL)
        self.predLines = self.predictionAx.plot(self.x2, self.pred)
        [self.angleLine, self.loadLine, self.temperatureLine, self.voltageLine] = self.obsAx.plot(x, self.angle, 'b', x, self.load, 'g', x, self.temperature, 'r', x, self.voltage, 'y')
        self.rupeeLines = self.rupeeAx.plot(self.x2L, self.horde.getRupee())
        self.udeLines = self.udeAx.plot(x, self.horde.getUDE())


        #Set line labels
        obsNames = ["angle (deg)", "load (dyna.)", "temp. (cels.)", "volt."]
        self.angleLine.set_label(obsNames[0])
        self.loadLine.set_label(obsNames[1])
        self.temperatureLine.set_label(obsNames[2])
        self.voltageLine.set_label(obsNames[3])
        predNames = ['0', '0.5', '0.75', '0.9', '0.98', '0.99', '0.999', 'State dep.', '1(off-pol)', '0(off-pol)']
        for i in range(0, 10):
            self.predLines[i].set_label(predNames[i])
            self.rupeeLines[i].set_label(predNames[i])
            self.udeLines[i].set_label(predNames[i])
            if i == 7:
                self.predLines[i].set_linestyle('dashed')
                self.rupeeLines[i].set_linestyle('dashed')
                self.udeLines[i].set_linestyle('dashed')
            if i > 7:
                self.predLines[i].set_linewidth(2)
                self.rupeeLines[i].set_linewidth(2)
                self.udeLines[i].set_linewidth(2)

        #Set legend position and y-limits
        self.obsAx.axes.legend(bbox_to_anchor=(1, .5), loc='center left', ncol=2, title='Observations')
        self.obsAx.axes.set_ylim(self.obsMinY, self.obsMaxY)
        self.predictionAx.axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, title='Gamma(On-policy)/Target(Off-policy)')
        self.predictionAx.axes.set_ylim(self.predMinY, self.predMaxY)
        self.rupeeAx.axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, title='Gamma(On-policy)/Target(Off-policy)')
        self.rupeeAx.axes.set_ylim(self.rupMinY, self.rupMaxY)
        self.udeAx.axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, title='Gamma(On-policy)/Target(Off-policy)')
        self.udeAx.axes.set_ylim(self.udeMinY, self.udeMaxY)

        #Set axis label
        self.obsAx.set_xlabel("Timesteps")
        self.predictionAx.set_xlabel("Timesteps")
        self.rupeeAx.set_xlabel("Timesteps")
        self.udeAx.set_xlabel("Timesteps")

        #Set title
        self.obsAx.set_title('Observations From Robot')
        self.predictionAx.set_title("Demon Predictions")
        self.rupeeAx.set_title("RUPEE Measure For Each Demon")
        self.udeAx.set_title("UDE For Each Demon")

        #Dynamic text
        self.numLearnText = plt.text(0,0, "# Learn step: " + str(0.0))

        plt.pause(0.05)

    def plot(self):
       #get and set data for ude, rupee and prediction plots
        for i in range(0, len(self.gvfList)):
            curGVF = self.gvfList[i]
            self.pred[i].append(curGVF.prediction)
            self.rupee[i].append(curGVF.rupee)
            self.ude[i].append(curGVF.ude)

        for i in range(0, len(self.pred)):
            currentPrediction = self.pred[i]
            currentRupee = self.rupee[i]
            currentUDE = self.ude[i]

            self.predLines[i].set_xdata(self.x2[i])
            self.predLines[i].set_ydata(currentPrediction[-self.graphSpan:])
            self.rupeeLines[i].set_xdata(self.x2L[i])
            self.rupeeLines[i].set_ydata(currentRupee[-self.graphSpanL:])
            self.udeLines[i].set_xdata(self.x2L[i])
            self.udeLines[i].set_ydata(currentUDE[-self.graphSpanL:])



            #get and set data for observation plot
        self.angle.append(myLib.radToDeg(self.observationManager.currentAngle))
        self.load.append(self.observationManager.currentLoad)
        self.voltage.append(self.observationManager.currentVoltage)
        self.temperature.append(self.observationManager.currentTemperature)

        self.angleLine.set_ydata(self.angle[-self.graphSpan:])
        self.loadLine.set_ydata(self.load[-self.graphSpan:])
        self.temperatureLine.set_ydata(self.temperature[-self.graphSpan:])
        self.voltageLine.set_ydata(self.voltage[-self.graphSpan:])

        self.numLearnText.set_text("# Learn Steps: " + str(self.observationManager.obsIndex))

        plt.pause(0.05)


    def saveFigure(self, fileName='HordeFigure_from_%s.png' % datetime.datetime.now()):
        plt.savefig('figures/%s' % fileName)

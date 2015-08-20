__author__ = 'sudish'
from Tkinter import *
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class gui(Tk):

    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot(111, projection="3d")

    def __init__(self):
        self.root = Tk()

        self.root.geometry("300x200")
        self.strD = StringVar()
        self.strT = StringVar()
        self.strA = StringVar()

        label_d = Label(self.root, text = "Enter Diameter")
        label_t = Label(self.root, text="Enter Thickness")
        label_a = Label(self.root, text="Enter Aperture")

        dEntry = Entry(self.root, textvariable = self.strD)
        tEntry = Entry(self.root, textvariable = self.strT)
        aEntry = Entry(self.root, textvariable = self.strA)


        label_d.grid(row = 0, column = 0, sticky = E)
        label_t.grid(row = 1, column = 0, sticky = E)
        label_a.grid(row = 2, column = 0, sticky = E)

        dEntry.grid(row = 0, column = 1)
        tEntry.grid(row = 1, column = 1)
        aEntry.grid(row = 2, column = 1)

        enter_button = Button(self.root, text='Calculate', command=self.calcProb)
        enter_button.grid(row = 3, column = 1)

        self.root.mainloop()

    def dofuncs(self):
        self.graph()
        self.calcProb()


    def calcProb(self):
        intD = float(self.strD.get())
        intT = float(self.strT.get())
        intA = float(self.strA.get())

        theta1 = self.getTheta1()
        theta2 = self.getTheta2()


        x = matrix([intD, intT, intA])
        x = insert(x, 0, 1)

        h1 = self.sigmoid(x * theta1.transpose())
        h1 = insert(h1, 0, 1)
        h2 = self.sigmoid(h1 * theta2.transpose())

        label_pred1 = Label(self.root, text = "Chance of being money bead: %%%.2f" %(h2[0,0] * 100))
        label_pred2 = Label(self.root, text = "Chance of being basket bead: %%%.2f" %(h2[0,1] * 100))
        label_pred1.grid(row = 4, columnspan = 3, sticky = W)
        label_pred2.grid(row = 5, columnspan = 3, sticky = W)

    """
    def graph(self):
        Xmoney = []
        Ymoney = []
        Zmoney = []
        Xbasket = []
        Ybasket = []
        Zbasket = []

        with open("data.txt") as file:
            data = file.read()
            strList = data.split()
            for i in xrange(0, 1281, 3):
                Xmoney.append(float(strList[i]))
                Ymoney.append(float(strList[i+1]))
                Zmoney.append(float(strList[i+2]))

            for i in xrange(1281, len(strList), 3):
                Xbasket.append(float(strList[i]))
                Ybasket.append(float(strList[i+1]))
                Zbasket.append(float(strList[i+2]))



        self.ax.scatter(Xmoney, Ymoney, Zmoney, c = 'r')
        self.ax.scatter(Xbasket, Ybasket, Zbasket, c = 'b')
        self.ax.set_xlabel("Diameter")
        self.ax.set_ylabel("Thickness")
        self.ax.set_zlabel("Aperture")
        self.fig.show()

    """
    def addPoint(self, D, T, A):
        self.ax.scatter(D, T, A, c = 'g')

    def getTheta1(self):
        theta1 = []
        for line in open("theta1.txt"):
            str_row = line.split()
            f_row = [float(val) for val in str_row]
            theta1.append(f_row)
        return matrix(theta1)


    def getTheta2(self):
        theta2 = []
        for line in open("theta2.txt"):
            str_row = line.split()
            f_row = [float(val) for val in str_row]
            theta2.append(f_row)
        return matrix(theta2)

    def sigmoid(self, z):
        g = 1.0 / (1.0 + exp(-z))
        return g


window = gui()








import numpy as np
from scipy.sparse.linalg import eigs
import matplotlib.pylab as plt
import time
t = time.time()

def gradientDescent(data, dataSettings, trainingSettings, cLambda, nTasks):
    gamma = trainingSettings['gamma']
    taskRange = dataSettings['taskRange']
    nIter = trainingSettings['nIter']
    convTol = trainingSettings['convTol']
    nDims = dataSettings['nDims']
    pointsPerT = data['pointsPerT']
    XtX = data['XtX']

    # functions
    objFun = lambda W: lossBatch(data, W) + cLambda*(gamma*(1/nTasks)*np.linalg.norm(W, 'fro')**2 + (1 - gamma)*VARbatch(W))
    gradFun = lambda W: gradLoss(data, W) + cLambda*(2*gamma*W/nTasks + (1-gamma)*gradVARbatch(W))

    # Lipschitz constant
    AllLipschitz = []
    cTaskIDX = np.int(np.squeeze(np.argwhere(taskRange == nTasks)))
    for cTask in range(0, cTaskIDX + 1):
        cMatrix = 2 / (pointsPerT[cTask]*nTasks) * XtX[cTask] + 2*cLambda*(1/nTasks)*(gamma +
                                                                       (1-gamma)*(1-1/nTasks))*np.identity(nDims)
        AllLipschitz.append(np.double(np.sqrt(eigs(cMatrix.dot(cMatrix.T), k=1)[0]).real))
    Lipschitz = 2*np.max(AllLipschitz)
    stepSize = 1 / Lipschitz
    # stepSize = 10 ** -3

    currW = trainingSettings['WpredVARbatch'] # np.random.randn(nDims, nTasks)
    currObj = objFun(currW)

    objectives = []
    currTol = 10 ** 10
    cIter = 0

    plt.figure()
    while (cIter < nIter) and (currTol > convTol):
        prevW = currW
        prevObj = currObj

        currW = prevW - stepSize * gradFun(prevW)

        currObj = objFun(currW)
        objectives.append(currObj)

        currTol = abs(currObj - prevObj) / prevObj

        if (cIter % 10000 == 0):
            plt.plot(np.log10(objectives), "b")
            plt.title("number of tasks: %s" % (nTasks))
            plt.pause(0.0001)
            print("nTasks: %3d | iter: %8ld | tol: %20.18f | time: %7.2f" % (nTasks, cIter, currTol, time.time() - t))
        cIter = cIter + 1
    objectives = np.array(objectives)
    plt.plot(np.log10(objectives), "b")
    plt.title("number of tasks: %s" % (nTasks))
    plt.pause(0.0001)
    plt.close()
    print("nTasks: %3d | iter: %8ld | tol: %20.18f | time: %7.2f" % (nTasks, cIter, currTol, time.time() - t))
    return currW, objectives



def lossBatch(data, W):
    nTasks = W.shape[1]
    pointsPerT = data['pointsPerT']
    XtX = data['XtX']
    XtY = data['XtY']
    YtY = data['YtY']

    lossFun = lambda w, xtx, xty, yty, pointsCurrT: (1/pointsCurrT) * (np.transpose(w).dot(xtx).dot(w)
                                                    - 2 *  np.transpose(w).dot(xty) + yty)
    losses = []
    for cTask in range(0, nTasks):
        losses.append(lossFun(W[:, [cTask]], XtX[cTask], XtY[cTask], YtY[cTask], pointsPerT[cTask]))
    loss = (1/nTasks) * np.sum(losses)
    return loss


def VARbatch(W):
    nTasks = W.shape[1]
    nDims = W.shape[0]

    var = (1/nTasks)*np.linalg.norm(W, 'fro')**2 - np.linalg.norm(np.mean(W, axis=1).reshape(nDims,1))**2
    return var


def gradLoss(data, W):
    nTasks = W.shape[1]
    nDims = W.shape[0]
    pointsPerT = data['pointsPerT']
    XtX = data['XtX']
    XtY = data['XtY']
    gradLoss = np.zeros((nDims, nTasks))
    for cTask in range(0, nTasks):
        gradLoss[:, [cTask]] = (2/pointsPerT[cTask])*(XtX[cTask].dot(W[:, [cTask]]) - XtY[cTask])
    gradLoss = (1/nTasks)*gradLoss
    return gradLoss


def gradVARbatch(W):
    nTasks = W.shape[1]
    nDims = W.shape[0]

    gradVAR = np.zeros((nDims, nTasks))
    for cTask in range(0, nTasks):
        gradVAR[:, [cTask]] = (2/nTasks)*(W[:, [cTask]] - np.mean(W, axis=1).reshape(nDims, 1))
    return gradVAR
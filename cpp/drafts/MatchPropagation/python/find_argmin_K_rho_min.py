import os
from pylab import *
import numpy

folders = ["graf", "wall"]
ext = [".dog", ".haraff", ".hesaff", ".mser"]

inlierThres = [ 1.5, 5 ]
indices = range(2, 7)
ell = 1
nReg = 1000
params = [(80, 0.5), (200,0.3)]

def getPrecRecallPath(dataset, img_index, N_K_type, ell, nReg, K, rhoMin,
                      inlierThres, ext):
    file = ( dataset + '/performance_hat_N_K/'
             + 'prec_recall_' + dataset
             + '_1_' + str(img_index)
             + '_' + N_K_type + '_'
             + '_sqEll_' + str(ell*ell)
             + '_nReg_ ' + str(nReg)
             + '_K_' + str(K) + '_rhoMin_' + str(rhoMin)
             + '_inlierThres_' + str(inlierThres)
             + ext + '.txt' )
    return file

def createLatexFriendlyPath(filePath):
    newFilePath = filePath.replace(".", "_")
    return newFilePath

def getNumber(file):
    number = 0;
    line = file.readline()
    if line.split()[2] != '-1.#IND':
        number = float(line.split()[2])
    return number

def readStat(file):
    tp = int(getNumber(file))
    fp = int(getNumber(file))
    tn = int(getNumber(file))
    fn = int(getNumber(file))
    prec = getNumber(file)
    recall = getNumber(file)
    tnr = getNumber(file)
    acc = getNumber(file)
    fmeas = getNumber(file)
    return [tp, fp, tn, fn, prec, recall, tnr, acc, fmeas]

def readStatFile(path):
    #print 'Reading ', path
    f = open(path, 'r')
    data = readStat(f)
    f.close()
    return data

def plotAndSaveData(stats, statType, params, inlierThres, fileName):
    fig, ax = plt.subplots(1)
    xticks = ['1-'+str(xx) for xx in range(2,7)]
    x = range(1,6)
    y = range(11)
    yticks = [float(i)/10. for i in y]
    ax.set_xticklabels(xticks)
    ax.xaxis.set_major_locator(MaxNLocator(len(x)))
    ax.set_yticklabels(yticks)
    #ax.yaxis.set_major_locator(MaxNLocator(len(y)))
    col = [ 'r', 'b', 'y', 'c' ]
    for i in range(2):
        K, rhoMin = params[i]
        # N_{K,\rho_0}
        leg = (statType+' with $\mathcal{N}_{'+str(K)+','+str(rhoMin)+'}$.')
        ax.plot(x, stats[i*2], label=leg,
                color=col[i*2], marker='o', ms=15, ls='-', lw=6)
        # \hat{N}_{K, \rho_0}
        leg = (statType + ' with $\widehat{\mathcal{N}}_{'+str(K)+','+str(rhoMin)+'}$.')
        ax.plot(x, stats[i*2+1], label=leg,
                color=col[i*2+1], marker='o', ms=15, ls='-', lw=6)
    # Shink current axis's height by 10% on the bottom
    box = ax.get_position()
    alpha = 0.2
    ax.set_ylabel(statType)
    ax.set_xlabel('Image Pair')
    ax.set_position([box.x0, box.y0 + box.height * alpha,
                     box.width, box.height * (1-alpha)])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125),
              fancybox=True, shadow=True, ncol=2)
    fig.savefig(fileName)


img_type = '.png'
for f in folders:
    for e in ext:
        for thres in inlierThres:
            allPrec = []
            allRecall = []
            for param in params:
                K, rhoMin = param
                stats_N_K = []
                stats_HatN_K = []
                for i in indices:
                    pathN_K = getPrecRecallPath(f, i, 'N_K', ell, nReg, K, rhoMin, thres, e)
                    dataN_K = readStatFile(pathN_K)
                    pathHatN_K = getPrecRecallPath(f, i, 'HatN_K', ell, nReg, K, rhoMin,
                                                   thres, e)
                    dataHatN_K = readStatFile(pathHatN_K)
                    stats_N_K.append(dataN_K)
                    stats_HatN_K.append(dataHatN_K)
                precN_K = []
                recallN_K = []
                precHatN_K = []
                recallHatN_K = []
                for i in range(len(stats_N_K)):
                    precN_K.append(stats_N_K[i][4])
                    precHatN_K.append(stats_HatN_K[i][4])
                    recallN_K.append(stats_N_K[i][5])
                    recallHatN_K.append(stats_HatN_K[i][5])
                allPrec.append(precN_K)
                allPrec.append(precHatN_K)
                allRecall.append(recallN_K)
                allRecall.append(recallHatN_K)

            precName = ('AnalyzeN_K/'+f+'_precision'
                        +'_thres_'+str(thres)+e)
            precName = createLatexFriendlyPath(precName) + img_type
            plotAndSaveData(allPrec, 'Precision rate', params, thres, precName)

            recallName = ('AnalyzeN_K/'+f+'_recall'
                          +'_thres_'+str(thres)+e)
            recallName = createLatexFriendlyPath(recallName) + img_type
            plotAndSaveData(allRecall, 'Recall rate', params, thres, recallName)

import os
from pylab import *
import numpy

folders = ["bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall"]
ext = [".dog", ".haraff", ".hesaff", ".mser"]

bounds = [ 0, 1.5, 5, 10, 20, 30, 40, 50, 100, 200 ]
ell = 1.2

def getFilePath(dataset, PfFolder, lb, ub, ell, ext):
    file = ( dataset + '/' + PfFolder + '/' + dataset
           + '_lb_' + str(lb) + '_ub_' + str(ub)
           + '_squaredEll_' + str(ell*ell) + ext + '.txt' )
    return file

def createLatexFriendlyPath(filePath):
    newFilePath = filePath.replace(".", "_")
    return newFilePath

def readStat(file):
    file.readline()
    line = file.readline()
    size = [ float(x) for x in line.split()[1:6] ]
    line = file.readline()
    min = [ float(x) for x in line.split()[1:6] ]
    line = file.readline()
    max = [ float(x) for x in line.split()[1:6] ]
    line = file.readline()
    mean = [ float(x) for x in line.split()[1:6] ]
    line = file.readline()
    median = [ float(x) for x in line.split()[1:6] ]
    line = file.readline()
    sigma = [ float(x) for x in line.split()[1:6] ]
    return [size, min, max, mean, median, sigma]

def readStatFile(path):
    print 'Reading ', filePath
    f = open(path, 'r')
    overlapStat = readStat(f)
    angleStat = readStat(f)
    f.close()
    return (array(overlapStat), array(angleStat))

def plotAndSaveResultsFromStat(stat, statType, fileName):
    print 'Plotting', statType, 'statistics'

    fig, ax = plt.subplots(1)
    x = ['1-'+str(xx) for xx in range(2,7)]
    ax.plot(x, stat[1,:], label='min '+ statType +' error',
            color='c', marker='o', ms=15, ls='-', lw=6)
    ax.plot(x, stat[2,:], label='max '+ statType +' error',
            color='b', marker='o', ms=15, ls='-', lw=6)
    ax.plot(x, stat[3,:], label='mean '+ statType +' error',
            color='m', marker='o', ms=15, ls='-', lw=6)
    ax.plot(x, stat[4,:], label='median '+ statType +' error',
            color='r', marker='o', ms=15, ls='-', lw=6)
    ax.plot(x, stat[5,:], label='std dev. of '+ statType +' error',
            color='g', marker='o', ms=15, ls='-', lw=6)
    # Shink current axis's height by 10% on the bottom
    box = ax.get_position()
    alpha = 0.2
    ax.set_position([box.x0, box.y0 + box.height * alpha,
                     box.width, box.height * (1-alpha)])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=2)
    fig.savefig(fileName)

def plotAndSaveMedianResultsFromStat(medianData, statType, bounds, fileName, col, legX, legY):
    print 'Plotting', statType, 'statistics'

    fig, ax = plt.subplots(1)
    xticks = ['1-'+str(xx) for xx in range(2,7)]
    x = range(1,6)
    ax.set_xticklabels(xticks)
    ax.xaxis.set_major_locator(MaxNLocator(len(x)))
    for i in range(4):
        leg = (statType + ', $|| \phi(\mathbf{x}) - \mathbf{y}||_2 \in '+
               '[' + str(bounds[i]) + ',\ ' + str(bounds[i+1]) + ']$')
        ax.plot(x, medianData[i], label=leg,
                color=col[i], marker='o', ms=15, ls='-', lw=6)
    # Shink current axis's height by 10% on the bottom
    box = ax.get_position()
    alpha = 0.2
    ax.set_ylabel(legY)
    ax.set_xlabel(legX)
    ax.set_position([box.x0, box.y0 + box.height * alpha,
                     box.width, box.height * (1-alpha)])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125),
              fancybox=True, shadow=True, ncol=2)
    fig.savefig(fileName)

def oldComputeMeds(allJacMeds):
    # Debugging.
    #print len(allJacMeds) # 8 datasets
    #print len(allJacMeds[0]) # 9 intervals [lb, ub]
    #print len(allJacMeds[0][0]) # 5 image pairs

    # maxMeds[image_pair]
    maxMeds = []
    for p in range(0,5):
        # $\max_\dataset \med(\dataset)$ pour la paire 1-(p+1):
        meds = []
        for d in range(len(folders)):
            meds.append(allJacMeds[d][0][p])
        #print meds
        maxMeds.append(max(meds))
    print maxMeds

    maxMeds2 = []
    for p in range(0,5):
        # $\max_\dataset \med(\dataset)$ pour la paire 1-(p+1):
        meds = []
        for d in range(len(folders)):
            meds.append(allJacMeds[d][1][p])
        #print meds
        maxMeds2.append(max(meds))
    print maxMeds2

    # minMeds[b][image_pair]
    minMeds = zeros((len(bounds)-1,5))
    for b in range(1,len(bounds)-1):
        for p in range(0,5):
            #print b, p
            # $\min_\dataset \med(\dataset)$ pour la paire 1-(p+2):
            meds = []
            for d in range(len(folders)):
                meds.append(allJacMeds[d][b][p])
            minMeds[b,p] = min(meds)

    minMeds[0,:] = maxMeds
    print minMeds
    allMeds = vstack((maxMeds, maxMeds2))
    allMeds = vstack((allMeds, minMeds[1:,:]))
    print allMeds
    return allMeds

def computeMeanMeds(allJacMeds):
    meanMeds = zeros((len(bounds)-1,5))
    for b in range(len(bounds)-1):
        for p in range(0,5):
            #print b, p
            # $\min_\dataset \med(\dataset)$ pour la paire 1-(p+2):
            meds = []
            for d in range(len(folders)):
                meds.append(allJacMeds[d][b][p])
            meanMeds[b,p] = sum(meds)/len(meds)
    print meanMeds
    return meanMeds

def plotAndSaveGlobMeds(maxMinMeds, statTypeName, fileName, col, legX, legY):
    fig, ax = plt.subplots(1)
    xticks = ['1-'+str(xx) for xx in range(2,7)]
    x = range(1,6)
    ax.set_xticklabels(xticks)
    ax.xaxis.set_major_locator(MaxNLocator(len(x)))
    for i in range(4):
        leg = (statTypeName +', $|| \phi(\mathbf{x}) - \mathbf{y}||_2 \in '+
               '[' + str(bounds[i]) + ',\ ' + str(bounds[i+1]) + ']$')
        ax.plot(x, maxMinMeds[i], label=leg,
                color=col[i], marker='o', ms=15, ls='-', lw=6)
    # Shink current axis's height by 10% on the bottom
    box = ax.get_position()
    alpha = 0.2
    ax.set_ylabel(legY)
    ax.set_xlabel(legX)
    ax.set_position([box.x0, box.y0 + box.height * alpha,
                     box.width, box.height * (1-alpha)])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.100),
              fancybox=True, shadow=True, ncol=2)
    fig.savefig(fileName)

################################################################################

#folders = ["ubc", "wall"]

#for folder in folders:
#    for e in ext:
#        for b in range(len(bounds)-1):
#            filePath = getFilePath(folder, 'P_f_approx',
#                                   bounds[b], bounds[b+1], ell, e)
#            overlapStat, angleStat = readStatFile(filePath)
#            newFilePath = createLatexFriendlyPath(filePath)
#            plotAndSaveResultsFromStat(overlapStat, 'Jaccard dist.',
#                                       newFilePath+'_overlap.eps')
#            plotAndSaveResultsFromStat(angleStat, 'angle diff.',
#                                       newFilePath+'_angle.eps')

for e in ext:
    allJacMeds = []
    allAngleMeds = []
    colJ = ['g', 'c', 'b', '#2100C2']
    colA = ['#ff00ff', '#ff0000', '#9B0000', '#520000']
    for folder in folders:
        overlapStats = []
        angleStats = []
        for b in range(len(bounds)-1):
            filePath = getFilePath(folder, 'P_f_approx',
                                   bounds[b], bounds[b+1], ell, e)
            overlapStat, angleStat = readStatFile(filePath)
            overlapStats.append(overlapStat)
            angleStats.append(angleStat)
        overlapMedian = []
        angleMedian = []
        # Get all medians.
        for i in range(len(overlapStats)):
            overlapMedian.append(overlapStats[i][4,:])
            angleMedian.append(angleStats[i][4,:])

        # Stats on Jaccard distance.
        fileJaccard = folder+e+"_jaccard"
        fileJaccard = createLatexFriendlyPath(fileJaccard)
        legY = ' $\widetilde{\mathcal{J}\ }(f,d,p,i)\ \in \ \ [0, 1]$ (Jaccard)'
        legX = 'image pair'
        plotAndSaveMedianResultsFromStat(overlapMedian, '$\widetilde{\mathcal{J}\ }(f,d,p,i)$', bounds,
                                         'analyzeForPf/'+fileJaccard+'.eps',
                                         colJ, legX, legY)
        fileAngle = folder+e+"_angle"
        fileAngle = createLatexFriendlyPath(fileAngle)
        legY = '$\widetilde{\mathcal{A}\ }(f,d,p,i)$ (Angle diff. in Degrees)'
        legX = 'image pair'
        plotAndSaveMedianResultsFromStat(angleMedian, '$\widetilde{\mathcal{A}\ }(f,d,p,i)$', bounds,
                                         'analyzeForPf/'+fileAngle+'.eps',
                                         colA, legX, legY)

        # allJacMeds[dataset][thres][pair]
        allJacMeds.append(overlapMedian)
        allAngleMeds.append(angleMedian)

    meanJacMeds = computeMeanMeds(allJacMeds)
    meanAngleMeds = computeMeanMeds(allAngleMeds)

    legY = '$\overline{\widetilde{\mathcal{J\ }}}(f,p,i)\ \in \ \ [0, 1]$ (Jaccard)'
    legX = 'image pair'
    fileJacPath = createLatexFriendlyPath("globMeds_jaccard"+e)
    plotAndSaveGlobMeds(meanJacMeds,
                        "$\overline{\widetilde{\mathcal{J\ }}}(f,p,i)$",
                        'analyzeForPf/'+fileJacPath+".eps",
                        colJ, legX, legY)

    legY = '$\overline{\widetilde{\mathcal{A\ }}}(f,p,i)$ (in Degrees)'
    legX = 'image pair'
    fileAnglePath = createLatexFriendlyPath("globMeds_angle"+e)
    plotAndSaveGlobMeds(meanAngleMeds,
                        "$\overline{\widetilde{\mathcal{A\ }}}(f,p,i)$",
                        'analyzeForPf/'+fileAnglePath+".eps",
                        colA, legX, legY)

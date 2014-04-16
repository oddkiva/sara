import os
from pylab import *
import numpy

folders = ["bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall"]
ext = [".dog", ".haraff", ".hesaff", ".mser"]
pairs = range(0,5)

bounds = [ 0, 1.5, 5, 10, 20, 30]
ell = 1.2

def getFilePath(dataset, qual_loc_aff_folder,
                p,
                ell, nReg, K, rhoMin,
                lb, ub, ext):
    file = ( dataset + '/' + qual_loc_aff_folder + '/' + dataset
             + '_1_' + str(p)
             + '_sqEll_' + str(ell*ell)
             + '_nReg_ ' + str(nReg)
             + '_K_' + str(K)
             + '_rhoMin_' + str(rhoMin)
             + '_lb_' + str(lb) + '_ub_' + str(ub)
             + ext + '.txt' )
    return file

def createLatexFriendlyPath(filePath):
    newFilePath = filePath.replace(".", "_")
    return newFilePath

def readStat(file):
    file.readline()
    line = file.readline()
    size = [ float(x) for x in line.split()[2:] ]
    line = file.readline()
    min = [ float(x) for x in line.split()[2:] ]
    line = file.readline()
    max = [ float(x) for x in line.split()[2:] ]
    line = file.readline()
    mean = [ float(x) for x in line.split()[2:] ]
    line = file.readline()
    median = [ float(x) for x in line.split()[2:] ]
    line = file.readline()
    sigma = [ float(x) for x in line.split()[2:] ]
    return [size, min, max, mean, median, sigma]

def readStatFile(path):
    print 'Reading ', filePath
    f = open(path, 'r')
    distStat = readStat(f)
    overlapStat = readStat(f)
    angleStat = readStat(f)
    f.close()
    return (array(distStat), array(overlapStat), array(angleStat))



################################################################################

out_ext = '.png'
#out_ext = '.eps'

nReg = 5000
K = 200
rhoMin = 0.3
ell = 1

ext = [ext[0]]

folders = [folders[0]]

dists = zeros((len(ext), len(folders), len(bounds), len(pairs)))
Jaccs = zeros((len(ext), len(folders), len(bounds), len(pairs)))
angles = zeros((len(ext), len(folders), len(bounds), len(pairs)))

for e in range(len(ext)):
    for f in range(len(folders)):
        for b in range(len(bounds)-1):
            for p in pairs:
                filePath = getFilePath(folders[f], 'Quality_Local_Aff', p+2,
                                       ell, nReg, K, rhoMin,
                                       bounds[b], bounds[b+1], ext[e])
                dist, Jaccard, angle = readStatFile(filePath)
##                print 'distance\n', dist
##                print 'Jaccard\n', Jaccard
##                print 'Angle\n', angle
                dists[e,f,b,p] = dist[4,0]
                Jaccs[e,f,b,p] = Jaccard[4,0]
                angles[e,f,b,p] = angle[4,0]

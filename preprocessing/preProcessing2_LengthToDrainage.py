import numpy as np
import os

"""
Preprocessing_2: Length to drainage

For a given drainage/stream network the distance to the next drainage cell will be calculated
Algorithm is based on the D8-Algorithm/Codification of flow direction. Its calucation steps incorporate
the search for flow paths and the recursive calculation of flow distances.

Required data (identical resolution & extent):
- Flow directions   (ESRI) ASCII Format
- Stream network    (ESRI) ASCII Format
Outptut:
- Flow length [m] to next stream in ASCII-Format

Special Features:
- Memory Mapping:   Enables processing of large datasets (>= 2GB for 32Bit)
                    (Fast hardware (SSD-hard drive) recommended, otherwise long computation time is to be expected)

Compiled by: Henning Oppel
Last edited: 27.03.2017

"""

#User specifications
cellsize = 100.                     # Cellsize of given data
fdir = 'in_flowdir.asc'             # Filename of flow direction data
streams = 'streams.asc'             # Filename of stream network file
ausgabe = 'dist2Stream.asc'         # Output filename
mmap = True                         # Enable Memory Mapping



"""
Part 1: Data preparation
"""

#Read Flow direction
print 'Reading Flowdirection...'
direc = []
header = []
f = open(fdir, 'r')
for z, line in enumerate(f.readlines()):
    if z < 6:
        header.append(line)
    else:
        direc.append([])
        hilf = line.split()
        for elem in hilf:
            direc[z].append(int(elem))
f.close()


#Create target array
print 'Creating Targetarray...'
if mmap == True:
    if os.path.isfile('temp') == False:
        target = np.memmap('temp', dtype = 'float32', mode = 'w+', shape = (len(direc), len(direc[0])))
    else:
        target = np.memmap('temp', dtype = 'float32', mode = 'c', shape = (len(direc), len(direc[0])))
else:
    target = np.zeros(fdir.shape)


#Identify streams
print 'Identifying Streams ...'
l = open(streams, 'r')
for z, line in enumerate(l.readlines()[6:]):
    hilf = line.split()
    for s, elem in enumerate(hilf):
        if float(elem) > -9999.:
            direc[z][s] = 'stream'
l.close()



"""
Part 2: Search flow path
"""

#Dictionary for D8-codes
options = {}
options[1] = [0,1]
options[2] = [1,1]
options[4] = [1,0]
options[8] = [1,-1]
options[16] = [0,-1]
options[32] = [-1, -1]
options[64] = [-1,0]
options[128] = [-1,1]


#Instance support array for path-way search
done = np.zeros((len(direc),len(direc[0])))

print 'Starting analysis...'
#Each point of the basin is a flow-path starting point
for z, line in enumerate(direc):
    for s, elem in enumerate(line):
        #Check if point is part of the basin, stream or already done
        if str(elem) == 'stream':
            target[z][s] = 0
        elif elem == -9999:
            target[z][s] = -9999
        else:
            if done[z][s] == 0.:
                ##Create support variables
                ac_z = z
                ac_s = s
                path = []
                l0 = 0
                ##Follow flow direction path until a stream is reached
                while str(direc[ac_z][ac_s]) != 'stream':
                    #Identify where the water is moving from actual position
                    next_z = ac_z + options[direc[ac_z][ac_s]][0]
                    next_s = ac_s + options[direc[ac_z][ac_s]][1]
                    
                    #Save direction and distance
                    path.append([ac_z,ac_s,np.sqrt(np.sum(np.power(options[direc[ac_z][ac_s]],2)))])
                    done[ac_z][ac_s] = 1.
                    
                    #Abort criteria
                    if next_z < 0 or next_s < 0:
                        #Reached (lower) edge of grid, add half cellsize
                        nl = path[-1][2] / 2.
                        path[-1][2] = nl
                        break
                    elif next_z == len(direc) or next_s == len(direc[0]):
                        #Reached (upper) edge of grid, add half cellsize
                        nl = path[-1][2] / 2.
                        path[-1][2] = nl
                        break
                    elif direc[next_z][next_s] == -9999:
                        #Reached outlet of basin, add half cellsize
                        nl = path[-1][2] / 2.
                        path[-1][2] = nl
                        break
                    
                    elif done[next_z][next_s] == 1.:
                        #Reached processed cell, take assigned flow length for summation
                        l0 = target[next_z][next_s]/cellsize
                        break


                    #If no criteria is true: Move to next cells
                    ac_z = next_z
                    ac_s = next_s
                    

                #Move upstream along identified flow path
                for cell in path[::-1]:
                    #Sum up flow length
                    l0 += cell[2]
                    #Multiply with cellsizes
                    target[cell[0]][cell[1]] = l0*cellsize
                    
                    
print 'writing results...'
out = open(ausgabe, 'w')
for line in header:
    print line
for line in target:
    outstring = ''
    for elem in line:
        outstring += str(elem) + '\t'
    outstring += '\n'
    out.write(outstring)
out.close()

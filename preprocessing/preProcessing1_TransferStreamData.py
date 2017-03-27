import numpy as np
import os

"""
preProcessing 1: Transfer Stream Data

Transfer an arbitrary (cell-wise) stream information to its drainage area
Algorithm is based on the D8-Algorithm/Codification of flow direction. Its calucation steps incorporate
the search for flow paths and the recursive calculation of flow distances.

Required data (identical resolution & extent):
- Flow directions   (ESRI) ASCII Format
- Stream network    (ESRI) ASCII Format
- Arbitrary Information     (ESRI) ASCII Format

User specifications:
- Filenames (see below) for required grids
- writeIDS -> boolean: (True) write IDs of streamcells to target raster or (False) given information

Erstellt von: Henning
Stand: 23.05.2016

"""
#User specification
fdir = 'in_flowdir.asc'         # Filename of flow direction grid
gew = 'resa_streams.asc'        # Filename of stream network grid
infos = 'in_strahler.asc'       # Filename of information grid
ausgabe = 'ac_strahler.asc'     # Filename of output file
writeIDS = False                # Write stream IDs or information
mmap = False                    # Enable memory mapping

"""
Part 1: Data preparation
"""

print 'Reading Flowdirection...'
direc = []
header= []
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

print 'Creating Targetarray...'
if mmap == True:
    if os.path.isfile('temp_schritt2') == False:
        target = np.memmap('temp_schritt2', dtype = 'float32', mode = 'w+', shape = (len(direc), len(direc[0])))
    else:
        target = np.memmap('temp_schritt2', dtype = 'float32', mode = 'c', shape = (len(direc), len(direc[0])))
else:
    target = np.zeros((len(direc), len(direc[0])))

streamIDs = {}
ID = 0

#Read stream network
gewaesser = np.loadtxt(gew, skiprows = 6)

print 'Identifying Streams ...'
l = open(infos, 'r')
for z, line in enumerate(l.readlines()[6:]):
    hilf = line.split()
    for s, elem in enumerate(hilf):
        if float(elem) > -9999. and gewaesser[z][s] > -9999.:
            direc[z][s] = 'stream'
            if writeIDS == True:
                streamIDs[(z,s)] = ID         
            else:
                streamIDs[(z,s)] = float(elem)
            ID += 1
l.close()

"""
Part 2: Find flow paths
"""
#Dictionary with D8-code
options = {}
options[1] = [0,1]
options[2] = [1,1]
options[4] = [1,0]
options[8] = [1,-1]
options[16] = [0,-1]
options[32] = [-1, -1]
options[64] = [-1,0]
options[128] = [-1,1]

#Instance assisting array
done = np.zeros((len(direc),len(direc[0])))

print 'Starting analysis...'
#Each point of the basin is a pathway start
for z, line in enumerate(direc):
    for s, elem in enumerate(line):
        #Check if point is a stream, outside of basin or already processed
        if str(elem) == 'stream':
            target[z][s] = streamIDs[(z,s)]
        elif elem == -9999:
            target[z][s] = -9999
        else:
            if done[z][s] == 0.:
                #Activate search
                #Support - local variables
                ac_z = z
                ac_s = s
                path = []
                l0 = 0
                ##Repeat until stream cell is reached
                while str(direc[ac_z][ac_s]) != 'stream':
                    #Where does the water moves from here
                    next_z = ac_z + options[direc[ac_z][ac_s]][0]
                    next_s = ac_s + options[direc[ac_z][ac_s]][1]
                    
                    #Save path and length to next cell
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
                        #l0 = target[next_z][next_s]/cellsize
                        break


                    #If no criteria is true: Move to next cells
                    ac_z = next_z
                    ac_s = next_s
                    

                #If reached cell is a stream
                if str(direc[ac_z][ac_s]) == 'stream':
                    #Move upstream along identified flow path
                    for cell in path[::-1]:
                        target[cell[0]][cell[1]] = streamIDs[(ac_z,ac_s)]
                elif next_z < len(direc) and next_s < len(direc[0]):
                    if done[next_z][next_s] == 1.:
                        for cell in path[::-1]:
                            target[cell[0]][cell[1]] = target[next_z][next_s]
                        
                    
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
            
            
            
            

            

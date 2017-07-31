# -*- coding: utf-8 -*-
"""
/***************************************************************************
 acs
                                 A QGIS plugin
 Ascertainment by Catchment Structure
                              -------------------
        begin                : 2017-06-27
        git sha              : $Format:%H$
        copyright            : (C) 2017 by Laura Bienstein
        email                : laura.bienstein@rub.de
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, QFileInfo
from PyQt4.QtGui import QAction, QIcon, QFileDialog

##from qgis.PyQt.QtCore import QCoreApplication, QLocale
##from qgis.PyQt.QtWidgets import QPushButton, QApplication
from qgis.core import QGis, QgsExpression, QgsMessageLog, qgsfunction, QgsMessageOutput, QgsRasterLayer
from qgis.gui import QgsMessageBar, QgisInterface
from qgis.utils import *


# Initialize Qt resources from file resources.py
import resources
import os, time, copy, sys
sys.path.append(r'C:\Program Files\QGIS 2.18\apps\Python27\lib\site-packages')
# Import the code for the dialog
from acs_subdivide_dialog import acsDialog
import os.path
import numpy as np
from scipy.cluster.vq import whiten, kmeans2
##if 'PROGRAMFILES(X86)' in os.environ:
##    import ACS_Win64 as ws2


class acs:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'acs_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = acsDialog()
        
        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&acs_subdivide')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'acs')
        self.toolbar.setObjectName(u'acs')
        
        self.dlg.lineEdit.clear()
        self.dlg.lineEdit_2.clear()
        self.dlg.lineEdit_3.clear()
        self.dlg.lineEdit_4.clear()
        self.dlg.lineEdit_5.clear()
        self.dlg.lineEdit_6.clear()
        self.dlg.lineEdit_7.clear()
        self.dlg.lineEdit_8.clear()
        self.dlg.lineEdit_9.clear()
        self.dlg.lineEdit_10.clear()
        self.dlg.lineEdit_11.clear()
        self.dlg.lineEdit_12.clear()
    
        

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('acs', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        # Create the dialog (after translation) and keep reference
        self.dlg = acsDialog()

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToRasterMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/acs/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'ACS Subdivide'),
            callback=self.run,
            parent=self.iface.mainWindow())


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginRasterMenu(
                self.tr(u'&acs_subdivide'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    def select_output(self):
        filename = QFileDialog.getSaveFileName(self.dlg, '','', '')
        self.dlg.lineEdit_11.setText(filename)
        return

    def select_input(self):
        name =  QFileDialog.getOpenFileName(self.dlg, '', '', '*.asc')
        self.dlg.lineEdit.setText(name)
        return

    def select_start(self):
        name =  QFileDialog.getOpenFileName(self.dlg, '', '', '*.txt')
        self.dlg.lineEdit_2.setText(name)
        return

    def select_strahler(self):
        name =  QFileDialog.getOpenFileName(self.dlg, '', '', '*.asc')
        self.dlg.lineEdit_3.setText(name)
        return

    def select_flowdir(self):
        name =  QFileDialog.getOpenFileName(self.dlg, '', '', '*.asc')
        self.dlg.lineEdit_4.setText(name)
        return

    def select_flowacc(self):
        name =  QFileDialog.getOpenFileName(self.dlg, '', '', '*.asc')
        self.dlg.lineEdit_5.setText(name)
        return

    def select_streamflow(self):
        name =  QFileDialog.getOpenFileName(self.dlg, '', '', '*.asc')
        self.dlg.lineEdit_6.setText(name)
        return

    def select_hillslope(self):
        name =  QFileDialog.getOpenFileName(self.dlg, '', '', '*.asc')
        self.dlg.lineEdit_7.setText(name)
        return

    def select_dem(self):
        name =  QFileDialog.getOpenFileName(self.dlg, '', '', '*.asc')
        self.dlg.lineEdit_12.setText(name)
        return
        

    def run(self):
        
        """Run method that performs all the real work"""
##        layers = self.iface.legendInterface().layers()
##        layer_list = []
##        for layer in layers:
##            layer_list.append(layer.name())
##        self.dlg.comboBox.addItems(layer_list)
        # show the dialog

        def allout():
            """Output function for GIS-Tool, gives SPS-File, basins and zones"""

            #1. Separation points as txt-file
            ##global sps, xll, yll, my, delta, kat, subbas, zonal
            f = open(self.dlg.lineEdit_11.text()+'.txt', 'w')
            f.write('SPS-Nr.\tX\tY\tCategory\teffect\n')
            for elem in sps:
                k = grid2coords([sps[elem]], xll, yll, my, delta)
                f.write(str(elem) + '\t' + str(k[0][0]) + '\t' + str(k[0][1]) + '\t' + str(kat[elem][0])+ '\t' + str(kat[elem][1]) + '\n')
            f.close()
            #2. Basins
            gridout(subbas, self.dlg.lineEdit_11.text()+'_result_basins')
            #3. Zones
            x = copy.deepcopy(zonal)
            gridout(x, self.dlg.lineEdit_11.text() + '_zones' )

            return 'done'


        def grid2coords(data, xll, yll, my, delta):
            """Calculates Coordinates from Gridpositions. Data = np.array[x,y]"""
            pc = []
            for point in data:
                x = xll + point[0]*delta + 0.5*delta
                y = yll + (my-point[1])*delta - 0.5*delta
                pc.append([x,y])

            return np.array(pc)


        def coords2grid(data, xll, yll, my, delta):
            """Calculates Gridposition for given Coordinates. Data = np.array[x,y]"""
            pc = []
            for point in data:
                x = np.rint((point[0]-xll-0.5*delta)/delta)
                y = np.rint(((yll+my*delta)-point[1]-.5*delta)/delta)
                pc.append([x,y])

            return np.array(pc)


        def structure(basin):
            """Structure Analysis for given basin, returns [OFL-Histogram, SFL-Histogram,
                xw-Histogram, yw-Histogram, acc-Histogram, sfl-list, x-list, y-list, acc-list"""

            ##global kennw, lsfl, oflen, xs, ys, accumulation, mask
            kw = np.ravel(kennw[np.equal(basin,1)])
            sfl = np.ravel(lsfl[np.equal(basin,1)])
            ofl = np.ravel(oflen[np.equal(basin,1)])
            x = np.ravel(xs[np.equal(basin,1)])
            y = np.ravel(ys[np.equal(basin,1)])
            accu = np.ravel(accumulation[np.equal(basin,1)])
            listmask = np.ravel(mask[np.equal(basin,1)])
            
            ##Reduce data to cells where data is available
            sfl = sfl[np.equal(listmask, 1)]
            ofl = ofl[np.equal(listmask, 1)]
            x = x[np.equal(listmask, 1)]
            y = y[np.equal(listmask, 1)]
            accu = accu[np.equal(listmask, 1)]
            kw = kw[np.equal(listmask, 1)]
            koo = np.transpose([x,y])
            
            ##Correction of stream-flow length array:
            if len(sfl) > 0.:
                sfl = np.subtract(sfl, min(sfl))
                #Histogram for SFL
                up = np.ceil(max(sfl)/dx) + 1
                bins = np.arange(0, up*dx, dx)
                ##Bins for OFL
                upy = np.ceil(np.percentile(ofl, 99)/dy) + 1
                ybins = np.arange(0, upy*dy, dy)
                if max(ofl) > (upy-1)*dy:
                    if max(ofl) > upy*dy:
                        ybins = np.append(ybins, max(ofl)+1)
                    elif max(ofl) == upy*dy:
                        ybins = np.append(ybins, upy*dy+1)
                    else:
                        ybins = np.append(ybins, upy*dy)

                
                #Define Data-arrays for grouping of data in distance stripes
                ##defined from min(dis_fl)-1 to max(dis_sfl)-1
                dis_sfl = np.digitize(sfl, bins)
                dis_ofl = np.digitize(ofl, ybins)
                ##Data to be grouped
                data = []
                xw = []
                yw = []
                acc = []
                #Edit
                ##Additional grouping for ofl
                odata = []

                ##EDIT: Exceptionhandling for small basins, comprising only one distance-stripe
                if len(bins) == 1:
                    last = 1
                else:
                    last = len(bins)-1

                for i in range(0,last):
                    data.append([])
                    xw.append([])
                    yw.append([])
                    acc.append([])
                    odata.append([])
                    for j in range(0,len(ybins)):
                        odata[i].append([])
                    
                ##Sort data by SFL-histogram
                for i,k,c,a,j in zip(dis_sfl,kw,koo,accu, dis_ofl):
                    data[i-1].append(k)
                    xw[i-1].append(c[0])
                    yw[i-1].append(c[1])
                    acc[i-1].append(a)
                    odata[i-1][j-1].append(k)

                odata = np.array(odata)

                return [odata, data, xw, yw, acc, sfl, accu, koo]
            else:
                raise ValueError('No Data available in given basin')

        def real_structure(basin):
            """FAcc Structure Analysis for given basin, returns [xw-Histogram, yw-Histogram,
            acc-Histogram, sfl-list, acc-list, x-y-list] """

            ##global kennw, lsfl, oflen, xs, ys, accumulation, mask
            sfl = np.ravel(lsfl[np.equal(basin,1)])
            x = np.ravel(xs[np.equal(basin,1)])
            y = np.ravel(ys[np.equal(basin,1)])
            accu = np.ravel(accumulation[np.equal(basin,1)])
            ##Create array for coordinates
            koo = np.transpose([x,y])
                    
            ##Correction of stream-flow length array:
            sfl = np.subtract(sfl, min(sfl))
            #Histogram for SFL
            up = np.ceil(max(sfl)/dx) + 1
            bins = np.arange(0, up*dx, dx)
            
            #Define Data-arrays for grouping of data in distance stripes
            ##defined from min(dis_fl)-1 to max(dis_sfl)-1
            dis_sfl = np.digitize(sfl, bins)
            ##Data to be grouped
            xw = []
            yw = []
            acc = []
            
            ##EDIT: Exceptionhandling for small basins, comprising only one distance-stripe
            if len(bins) == 1:
                last = 1
            else:
                last = len(bins)-1

            for i in range(0,last):
                xw.append([])
                yw.append([])
                acc.append([])
                
            ##Sort data by SFL-histogram
            for i,c,a in zip(dis_sfl,koo,accu):
                xw[i-1].append(c[0])
                yw[i-1].append(c[1])
                acc[i-1].append(a)
            
            return [xw, yw, acc, sfl, accu, koo]



        def searchSP(order, xyDic, posGrid, negGrid):
            """Searches the best SeprarationPoint in given Order(and xy-Coordinaten), returns
            SP and associated basin"""

            ##global direc
            
            deckung = []
            koords = {}
            deckung_vor = []
            target = np.array([[2,4,8],[1,0,16],[128,64,32]])
            basin = np.zeros(direc.shape)
            
            #Probing of the datafield (in case of more than 200 possible points, aim: reduce number of iterations)
            if len(order) > 200:
                ##5% of the datafiel is anaylsed
                breaks = np.rint(np.multiply(np.arange(0.05, 1.0, 0.05), len(order)))
                
                for b in np.unique(breaks):
                    p = order[int(b)]
                    ws = watershed([xyDic[p][0], xyDic[p][1]])
                    positive = np.sum(np.multiply(ws,posGrid))/np.sum(posGrid)
                    neg = np.sum(np.multiply(ws,negGrid))/np.sum(negGrid)

                    deckung_vor.append(positive-neg)

                ##evaluate
                intervalle = [[0, max(0.5*np.rint(np.mean(np.diff(breaks))),1)]]
                for b, elem in zip(breaks, deckung_vor):
                    if elem > 0.5*(max(deckung_vor)-min(deckung_vor)):
                        intervalle.append([b-0.5*np.rint(np.mean(np.diff(breaks))), b+0.5*np.rint(np.mean(np.diff(breaks)))])

                ##create new list
                nliste = []
                for r in intervalle:
                    for i in range(int(r[0]), int(r[1])):
                        nliste.append(liste[i])

            #If datafield is small use all points
            else:
                nliste = order
                
            #Iterate all possible drainage points and evaluate
            for p in nliste:
                ws = watershed([xyDic[p][0], xyDic[p][1]])
                
                #Test how many points of Targetarea are encompassed by basin
                positive = np.sum(np.multiply(ws,posGrid))/np.sum(posGrid)
                neg = np.sum(np.multiply(ws,negGrid))/np.sum(negGrid)
                
                #Save evaluation (coverage)
                deckung.append(positive-neg)
                koords[positive-neg] = xyDic[p]


            #Return basin/drainage with best coverage
            sps = koords[max(deckung)]
            bsps = watershed([koords[max(deckung)][0], koords[max(deckung)][1]])
            if len(order) > 200:
                return [sps, bsps, deckung, deckung_vor, breaks]
            else:
                return [sps, bsps]


        def OFLvariance(ofl_hist, mod = False):
            """Gives SFL-array with average variances on surface classes,
            input = Characteristic histogram (SFL_classes, ofl_classes)"""

            no = np.zeros((len(ofl_hist), len(ofl_hist[0])))
            wx = np.zeros((len(ofl_hist), len(ofl_hist[0])))
            ovar = np.zeros((len(ofl_hist), len(ofl_hist[0])))
            for i, line in enumerate(ofl_hist):
                for j in range(0,len(ofl_hist[i])):
                    no[i][j] = len(ofl_hist[i][j])
                    if no[i][j] > 0.:
                        ovar[i][j] = np.std(ofl_hist[i][j])
                    else:
                        ovar[i][j] == 0.
                if np.sum(no[i]) > 0.:
                    wx[i] = np.divide(no[i], np.sum(no[i]))
                
            if mod == False:
                aovar = np.nanmean(ovar, axis=1)
                return aovar
            else:
                eins = np.multiply(ovar, wx)
                aovar = np.nansum(eins, axis = 1)
                return [aovar, np.sum(no, axis = 1)]
                    

        def gridout(raster, name):
            """Writes an ASCII-Grid file for given 2D-Raster"""
            ##global xll, yll, delta
            
            f = open(name + '.asc', 'w')
            f.write('ncols         ' + str(len(raster[0])) + '\n')
            f.write('nrows         ' + str(len(raster)) + '\n')
            f.write('xllcorner     ' + str(xll) + '\n')
            f.write('yllcorner     ' + str(yll) + '\n')
            f.write('cellsize      ' + str(delta) + '\n')
            f.write('NODATA_value  -9999\n')

            if -9999. not in np.unique(raster):
                raster[np.equal(raster,0)] = -9999
            
            for line in raster:
                out = ''
                for elem in line:
                    out += str(elem) + ' '
                f.write(out + '\n')

            f.close()

            return 'done'

        def spsout():
            """Writes all new-found Separation Points to txt File"""
            ##global sps, xll, yll, my, delta, kat
            f = open(self.dlg.lineEdit_11.text()+'result_SeparationPoints.txt', 'w')
            f.write('SPS-Nr.\tX\tY\tCategory\teffect\n')
            for elem in sps:
                k = grid2coords([sps[elem]], xll, yll, my, delta)
                f.write(str(elem) + '\t' + str(k[0][0]) + '\t' + str(k[0][1]) + '\t' + str(kat[elem][0])+ '\t' + str(kat[elem][1]) + '\n')
            f.close()

            return 'done'

        def allbasins(zones = True):
            """ Writes all Basin-Arrays to a single ESRI-GRID"""
            ##global subbas, zonal
            basin = copy.deepcopy(subbas)
            z = copy.deepcopy(zonal)
            z[np.equal(z, 9999.)] = 9.
            if zones == True:
                basin = np.add(basin, np.divide(z, 10.))
            gridout(basin, self.dlg.lineEdit_11.text()+'result_basins')

            return 'done'

        def evaluate(basin, subbasins):
            """Routine to evaluate zonal classification"""    
            ##global kennw, lsfl, oflen, mask
            kw = np.ravel(kennw[np.equal(basin,1)])
            sfl = np.ravel(lsfl[np.equal(basin,1)])
            ofl = np.ravel(oflen[np.equal(basin,1)])
            listmask = np.ravel(mask[np.equal(basin,1)])
            s_lists = []
            for sub in subbasins:
                s_lists.append(np.ravel(sub[np.equal(basin, 1)]))
                
            ##Reduce data to cells where data is available
            sfl = sfl[np.equal(listmask, 1)]
            ofl = ofl[np.equal(listmask, 1)]
            kw = kw[np.equal(listmask, 1)]
            for i in range (0, len(subbasins)):
                s_lists[i] = s_lists[i][np.equal(listmask, 1)]

            sfl = np.subtract(sfl, min(sfl))
            up = np.ceil(max(sfl)/dx) + 1
            bins = np.arange(0, up*dx, dx)
            
            upy = np.ceil(np.percentile(ofl, 99)/dy) + 1
            ybins = np.arange(0, upy*dy, dy)
            if max(ofl) > (upy-1)*dy:
                if max(ofl) > upy*dy:
                    ybins = np.append(ybins, max(ofl)+1)
                elif max(ofl) == upy*dy:
                    ybins = np.append(ybins, upy*dy+1)
                else:
                    ybins = np.append(ybins, upy*dy)

            #Distance-class indicators for each value
            dis_sfl = np.digitize(sfl, bins)
            dis_ofl = np.digitize(ofl, ybins)

            #Which variance in considered?
            #-> Average Variance on hillslope surface

            #Build Histogram
            ods = []
            for sub in subbasins:
                ods.append([])
            ##EDIT: Exceptionhandling for small basins, comprising only one distance-stripe
            if len(bins) == 1:
                last = 1
            else:
                last = len(bins)-1
            #Create empy histogram
            for i in range(0,last):
                for x in range(0, len(subbasins)):
                    ods[x].append([])
                for j in range(0, len(ybins)):
                    for y in range(0, len(subbasins)):
                        ods[y][i].append([])

            ##Iterate parallel lists
            x = 0
            for sc, oc, k in zip(dis_sfl, dis_ofl, kw):
                #Test: value goes to which zone?
                for y, sub in enumerate(s_lists):
                    if sub[x] == 1.:
                        ods[y][sc-1][oc-1].append(k)
                x += 1
            
            #Now: Each zones/sub-basin has its own histogram
            #Calculate each aovar and SFL-member
            aos = []
            nos = []
            for i, sub in enumerate(subbasins):
                x = OFLvariance(ods[i], mod = True)
                aos.append(x[0])
                nos.append(x[1])

            #Calculate wheightend average
            aovar = np.zeros(aos[0].shape)
            nums = np.sum(np.array(nos), axis = 0)
            for i in range(0, len(aovar)):
                for j in range(0, len(subbasins)):
                    aovar[i] += aos[j][i] * (nos[j][i]/nums[i])

            return aovar


        def zone_helper(basin, i, o, h, mS):
            """Definition of "close to stream" and "far from stream zones"""
            #Reduce grids to examined sub-basin
            ##Reinitialise in each call
            acStr = np.multiply(strahler, basin)
            acOfl = np.multiply(oflen, basin)
            ##global ufer, hills
            #Select close to stream cells
            ##OFL <= o*dy = 1; Stahler >= mStr-i = 1

            acOfl[np.less_equal(acOfl, o*dy)] = 1
            acOfl[np.greater(acOfl, o*dy)] = 0
            acStr[np.less(acStr, mS-i)] = 0
            acStr[np.greater_equal(acStr, mS-i)] = 1
            ufer = np.multiply(acStr, acOfl)

            #create hillslope & plateau array
            plat = np.zeros(basin.shape)
            hills = copy.copy(basin)
            hills[np.equal(ufer,1)] = 0
            #If parameter h = 0 -> no plateau
            if h > 0:
                ##Clip heights to hillslope
                acDem = np.multiply(dem, hills)
                ##Threshold
                miH = np.min(acDem[np.equal(hills, 1)])
                maH = np.max(acDem[np.equal(hills, 1)])
                if miH != maH:
                    hx = (maH - miH) * (float(h)/float(hsteps)) + miH
                    plat[np.greater_equal(acDem, hx)] = 1
                else:
                    plat = np.zeros(basin.shape)
                #Set hillslope to 0 where plateau = 1
            hills[np.equal(plat, 1)] = 0

            ##global aovar
            #Calculate variance and number of cells in zones
            ##Edit: Only perform for reasonable partition
            if np.sum(hills)*np.sum(ufer)>0.:
                try:
                    od_hilf = evaluate(basin, [ufer, hills, plat])
                except:
                    od_hilf = np.zeros(aovar.shape)
                    for x in range(0, len(od_hilf)):
                        od_hilf[x] = np.inf
            else:
                #Else: Retrun infite variance
                od_hilf = np.zeros(aovar.shape)
                for x in range(0, len(od_hilf)):
                    od_hilf[x] = np.inf

            hills[np.equal(hills,1)] = 2.
            plat[np.equal(plat, 1)] = 3.

            return [ufer, hills, od_hilf, plat]


        def calc_zones(basin, aovar):
            """Calculates zones for given strahler {(Maxvalue mS) - i} and
            OFL-distance o within defined basin. Gives [wetlands, hillslopes, weighted Variance]"""
            
            ##global strahler, oflen, dem, dy
            #Iteration variables
            i = 0   #Strahler-order
            o = 1   #"Wetland" Width
            h = 0   #Plateau heights
            k = 0
            rep = False
            #Calculate limits for iteration (maximum distance/strahler order)
            acStr = np.multiply(strahler, basin)
            acOfl = np.multiply(oflen, basin)
            acDem = np.multiply(dem, basin)
            mS = np.max(acStr)
            mO = np.min([np.unique(np.rint(np.divide(acOfl, dy)))[-2], 5])

            #Storages
            liste = []
            einstellung = {}
               
            while rep == False:
                
                #Zonen bestimmen
                u = zone_helper(basin, i, o, h, mS)
                
                #Calculate reduction of variance
                leistung = np.sum(u[2])
                
                #Save result
                liste.append(leistung)
                einstellung[k] = [o, i, h]
                
                #Increase order
                i += 1
                k += 1
                #Test for next order
                if mS-i <= 1:
                    #If order > maximum order -> increase ofl
                    i = 0
                    o += 1
                    #Test for next plateu division -> increase heights
                    if o > mO:
                        o = 1
                        h += 1
                        #Abort criteria
                        if h == hsteps:
                            rep = True

            #Return best outcome of iteratio
            x = np.argmin(liste)
            i = einstellung[x][1]
            o = einstellung[x][0]
            h = einstellung[x][2]

            u = zone_helper(basin, i, o, h, mS)
            return u


        def watershed(pour_point):
  
            """Delineates the watershed of given point. Pour-Point = [x,y]"""

            points = [pour_point]
            basin = np.zeros(direc.shape)
            basin[int(pour_point[1])][int(pour_point[0])] = 1

            

            #D8-Basis
            target = np.array([[2,4,8],[1,0,16],[128,64,32]])

            #Zaehlvariable fuer Suche
            i = 0
            while i < len(points):
                x = points[i][0]
                y = points[i][1]

                #Nachbarpunkt ueberpruefen ob diese in Punkt enwaessern
                for dx in range(-1,2,1):
                    for dy in range(-1,2,1):
                        if y+dy <= my-1 and x+dx <= mx-1:
                            d = direc[int(y+dy)][int(x+dx)]
##                            message = '%d' %d
##                            QgsMessageLog.logMessage(message ,'Message')
                            if d == target[dy+1][dx+1]:
                                if basin[int(y+dy)][int(x+dx)] == 0:
                                    points.append([int(x+dx),int(y+dy)])
                                    basin[int(y+dy)][int(x+dx)] = 1
                i += 1
            return basin

        self.dlg.pushButton.clicked.connect(self.select_input)
        self.dlg.pushButton_2.clicked.connect(self.select_start)
        self.dlg.pushButton_3.clicked.connect(self.select_strahler)
        self.dlg.pushButton_4.clicked.connect(self.select_flowdir)
        self.dlg.pushButton_5.clicked.connect(self.select_flowacc)
        self.dlg.pushButton_6.clicked.connect(self.select_streamflow)
        self.dlg.pushButton_7.clicked.connect(self.select_hillslope)
        self.dlg.lineEdit_8.setText('4')
        self.dlg.lineEdit_9.setText('10000')
        self.dlg.lineEdit_10.setText('100')
        self.dlg.pushButton_8.clicked.connect(self.select_output)
        self.dlg.pushButton_9.clicked.connect(self.select_dem)
        
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed


        if result:
            e = 0.5
            thR = 0.02
            dyn = [0.15, 10]
            maxIter = 10
            sepTh = 0.05
            preff = 1

            message = 'Initialisation...'
            QgsMessageLog.logMessage(message ,'ACS Subdivide')

            direc = np.loadtxt(self.dlg.lineEdit_4.text(), skiprows = 6)
            dem = np.loadtxt(self.dlg.lineEdit_12.text(), skiprows = 6)
            accumulation = np.loadtxt(self.dlg.lineEdit_5.text(), skiprows = 6)
            lsfl = np.loadtxt(self.dlg.lineEdit_6.text(), skiprows = 6)
            oflen = np.loadtxt(self.dlg.lineEdit_7.text(), skiprows = 6)
            strahler = np.loadtxt(self.dlg.lineEdit_3.text(), skiprows = 6)
            kennw = np.loadtxt(self.dlg.lineEdit.text(), skiprows = 6)
            hsteps = float(self.dlg.lineEdit_8.text())
            dx = float(self.dlg.lineEdit_9.text())
            dy = float(self.dlg.lineEdit_10.text())

            #Mask for data intersection
            mask = np.zeros(direc.shape)
            mask[np.equal(mask,0)] = 1.
            mask[np.equal(direc,-9999.)] = 0.
            mask[np.equal(dem, -9999.)] = 0.
            mask[np.equal(accumulation,-9999.)] = 0.
            mask[np.equal(kennw,-9999.)] = 0.
            mask[np.equal(lsfl,-9999.)] = 0.
            mask[np.equal(oflen,-9999.)] = 0.
            #mask[np.equal(strahler,-9999.)] = 0.

            #Exclude missing or NoData
            direc[np.equal(direc, -9999.)] = 0.
            dem[np.equal(dem, -9999.)] = 0.
            accumulation[np.equal(accumulation, -9999.)] = 0.
            kennw = np.multiply(kennw, mask)
            lsfl[np.equal(lsfl, -9999.)] = 0.
            lsfl[np.equal(lsfl, -9999.)] = 0.
            strahler[np.equal(strahler, -9999.)] = 0.

            kf = open(self.dlg.lineEdit.text(), 'r')
            kw, koo = [], []
            i = 0
            for zkw in kf.readlines()[:5]:
                if i == 0:
                    hilf = zkw.split()
                    mx = float(hilf[1])
                elif i == 1:
                    hilf = zkw.split()
                    my = float(hilf[1])
                elif i == 2:
                    hilf = zkw.split()
                    xll = float(hilf[1])
                elif i == 3:
                    hilf = zkw.split()
                    yll = float(hilf[1])
                elif i == 4:
                    hilf = zkw.split()
                    delta = float(hilf[1])
                i += 1

            #Instance storage arrays
            sps = {}
            bsps = {}
            kat = {}
            bs = []
            c_sfl = copy.copy(lsfl)
            zonal = np.zeros(direc.shape)
            subbas = np.zeros(direc.shape)
            subbas[np.equal(subbas, 0.)] = -9999.
            target = np.array([[2,4,8],[1,0,16],[128,64,32]])

            
            f = open(self.dlg.lineEdit_2.text(), 'r')
            for i, line in enumerate(f.readlines()[1:]):
                hilf = line.split()
                sps[i] = coords2grid([[float(hilf[1]), float(hilf[2])]], xll, yll, my, delta)[0]
                basin = np.zeros(direc.shape)
                #neu
                basin = watershed([sps[i][0], sps[i][1]])
                subbas[np.equal(basin, 1)] = i
                #Alt
                bsps[i] = copy.deepcopy(basin)
                bs.append(c_sfl[int(sps[i][1])][int(sps[i][0])])
                kat[i] = [1, 1]
            f.close()

            ##Recalculate watersheds for multiple initial drainage points
            if len(sps) > 1:
                order = np.argsort(np.array(bs))
                temp = np.zeros(direc.shape)
                for s in order:
                    temp[np.equal(bsps[s], 1)] = s+1
                for s in sps:
                    temp2 = np.zeros(direc.shape)
                    temp2[np.equal(temp, s+1)] = 1
                    #Alt
                    bsps[s] = copy.copy(temp2)
                    #neu
                    subbas[np.equal(temp2, 1)] = s

                del temp2, temp

            ##Create x-y-Arrays
            xs = np.zeros(direc.shape)
            ys = np.zeros(direc.shape)
            xses = np.arange(0, mx, 1)
            for row, yline in enumerate(ys):
                ys[row][np.equal(yline, 0)] = row
                xs[row] = xses
                

            #Initialise booleans and position variables
            s = 0
            n = len(sps)
            unten = False
            done = False
            rep = False
            sep = False
            oben = False

            #Deactivate separation at major ramification in case of pre-partition
            if len(sps) > 1:
                deak_vs = True
            else:
                #Definition of Major-ramification threshold (1%)
                deak_vs = False
                mjR = np.max(accumulation)*thR

            #Clean up
            kf.close()
            del i, hilf, c_sfl, bsps

            message = 'Starting Separation...'
            QgsMessageLog.logMessage(message ,'ACS Subdivide')

            #Iterate over all drainage points on bucket list (filled during loop)
            while s < len(sps):

                bsps = np.zeros(direc.shape)
                bsps[np.equal(subbas, s)] = 1.

                #Analyse if basin[s] is too small to be analysed(AE/1km > dx)
                ktest = copy.copy(kennw)
                ktest[np.less(bsps, 1)] = 0.
                if np.divide(np.sum(bsps)*100.*100., np.power(1000,2)) <= float(dx)/1000. or np.sum(ktest) == 0.:
                    #Calculate zones for uniform result
                    if s == 0:
                        strc_mj = structure(bsps)
                        odata_mj = strc_mj[0]
                        aovar = OFLvariance(odata_mj, mod = True)[0]
                    if np.sum(bsps) > 0:
                        u = calc_zones(bsps, aovar)
                        #Save if variance is lowered (independent from threshold)
                        if np.sum(u[2]) < np.sum(aovar):
                            zonal = np.add(zonal, u[0])   #close to stream zones
                            zonal = np.add(zonal, u[1])   #far from stream zones
                            zonal = np.add(zonal, u[3])   #plateau zones
                        else:
                            zonal[np.equal(bsps,1)] = 9999   #none
                    s += 1
                    del ktest
                else:
                    del ktest
                    """Structure analysis"""
                    
                    #Calculate extended width-function (EWF)
                    strc = structure(bsps)

                    ##Read needed arrays
                    ##Histograms
                    odata = strc[0]
                    data = strc[1]
                    xw = strc[2]
                    yw = strc[3]
                    acc = strc[4]
                    ##Lists
                    sfl = strc[5]
                    accu = strc[6]
                    koo = strc[7]
                    ##Clear
                    del strc
                    
                    #On first call: Calculate Threshold omega
                    if done == False:
                        #Consider entire catchment
                        if len(sps)>1:
                            ae = []
                            for bae in sps:
                                basin = np.zeros(direc.shape)
                                b = watershed([sps[bae][0], sps[bae][1]])
                                ae.append(np.sum(b))
                            bs = np.argmax(np.array(ae))
                            strc_mj = structure(watershed([sps[bs][0], sps[bs][1]]))
                            odata_mj = strc_mj[0]
                            aovar = OFLvariance(odata_mj, mod = True)[0]

                            del basin, strc_mj, odata_mj, order, bs, ae, b
                        else:
                            aovar = OFLvariance(odata, mod = True)[0]

                        #Calculate threshold
                        w = np.power(np.divide(aovar-np.max(aovar),np.min(aovar)-np.max(aovar)),e)
                        th = np.divide(sum(np.multiply(aovar,w)),sum(w))
                        done = True
                        #print th        
                    """Separation"""
                    
                    #
                    # 0 - Pre-subdvisions at major ramifiacations
                    #
                    
                    #Only performed if activated and FAcc above mjR present in (sub-)basin
                    if deak_vs == False and len(accu[np.greater(accu, mjR)]) > 0:

                            #Independent from variance requirements a subdivison at major ramifications
                            #will be performed.    
                            strc = real_structure(bsps)
                            r_koo = strc[5]
                            r_accu = strc[4]
                            r_sfl = strc[3]
                            del strc

                            #Omit all values below mjR
                            rac = np.array(r_accu)[np.greater(r_accu, mjR)]
                            rsfl = np.array(r_sfl)[np.greater(r_accu, mjR)]
                            ##Retain coordinates
                            rK = np.array(r_koo)[np.greater(r_accu, mjR)]
                            
                            #Find Gaps
                            ##Sort FAcc by SFL
                            sac = rac[np.argsort(rsfl)]
                            srK = rK[np.argsort(rsfl)]
                            ##Difference series
                            dsac = np.diff(sac)
                            dsac = np.insert(dsac, 0, -1)

                            #Subdivision after ramification (befor confluence) possible?
                            if len(np.argwhere(dsac>0)) > 0:

                                #Take first point with positive difference
                                bp1 = np.argwhere(dsac>0)[0][0]  #(Separationpoint 1)
                                bp2 = np.argwhere(dsac == min(dsac[:bp1]))[0][0] #(Separationpoint 2)
                                
                                ##Bp1 move if additional points between bp1 and bp2
                                if bp1-bp2 > 1:
                                    bp1 = bp2 + 1

                                ##Calculate watersheds
                                target = np.array([[2,4,8],[1,0,16],[128,64,32]])
                                basin = np.zeros(direc.shape)
                                w1 = watershed([srK[bp1][0], srK[bp1][1]])
                                w2 = watershed([srK[bp2][0], srK[bp2][1]])
                                ###New basin has to be smaller than initial basin
                                w1[np.equal(bsps, 0)] = 0
                                w2[np.equal(bsps, 0)] = 0

                                #Save results (if new drainge points)
                                check = False
                                for c in sps:
                                    if np.sum(srK[bp1] == sps[c]) > 0. or np.sum(srK[bp2] == sps[c]) > 0.:
                                        check = True
                                        sep = False
                                if check == False:
                                    ##Reduce initial basin
                                    bsps[np.equal(w1,1)] = 0
                                    bsps[np.equal(w2,1)] = 0
                                    ##Save new basins (to bucket list)
                                    sps[n] = srK[bp1]
                                    subbas[np.equal(w1, 1)] = n
                                    n += 1
                                    sps[n] = srK[bp2]
                                    subbas[np.equal(w2, 1)] = n
                                    n += 1
                                    kat[n-2] = [0,1]
                                    kat[n-1] = [0,1]

                                    #Restart separation routine
                                    sep = True

                                #Clean up
                                del check, w1, w2, bp1, bp2, dsac, srK, sac, rK, rsfl, rac
                            else:
                                #Clean up and proceed
                                sep = False
                            del r_accu, r_koo, r_sfl                        


                    #
                    # 1 - Calculate variance and objective function
                    #
                    if sep == False:

                        #Call objective function
                        aovar = OFLvariance(odata, mod = True)[0]
                        
                        #Difference of cumulative  standard deviation and threshold, normed for threshold
                        cth = np.arange(round(th,4), round(len(aovar)*th,4), round(th,4)) #kumulative wichtung
                        ##Edit: Add dynmicly reducing theshold. Breaking points are moved from the beginning
                        dynTh = np.zeros(aovar.shape)
                        dynTh[np.equal(dynTh, 0.)] = th
                        dynTh[:dyn[1]] += np.arange(dyn[0]*th, 0, -(dyn[0]/float(dyn[1]))*th)[:min(dyn[1], len(dynTh))]
                        
                        #Actual objective function
                        ##Test: Stripes with standarddeviation greater than threshold?
                        greater = np.greater_equal(aovar, dynTh)


                        #
                        #   2.1   -   No partition required
                        #
                
                        if sum(greater) <= 1:
                            #(Nearly) No class above the threshold
                            #Calculate zones for consistent result
                            u = calc_zones(bsps, aovar)
                            #Save if variance is lowered
                            if np.sum(u[2]) < np.sum(aovar):
                                zonal = np.add(zonal, u[0])   #Close to stream                        
                                zonal = np.add(zonal, u[1])   #Far from stream
                                zonal = np.add(zonal, u[3])   #plateau zones
                            else:
                                zonal[np.equal(bsps,1)] = 9999   #nothing

                            s += 1

                        #
                        #   2.2   -   Clip "good" regions
                        #

                        elif sum(greater==False) >= 3:
                            #Only few stripes above threshold

                            #
                            #   2.2.0 -   Preparation
                            #
                            
                            #Get indexes of all "good" stripes
                            zones = []
                            for i, val in enumerate(aovar):
                                if val in aovar[greater==False]:
                                    zones.append(i)
                            ##Indentify coherent regions
                            bereiche = {}
                            laenge = []
                            start = zones[0]
                            vor = start
                            for val in zones[1:]:
                                if val == vor + 1:
                                    #Values are coherent
                                    vor = val
                                else:
                                    #End of region
                                    laenge.append(vor-start)
                                    if val-start not in bereiche:
                                        bereiche[vor-start] = [start, vor]
                                    start = val
                                    vor = val
                            laenge.append(val-start)
                            if val-start not in bereiche:
                                bereiche[val-start] = [start, val]
                            
                            #Each call will clip a single region. In case of multiply incoherent "good"
                            #regions remaining regions will be clipped in subsequent re-analysis
                            ##Select biggest region
                            
                            b = bereiche[max(laenge)]
                            #Clean up
                            del start, val, zones, laenge, bereiche
                            
                            alle = np.arange(0, len(aovar))
                            #Identify target area
                            yepX = np.array(xw)[np.arange(b[0], b[1]+1)]
                            yepY = np.array(yw)[np.arange(b[0], b[1]+1)]
                            yepA = np.array(acc)[np.arange(b[0], b[1]+1)]

                            ##Non-Target area (further differentiation after if-statement)
                            ne = np.delete(alle, np.arange(b[0], b[1]+1))

                            ##Arrange as grids
                            yep = np.zeros(direc.shape)
                            nope = np.zeros(direc.shape)
                            for xses, yses in zip(yepX, yepY):
                                for x, y in zip(xses, yses):
                                    yep[int(y)][int(x)] = 1

                            #Remember the number iteration started!
                            nv = copy.deepcopy(n) 

                            #
                            #   2.2.1   Search downstream drainage
                            #

                            if b[1] < len(aovar)-1:
                                #Not applicable for last distance-strip (Since no downstream)
                                
                                #Remove all upstream cells
                                if b[0] > 0:
                                    ne = np.delete(ne, np.arange(0, b[0]))
                                noX = np.array(xw)[ne]
                                noY = np.array(yw)[ne]
                                noA = np.array(acc)[ne]

                                for xses, yses, acses in zip(noX, noY, noA):
                                    for x, y in zip(xses, yses):
                                        nope[int(y)][int(x)] = 1

                                ##While-loop if multiple drainage points are required
                                unten = False
                                j = 0
                                dvor = 0
                                
                                while unten == False:

                                    #Max FAcc in Non-Target Area
                                    m = max(noA[0])
                                    
                                    #Define possible separation points
                                    ##in non-target area
                                    pout_x = np.array(noX[0])[np.greater_equal(noA[0],m*0.95)]
                                    pout_y = np.array(noY[0])[np.greater_equal(noA[0],m*0.95)]
                                    ##in target area
                                    pout_x = np.append(pout_x, np.array(yepX[-1])[np.greater_equal(yepA[-1],m)])
                                    pout_y = np.append(pout_y, np.array(yepY[-1])[np.greater_equal(yepA[-1],m)])

                                    #Assemble order list with appendend Coordinates
                                    dic = {}
                                    liste = []
                                    for i, c in enumerate(np.transpose([pout_x, pout_y])):
                                        dic[i] = c
                                        liste.append(i)
                                    
                                    #Call iterative search for separation point
                                    sp = searchSP(liste, dic, nope, yep)

                                    #Clip result to active basin
                                    sp[1][np.equal(bsps, 0)] = 0
                                    #Test if search was successful, only applicable from 2.iteration
                                    if j > 0:
                                        if np.sum(sp[0] == sps[n-1]) < 2:
                                            ##Save results
                                            sps[n] = sp[0]
                                            sp[1][np.equal(bsps, 0.)] = 0.
                                            subbas[np.equal(sp[1], 1.)] = n
                                            kat[n] = [1, 1]
                                            n += 1
                                        else:
                                            unten = True
                                    else:
                                        ##Save results
                                        sps[n] = sp[0]
                                        subbas[np.equal(sp[1], 1.)] = n
                                        kat[n] = [1, 1]
                                        n += 1

                                        ##Remove new found sub-basin from active watershed
                                        bsps[np.equal(sp[1],1)] = 0.
                                        

                                    ##Calculate coverage
                                    deckung = np.sum(np.multiply(bsps, yep))/np.sum(yep)

                                    if round(deckung,5) == round(dvor,5):
                                        unten = True
                                    dvor = deckung
                                    if deckung >= 0.80 or j == maxIter:
                                        unten = True
                                    j += 1

                                del pout_x, pout_y, deckung, dic, noX, noY, noA, nope, dvor

                            else:
                                unten = False

                            #
                            #   2.2.2   Search upstream drainage
                            #
                            
                            #Refill array with all sub-basins
                            ne = np.delete(alle, np.arange(b[0], b[1]+1))
                           
                            #Search fuer upstream
                            if b[0] > 0:
                                #Only applicable for stripes at least one step away from outlet
                                
                                ##Remove downstream regions
                                if b[1] < len(aovar)-1:
                                    ne = np.delete(ne, np.arange(b[1]+1, len(aovar)))
                                noX = np.array(xw)[ne]
                                noY = np.array(yw)[ne]
                                noA = np.array(acc)[ne]

                                nope = np.zeros(direc.shape)
                                for xses, yses, acses in zip(noX, noY, noA):
                                    for x, y in zip(xses, yses):
                                        nope[int(y)][int(x)] = 1
                                
                                ##While-loop for multiple separation points
                                oben = False
                                j = 0
                                dvor = 0
                                #Create assisting array for SP-search
                                hilf = np.zeros(direc.shape)
                                target = copy.deepcopy(yep)
                                target[np.equal(bsps, 0.)] = 0.
                                while oben == False:
                                    
                                    #Max FAcc in non-target Area
                                    m = max(yepA[-1])
                                    
                                    #Identify possible separation points
                                    ##in target area
                                    pout_x = np.array(yepX[0])[np.greater_equal(yepA[0],m*0.95)]
                                    pout_y = np.array(yepY[0])[np.greater_equal(yepA[0],m*0.95)]

                                    #Assemble order list with appendend Coordinates
                                    dic = {}
                                    liste = []
                                    for i, c in enumerate(np.transpose([pout_x, pout_y])):
                                        dic[i] = c
                                        liste.append(i)

                                    #Call SPs-search routine
                                    sp = searchSP(liste, dic, target, nope)
                                                           
                                    #Test if search was successful, only applicable from 2.iteration
                                    if j > 0:
                                        if np.sum(sp[0] == sps[n-1]) < 2:
                                            ##Save results
                                            sps[n] = sp[0]
                                            sp[1][np.equal(bsps, 0.)] = 0.
                                            subbas[np.equal(sp[1], 1.)] = n
                                            kat[n] = [1,1]
                                            hilf[np.equal(sp[1],1)] = 1
                                            bsps[np.equal(sp[1],1)] = 0
                                            target[np.equal(sp[1],1)] = 0
                                            n += 1   
                                        else:
                                            oben = True
                                    else:
                                        ##Save results
                                        sps[n] = sp[0]
                                        sp[1][np.equal(bsps, 0.)] = 0.
                                        subbas[np.equal(sp[1], 1.)] = n
                                        kat[n] = [1, 1]
                                        n += 1
                                        ##Add clipped sub-basin to assisting array
                                        hilf[np.equal(sp[1],1)] = 1
                                        ##shorten existing array
                                        bsps[np.equal(sp[1],1)] = 0
                                        ##repeat for target area
                                        target[np.equal(sp[1],1)] = 0                              

                                    ##Calculate coverage
                                    deckung = np.sum(np.multiply(hilf, yep))/np.sum(yep)

                                    if round(deckung,5) == round(dvor,5):
                                        oben = True

                                    dvor = deckung
                                    if deckung >= 0.80 or j == maxIter:
                                        oben = True
                                    
                                    j += 1

                                #Clean up
                                del target, hilf, deckung, oben, dic, pout_x, pout_y
                                del noX, noY, noA, yepX, yepY, yepA, m, nope, yep, dvor

                                if s == 0:
                                    oben = False
                                
                        else:

                            #
                            #   2.3  - Subdivison of "bad" regions
                            #

                            i = 0
                            rep = False
                            while rep == False and i < maxIter:

                                #
                                #   2.3.1  - Subdivision at ramification
                                #

                                #Edit: Use all FAcc values
                                strc = real_structure(bsps)
                                r_acc = strc[2]
                                r_sfl = strc[3]
                                r_accu = strc[4]
                                r_koo = strc[5]
                                
                                #First try: Identify ramifiation
                                
                                wac = whiten(r_acc[0])
                                cla = kmeans2(wac, np.array([min(wac), max(wac)]))
                                aTh = max(np.array(r_acc[0])[np.equal(cla[1], 0)])*(1.-float(i/10.))

                                ##Omit all data below aTh
                                rac = np.array(r_accu)[np.greater(r_accu, aTh)]
                                rsfl = np.array(r_sfl)[np.greater(r_accu, aTh)]
                                ##Retain coordinates
                                rK = np.array(r_koo)[np.greater(r_accu, aTh)]
                                
                                del wac, cla, aTh, strc, r_acc, r_sfl, r_accu, r_koo
                                
                                #Identify gap
                                ##Sort FAcc by SFL
                                sac = rac[np.argsort(rsfl)]
                                srK = rK[np.argsort(rsfl)]
                                ##Compile difference series
                                dsac = np.diff(sac)
                                dsac = np.insert(dsac, 0, -1)

                                #Test if subdivision after ramification is possible
                                if len(np.argwhere(dsac>0)) > 0:
                        
                                    #First SP at first positive difference
                                    bp1 = np.argwhere(dsac>0)[0][0]  #(SP 1)
                                    bp2 = np.argwhere(dsac == min(dsac[:bp1]))[0][0] #(SP 2)
                                    
                                    ##Move bp1 is additional points between bp1 and bp2
                                    if bp1-bp2 > 1:
                                        bp1 = bp2 + 1

                                    
                                    ##Watersheds
                                    target = np.array([[2,4,8],[1,0,16],[128,64,32]])
                                    basin = np.zeros(direc.shape)
                                    w1 = watershed([srK[bp1][0], srK[bp1][1]])
                                    w2 = watershed([srK[bp2][0], srK[bp2][1]])
                                    
                                    ###Reduce watershed to active watershed
                                    w1[np.equal(bsps, 0)] = 0
                                    w2[np.equal(bsps, 0)] = 0

                                    #Test if subdivision is reasonable
                                    hilf = copy.copy(bsps)
                                    hilf[np.equal(w1,1)] = 0
                                    hilf[np.equal(w2,1)] = 0
                                    ##Calculate new sub-basin structure (EWF)
                                    od_hilf = evaluate(bsps, [hilf, w1, w2])
                                    obj_fac = np.sum(od_hilf)/np.sum(aovar)
                                    if obj_fac ==0. or np.isnan(obj_fac)==True:
                                        obj_fac = np.inf
                                    rep = True
                                    
                                #If subdivision is impossible
                                else:
                                    #Try 2: Increase ramification order
                                    i += 1
                                    if i == maxIter:
                                        obj_fac = np.inf
                                        rep = False

                            #Clean up
                            del od_hilf, hilf, target, basin, dsac, sac


                            #
                            #   2.3.2  - Zonal classification
                            #


                            u = calc_zones(bsps, aovar)
                            obj_zone = np.sum(u[2])/np.sum(aovar)


                            #
                            #   2.3.3  - Selection
                            #

                            #Afer performing both techniques decide which result is saved

                            # Alternative A:
                            # FAcc would have been discarded anymay
                            if obj_fac > 1-sepTh:
                                if obj_zone < 1.:
                                    zonal = np.add(zonal, u[0])   #close to stream                        
                                    zonal = np.add(zonal, u[1])   #far from stream
                                    zonal = np.add(zonal, u[3])   #plateau zones
                                else:
                                    zonal[np.equal(bsps,1)] = 9999   #none
                                s += 1

                            # Alternative B:
                            # Zones are significantly(preff) better than FAcc
                            elif obj_zone < preff*obj_fac:
                                zonal = np.add(zonal, u[0])   #close to stream                      
                                zonal = np.add(zonal, u[1])   #far from stream
                                zonal = np.add(zonal, u[3])   #plateau zones
                                s += 1

                            # Alternative C:
                            # FAcc (> sepTh) if superior to zones
                            else:
                                #Test: FAcc-Watersheds intersecting?
                                snip = np.add(w1, w2)
                                if float(np.sum(np.equal(snip,2.)))/float(max([np.sum(w1), np.sum(w2)])) > 0.1:
                                    if np.sum(w1) > np.sum(w2):
                                        bsps[np.equal(w1,1)] = 0
                                        sps[n] = srK[bp1]
                                        subbas[np.equal(w1, 1.)] = n
                                    else:
                                        bsps[np.equal(w2,1)] = 0
                                        sps[n] = srK[bp2]
                                        subbas[np.equal(w2, 1.)] = n
                                    n += 1
                                    kat[n-1] = [2, obj_fac]
                                else:
                                    #Save results
                                    ##Clip from active watershed
                                    bsps[np.equal(w1,1)] = 0
                                    bsps[np.equal(w2,1)] = 0
                                    #Save (add to bucket list)
                                    sps[n] = srK[bp1]
                                    subbas[np.equal(w1, 1.)] = n
                                    n += 1
                                    sps[n] = srK[bp2]
                                    subbas[np.equal(w2, 1.)] = n
                                    n += 1
                                    if i == 0:
                                        kat[n-2] = [1, obj_fac]
                                        kat[n-1] = [1, obj_fac]
                                    else:
                                        kat[n-1] = [2, obj_fac]
                                        kat[n-2] = [2, obj_fac]

            allout()

            iface.addRasterLayer(self.dlg.lineEdit_11.text()+'_result_basins.asc', os.path.basename(self.dlg.lineEdit_11.text())+'_result_basins.asc')
            iface.addRasterLayer(self.dlg.lineEdit_11.text() + '_zones.asc', os.path.basename(self.dlg.lineEdit_11.text())+'_zones.asc')

            message = 'Finished'
            QgsMessageLog.logMessage(message ,'ACS Subdivide')
           
            
            pass
        

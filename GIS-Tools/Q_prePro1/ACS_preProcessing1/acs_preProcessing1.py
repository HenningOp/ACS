# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ACS_preProcessing1
                                 A QGIS plugin
 Transfer stream data
                              -------------------
        begin                : 2017-07-26
        git sha              : $Format:%H$
        copyright            : (C) 2017 by Laura
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

from qgis.core import QGis, QgsExpression, QgsMessageLog, qgsfunction, QgsMessageOutput, QgsRasterLayer
from qgis.gui import QgsMessageBar, QgisInterface
from qgis.utils import *

import numpy as np
import os
# Initialize Qt resources from file resources.py
import resources
# Import the code for the dialog
from acs_preProcessing1_dialog import ACS_preProcessing1Dialog
import os.path


class ACS_preProcessing1:
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
            'ACS_preProcessing1_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = ACS_preProcessing1Dialog()
        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&acs_subdivide')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'ACS_preProcessing1')
        self.toolbar.setObjectName(u'ACS_preProcessing1')

        self.dlg.lineEdit.clear()
        self.dlg.lineEdit_2.clear()
        self.dlg.lineEdit_3.clear()
        self.dlg.lineEdit_4.clear()

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
        return QCoreApplication.translate('ACS_preProcessing1', message)


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
        self.dlg = ACS_preProcessing1Dialog()

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

        icon_path = ':/plugins/ACS_preProcessing1/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'ACS preProcessing 1'),
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

    def select_input(self):
        name = QFileDialog.getOpenFileName(self.dlg, '', '', '*.asc')
        self.dlg.lineEdit.setText(name)

    def select_flowDir(self):
        name = QFileDialog.getOpenFileName(self.dlg, '', '', '*.asc')
        self.dlg.lineEdit_3.setText(name)

    def select_streamNet(self):
        name = QFileDialog.getOpenFileName(self.dlg, '', '', '*.asc')
        self.dlg.lineEdit_2.setText(name)

    def select_output(self):
        filename = QFileDialog.getSaveFileName(self.dlg, '','', '')
        self.dlg.lineEdit_4.setText(filename)

    def run(self):
        """Run method that performs all the real work"""
        self.dlg.pushButton.clicked.connect(self.select_input)
        self.dlg.pushButton_2.clicked.connect(self.select_streamNet)
        self.dlg.pushButton_3.clicked.connect(self.select_flowDir)
        self.dlg.pushButton_4.clicked.connect(self.select_output)
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:

            k = open(self.dlg.lineEdit_2.text(), "r")
            a = k.readline()
            b = k.readline()
            c = k.readline()
            d = k.readline()
            e = k.readline()
            g = k.readline()
            k.close()
            hilf = e.split()
            cellsize = float(hilf[1])
            
            """
            Block 1: Daten vorbereitung
            """

            #Einlesen des Fliessrichtungsrasters
            message = 'Reading Flowdirection...'
            QgsMessageLog.logMessage(message ,'Message')
            direc = []
            f = open(self.dlg.lineEdit_3.text(), 'r')
            for z, line in enumerate(f.readlines()[6:]):
                direc.append([])
                hilf = line.split()
                for elem in hilf:
                    direc[z].append(int(elem))
            f.close()

            #Zielaaray erstellen
            ##  Wird als memory mapping Objekt erstellt um speicher zu sparen
            message = 'Creating Targetarray...'
            QgsMessageLog.logMessage(message ,'Message')
            if os.path.isfile(os.path.dirname(__file__)+'//temp_schritt2') == False:
                filename = os.path.dirname(self.dlg.lineEdit_4.text())+'//temp_schritt2'
                target = np.memmap(filename, dtype = 'float32', mode = 'w+', shape = (len(direc), len(direc[0])))
            else:
                target = np.memmap(os.path.dirname(__file__)+'//temp_schritt2', dtype = 'float32', mode = 'c', shape = (len(direc), len(direc[0])))


            streamIDs = {}
            ID = 0

            #Indentifizieren an welchen Stellen ein Gewässer ist
            gewaesser = np.loadtxt(self.dlg.lineEdit_2.text(), skiprows = 6)


            message = 'Identifying Streams ...'
            QgsMessageLog.logMessage(message ,'Message')
            l = open(self.dlg.lineEdit.text(), 'r')
            for z, line in enumerate(l.readlines()[6:]):
                hilf = line.split()
                for s, elem in enumerate(hilf):
                    if float(elem) > -9999. and gewaesser[z][s] > -9999.:
                        #Da Strecke im Gewässer nicht weiter interresant, an diese Stelle einen String schreiben
                        direc[z][s] = 'stream'
                        ##EXCHANGE
            ##            streamIDs[(z,s)] = ID         ##Exchange: ID
                        streamIDs[(z,s)] = float(elem)  ##Exchange: Info
                        ID += 1
            l.close()

            """
            Block 2: Fliesspfade finden
            """
            #Dictionary mit Optionen des Richtungsarrays anlegen
            options = {}
            options[1] = [0,1]
            options[2] = [1,1]
            options[4] = [1,0]
            options[8] = [1,-1]
            options[16] = [0,-1]
            options[32] = [-1, -1]
            options[64] = [-1,0]
            options[128] = [-1,1]


            #Suchalgorithmus wie immer -> nur auswertung anders


            done = np.zeros((len(direc),len(direc[0])))
            # done (Boolean Raster, in dem vermerkt wird ob eine Zelle bereits bearbeitet wurde oder nicht
            # base (Double Raster, Rasterwert der Fließlaenge des zugeordneten Streams)

            message = 'Starting analysis...'
            QgsMessageLog.logMessage(message ,'Message')
            #Jeder Puntk des Directionrasters ist ein startpunkt.
            for z, line in enumerate(direc):
                for s, elem in enumerate(line):
                    #Check, ob es sich um einen Fehlwert handelt, oder er schon gefunden wurde
                    if str(elem) == 'stream':
                        #Gewässer = NoData in Zielraster
            ##            target[z][s] = -9999
                        target[z][s] = streamIDs[(z,s)]
                       
                    elif elem == -9999:
                        #Fehlwert -> in Zielraster eintragen
                        target[z][s] = -9999

                    else:
                        #Zelle hat Richtungskoordinate
                        #Test ob diese bereits untersucht wurde
                        if done[z][s] == 0.:
                            #Suchroutine aktivieren
                            ##Hilfsvariablen um Ausgangszähler nicht zu überschreiben
                            ac_z = z
                            ac_s = s
                            path = []
                            l0 = 0
                            ##Es wird so lange gesucht bis ein Stream gefunden wurde
                            while str(direc[ac_z][ac_s]) != 'stream':
                                #Wohin fließt es von hier aus
                                next_z = ac_z + options[direc[ac_z][ac_s]][0]
                                next_s = ac_s + options[direc[ac_z][ac_s]][1]
                                
                                #Speichern, sowie die Länge zur mitte der nächsten Zelle
                                path.append([ac_z,ac_s,np.sqrt(np.sum(np.power(options[direc[ac_z][ac_s]],2)))])
                                done[ac_z][ac_s] = 1.
                                
                                #Abbruchtests
                                if next_z < 0 or next_s < 0:
                                    #Wenn Ausflusspunkt (oberer Rand) dann 1/2 Celllaenge
                                    nl = path[-1][2] / 2.
                                    path[-1][2] = nl
                                    break
                                elif next_z == len(direc) or next_s == len(direc[0]):
                                    #Wenn Ausflusspunkt (unterer Rand) dann 1/2 Celllaenge
                                    nl = path[-1][2] / 2.
                                    path[-1][2] = nl
                                    break
                                elif direc[next_z][next_s] == -9999:
                                    #Wenn Ausflusspunkt dann 1/2 Celllaenge
                                    nl = path[-1][2] / 2.
                                    path[-1][2] = nl
                                    break
                                
                                elif done[next_z][next_s] == 1.:
                                    #Wenn FL dieser Zelle bereits bestimmt, diese Laenge als
                                    # Ausgangspunkt für summation verwenden
                                    #l0 = target[next_z][next_s]/cellsize
                                    break


                                #Falls keines der Abbruchkriterien einritt: Zur nächsten Zelle
                                ac_z = next_z
                                ac_s = next_s
                                

                            #StreamIds eintragen
                            if str(direc[ac_z][ac_s]) == 'stream':
                                #Fließweg rückwärts abgehen
                                for cell in path[::-1]:
                                    target[cell[0]][cell[1]] = streamIDs[(ac_z,ac_s)]
                            elif next_z < len(direc) and next_s < len(direc[0]):
                                if done[next_z][next_s] == 1.:
                                    for cell in path[::-1]:
                                        target[cell[0]][cell[1]] = target[next_z][next_s]
                                    
                                
            message = 'writing results...'
            QgsMessageLog.logMessage(message ,'Message')
            path0 = self.dlg.lineEdit_4.text() + ".asc"
            out = open(path0, "w")
            out.write(str(a)+ str(b) + str(c) + str(d)+ str(e) + str(g))
            for line in target:
                outstring = ""
                for elem in line:
                    outstring += str(elem) + "\t"
                outstring += "\n"
                out.write(outstring)
            out.close()

            path = self.dlg.lineEdit_4.text() + '.asc'
            iface.addRasterLayer(path, 'preProcessing1')

            message = 'finished'
            QgsMessageLog.logMessage(message,'Message')
                                   

            pass

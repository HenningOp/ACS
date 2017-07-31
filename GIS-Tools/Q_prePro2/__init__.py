# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ACS_preProcessing2
                                 A QGIS plugin
 Length to drainage
                             -------------------
        begin                : 2017-07-26
        copyright            : (C) 2017 by Laura Bienstein
        email                : laura.bienstein@rub.de
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load ACS_preProcessing2 class from file ACS_preProcessing2.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .acs_preProcessing2 import ACS_preProcessing2
    return ACS_preProcessing2(iface)

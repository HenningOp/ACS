# -*- coding: utf-8 -*-
"""
/***************************************************************************
 acs
                                 A QGIS plugin
 Ascertainment by Catchment Structure
                             -------------------
        begin                : 2017-06-27
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
    """Load acs class from file acs.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .acs_subdivide import acs
    return acs(iface)

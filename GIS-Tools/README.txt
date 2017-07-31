Implementations of ACS-algorithm into toolboxes for GIS-Applications ArcGIS and QGIS.
Indented to make the algorithm applicable for users without scripting experience.

Minimum Requirements:
	ArcGIS	10.2.
	QGIS	2.0.

Installation:
	QGIS
	- Method 1:
		- Start QGIS
		- Go to: Extensions->Manage and install extensions->Settings
		- Add new extension repositories:
			https://github.com/HenningOp/ACS/tree/master/GIS-Tools/Q_ACS
			https://github.com/HenningOp/ACS/tree/master/GIS-Tools/Q_prePro1/ACS_preProcessing1
			https://github.com/HenningOp/ACS/tree/master/GIS-Tools/Q_prePro2
		- Update repositories
		- Go to: Extensions->Manage and install extensions->All
		- Install ACS, ACS_preprocessing1, ACS_preprocessing2
	
	- Method 2 (if behing Proxy):
		- Download archive Qgis_Plugin.rar
		- Extract to QGIS Plugin folger (default: C:\Users\user\.qgis2)
		- Start QGIS
		- Go to:  Extensions->Manage and install extensions->Settings
		- Update repositories
		- Go to: Extensions->Manage and install extensions->All
		- Install ACS, ACS_preprocessing1, ACS_preprocessing2
	
	ArcGIS
	- Download archive ArcGIS_Plugin.rar
	- Unzip files to a GIS conntected folder
	- Start ArcGIS
	- Open Toolbox->Add Toolbox
	- Select folder with unzipped files, select ACS.pyt

	
Further Extensions required:
	No extensions needed

Remarks on usage:
	- All data has to be proprocessed as discribed in the README.md of the main tool
	- ArcGIS tool is awfully slow. Usage of QGIS, or even better the stand alone
	  script, is highly recommended!
	  (Performace example: 1500 sqkm catchment, 100 x 100 m resolution
	   Script ~ 10 minutes
	   QGIS	  ~ 30 minutes
	   ArcGIS ~ 6 hours)
	- QGIS Toolbox does not give feedback on progress, though it might seem unresponsive
	  the tool is working. Give it some time.
	
	   
	   
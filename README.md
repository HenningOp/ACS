# ACS (Ascertainment by catchment structure)
Automated ascertainment of sub-basins &amp; zones to capture the spatial organisation of catchments

The ACS tool is designed to perform an automated and impartial sub-basin ascertainment for
natural basins, based on the spatial structure of an arbitrary catchment characteristic. A full
description of its theoretical background and sequence can be found in:
Oppel, H. and Schumann, A.: A method to employ the spatial organisation of catchments into semi-distributed hydrological models, in progress

System requirements:

	- OS:
		Linux
		Windows 7/8/10	64Bit
	- Script Language:
		Python	Version	2.7.10
	- Required Python Addins (Minimum):
		numpy	Version	1.11.0
		scipy	Version	0.15.1

Installation:

	1. Unzip/Copy script to destination folder
	2. Unzip/Copy data folder and ACS_OS.pyd to destination folder
	3. Script can be started without further installation

Running the program:

	1.	User specications
		- in_root		Foldername where data is stored
		- indata		Filename of considered catchment characteristic
		- start			Filename of initial drainage points file (line 1: Header, following lines: ID->X->Y)
						At least 1 point is required
		- hsteps		Number of quantiles iterated for plateau search (4 recommended)
		- dx			Width of distance stripes (Stream Flow) [m]    (10000 recommended)
		- dy			Width of distance stripes (Hillslopes) [m]    (100 recommended)
		- e				Non-linearity coefficient for objective function
		- thR			Percentage of maximum FAcc for major ramifications
	
	2.	Deploy Data in specified directory
		Note that all data has to comprise identical spatial extend, projection and coordinate system
		- considered catchment characteristic (as specified before)
		- initial drainage points (as specified before)
		- FlowAccumulation, name: in_flowacc.asc 
		- Flow Directions, name: in_flowdir.asc
		- Digital elevation model, name: in_dem.asc
		- Stream flow length, name: in_SFL.asc      (Applied preProcessing_1*)
		- Hillslope flow length, name: in_OFL.asc   (Applied preProcessing_2**)
		- Strahler order, name: in_strahler.asc     (Applied preProcessing_1*)
	
	3.	Run entire Script.
		After processing call follwoing functions for desired output:
		- allout()		Write Sub-basin grid, Zones grid, drainage points txt file
		- spsout()		Write drainage points txt-file
		- allbasin(TRUE/FALSE)	Write Sub-basin grid (TRUE) with zone indicator (as decimals) (FALSE) without zones
		- gridout(x, filename)	write grid x to file (filename)
		

*	PreProcessing_1: Write / Copy information from stream cells to hillslope cells.
*	PreProcessing_2: Calculate overland flow length to next drainage.
*	PreProcessing Scripts/Tools also available at http:\\...


Additional information:
	- See VersionLog.txt for documentation of changes and bugfixes in the code.
	- If anything is unclear or you found bugs (i'm pretty shure there are plenty of 'em) please feel free to contact me (henning.oppel@rub.de)!

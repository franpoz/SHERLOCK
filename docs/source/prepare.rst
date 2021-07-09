.. SHERLOCK PIPEline documentation master file, created by
   sphinx-quickstart on Thu Jul  8 08:43:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================
Preparation stage
=================

SHERLOCK needs to identify the type of source[s] that the user has selected in order to choose the proper data cooking
flow to finally provide standard information for the target star and the photometric measurements in time series format.
The easiest way to depict the process is by following the next diagram:

.. mermaid::

   flowchart TB
       A[Prepare data] --> B{Check mode}
           B --> C[Long cadence target]
           B --> D[Short cadence target]
           B --> E[File target]
           C --> F{Mission}

           D --> Short_builder[Build Lightcurve]
           Lightkurve[/Lightcurve\] -.-> Short_builder
           Short_builder --> StarInfo[Get Star Params]

           F --> TESS_Long[Build TESS Lightcurve]
           F --> Kepler_Long[Build Kepler Lightcurve]
           F --> K2_Long[Build K2 Lightcurve]
           ELEANOR[/ELEANOR Postcard/TessCut\] -.-> TESS_Long
           LightKurve[/Kepler TargetPixelFile\] -.-> Kepler_Long
           LightKurve[/Kepler TargetPixelFile\] -.-> K2_Long
           TESS_Long --> StarInfo[Get Star Params]
           TESS_Long --> StarInfo
           Kepler_Long --> StarInfo
           K2_Long --> StarInfo

           StarInfo --> Target_lightcurve(Prepared data)

           E --> HasName{Has target name?}
           File[/CSV File\] -.-> BuildFromFile
           File[/CSV File\] -.-> BuildFromFile1
           HasName -- No --> BuildFromFile[Build lightcurve]
           HasName -- Yes --> BuildFromFile1[Build lightcurve]
           BuildFromFile --> Target_lightcurve
           BuildFromFile1 --> StarInfo

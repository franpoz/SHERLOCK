# Examples description
Four directories are presented down the examples dir:

* Programmatical: These are coding examples directly using the Sherlock class in Python code. We present several
possible configurations to be done by using the Sherlock class construction.

* Properties: The examples placed under this directory will assume that Sherlock will be launched by using
`python3 -m sherlockpipe --properties my_properties.yaml`, which will read all the default properties 
defined and the ones you provide in your file generating the proper Sherlock configurations for the run 
to be executed.

* Vetting: The examples placed under this directory will assume that the vetting procedure
will be launched with `python3 -m sherlockpipe.vet --properties my_properties.yaml`.

* Fitting: The examples placed under this directory will assume that the fitting procedure
will be launched with `python3 -m sherlockpipe.fit --properties my_properties.yaml`.
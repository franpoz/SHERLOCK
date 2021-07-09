.. SHERLOCK PIPEline documentation master file, created by
   sphinx-quickstart on Thu Jul  8 08:43:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================
Candidates Search
=================

Once the preparation stage is already performed, the search iterations begin. SHERLOCK uses wotan to generate ``N``
different lightcurves whose main difference is the window size of the detrending algorithm used to generate them. That
is, we increase the window size from the lowest possible value that would not affect a long transit until an upper value
that can be customized by the user.

.. mermaid::

   flowchart TB
       A[Detrend target lightcurve] --> B[/Detrended light curves\]
       B --> C[Search for candidate]
       C --> D[Compute best signal for lightcurve]
       D --> F{More detrends?}
       F -- Yes --> G[Select different window size]
       G --> C
       F -- No --> Compute[Compute Best Signal]
       Signals[/Selected signals set\] -.-> Compute
       SelectionAlgorithm[/SelectionAlgorithm/] -.-> Compute
       Compute --> Good{Bad signal or max runs reached?}
       Good -- No --> Mask[Mask selected signal]
       Mask --> B
       Good -- Yes --> End(No more signals)

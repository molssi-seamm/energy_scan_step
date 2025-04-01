=======
History
=======
2025.4.1 -- Bugfix: Error handling the coordinates in some cases
   * There was an error in handling the coordinates created by changes in the RDKit module
     in molsystem. This fixes it.
   * The incorrect logger was used for debugging.
   * Small format updates to due to changes in the code formatting rules in black.
     
2024.12.14 -- Cleanup! Reasonable working version.

2024.5.23 -- Initial working version
   * Can handle scans, freezing coordinates. Is not optimized yet.
   * Produces a graph of 1- and 2-D scans
   * Creates an SDF file with the optmized points along scans, which can be viewed in
     the Dashboard.

2024.4.3 (2024-04-03)
---------------------

* Plug-in created using the SEAMM plug-in cookiecutter.

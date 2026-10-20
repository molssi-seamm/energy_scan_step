=======
History
=======
2026.10.20 -- A resumed scan picks up where it stopped
    * A scan run again in the same job -- resumed, or resubmitted after it ran out of
      time -- started over from its first point in new directories ('... 2'). Each
      finished point is now recorded in its directory (point.json: the energy, the
      optimized geometry and how the optimization went), and a point an earlier
      run finished is reused when its constraints, atoms and starting geometry are
      the same, so the scan continues as if it had not stopped. The point that was
      running when the job stopped is set aside ('... .incomplete-1'), keeping its
      output, and run again. The table marks the reused points.

2026.10.19 -- Bugfix: scanning a dihedral or angle in a ring
    * A dihedral or angle about a bond in a ring, such as the pucker of ethylene
      carbonate, could not be scanned or fixed: RDKit cannot set it, and the scan
      stopped. It is now left to the optimizer, which drives it to each value with a
      constraint ($set in geomeTRIC); the output notes this.

2026.9.25 -- Bugfix: the Energy Scan step failed to load on Python 3.12 outside conda
    * The 2026.3.1 release, which replaced the deprecated ``pkg_resources`` module
      with ``importlib``, never reached PyPI because its publishing run failed, so
      installations still had the 2025.8.20 code. In an environment without
      setuptools (a plain venv, for example) that version failed to import with
      ``No module named 'pkg_resources'`` and the step was missing from the
      flowchart editor. This release publishes the fix.

2026.3.1 -- Internal: switching from deprecated library pkg_resources to importlib

2025.8.20 -- Improved the graphs and added spins to them if available.
   * Added the ability to output graphs to one or more of the following formats: html,
     png, webp, svg, or pdf.
   * Capture the spin values, S**2 and the projected S**2, if captured by the quantum
     code, and add them to the graphs.
   * Fixed an issue if the two atoms in a bond scan were not actually bonded.

2025.8.6 -- Allows the values defining the scans to be variables.
   * This enhancement allows the variables defining the scan or frozen coordinate to be
     variables as well as static values.

2025.8.5 -- Bugfix: not correctly naming the configurations in the scan
   * The code did not correctly name the configurations generated during a scan, so
     their names were None.

2025.5.7 -- Enhancement to continue if minimization does not converge
   * Continue even if the minimization does not fully converge, as sometimes seems to be
     the case.
     
2025.4.11 -- Enhancements to allow repeated points
   * Allows the path to have repeated points, which is useful to e.g. go forwards and
     backawards to see the hysteresis.
     
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

# -*- coding: utf-8 -*-

"""Non-graphical part of the Energy Scan step in a SEAMM flowchart"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import pkg_resources
import pprint  # noqa: F401
import re
import sys
import traceback

import geometric
import geometric.molecule
import numpy as np
import rdkit.Chem
from tabulate import tabulate

import energy_scan_step
import molsystem
import read_structure_step
import seamm
from seamm_util import getParser, parse_list, Q_, units_class
import seamm_util.printing as printing
from seamm_util.printing import FormattedText as __

# In addition to the normal logger, two logger-like printing facilities are
# defined: "job" and "printer". "job" send output to the main job.out file for
# the job, and should be used very sparingly, typically to echo what this step
# will do in the initial summary of the job.
#
# "printer" sends output to the file "step.out" in this steps working
# directory, and is used for all normal output from this step.

logger = logging.getLogger(__name__)
job = printing.getPrinter()
printer = printing.getPrinter("Energy Scan")

# Add this module's properties to the standard properties
path = Path(pkg_resources.resource_filename(__name__, "data/"))
csv_file = path / "properties.csv"
if path.exists():
    molsystem.add_properties_from_file(csv_file)

# Regexp to remove ansi escape sequences
ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


class SEAMMEngine(geometric.engine.Engine):
    """Helper class that is a geomeTRIC engine and connects to SEAMM."""

    def __init__(self, step, molecule):
        """Initialize this geomeTRIC engine.

        Parameters
        ----------
        step : seamm.node
            The SEAMM plug-in using this.
        molecule : geometric.molecule.Molecule
            The geomeTRIC molecule to work with
        """
        self.step = step
        super().__init__(molecule)

    def calc_new(self, coords, dirname):
        """The method to calculate the new energy and forces.

        Parameters
        ----------
        coords : np.ndarray
            A 1-D Numpy array of the coordinates, in Bohr
        dirname : The name of a directory (not used)

        Returns
        -------
        data : {str : any}
            The result returned to geomeTRIC:

                "energy" : The energy, in Hartree

                "gradient" : The 1st derivative, or gradient, of the energy in
                Hartree/bohr as a 1-D Numpy array.
        """
        xyz = coords.reshape(-1, 3) * Q_(1.0, "a_0").m_as("angstrom")

        energy, gradient = self.step.calculate_gradients(xyz)

        return {"energy": energy, "gradient": gradient.ravel()}


class EnergyScan(seamm.Node):
    """
    The non-graphical part of a Energy Scan step in a flowchart.

    Attributes
    ----------
    parser : configargparse.ArgParser
        The parser object.

    options : tuple
        It contains a two item tuple containing the populated namespace and the
        list of remaining argument strings.

    subflowchart : seamm.Flowchart
        A SEAMM Flowchart object that represents a subflowchart, if needed.

    parameters : EnergyScanParameters
        The control parameters for Energy Scan.

    See Also
    --------
    TkEnergyScan,
    EnergyScan, EnergyScanParameters
    """

    def __init__(
        self,
        flowchart=None,
        title="Energy Scan",
        namespace="org.molssi.seamm",
        extension=None,
        logger=logger,
    ):
        """A step for Energy Scan in a SEAMM flowchart.

        You may wish to change the title above, which is the string displayed
        in the box representing the step in the flowchart.

        Parameters
        ----------
        flowchart: seamm.Flowchart
            The non-graphical flowchart that contains this step.

        title: str
            The name displayed in the flowchart.
        namespace : str
            The namespace for the plug-ins of the subflowchart
        extension: None
            Not yet implemented
        logger : Logger = logger
            The logger to use and pass to parent classes

        Returns
        -------
        None
        """
        # logger.setLevel(logging.DEBUG)
        logger.debug(f"Creating Energy Scan {self}")
        self.subflowchart = seamm.Flowchart(
            parent=self, name="Energy Scan", namespace=namespace
        )

        super().__init__(
            flowchart=flowchart,
            title="Energy Scan",
            extension=extension,
            module=__name__,
            logger=logger,
        )

        self._metadata = energy_scan_step.metadata
        self.parameters = energy_scan_step.EnergyScanParameters()
        self._step = 0
        self._file_handler = None
        self._constraints = None
        self._working_directory = None
        self._working_configuration = None

    @property
    def version(self):
        """The semantic version of this module."""
        return energy_scan_step.__version__

    @property
    def git_revision(self):
        """The git version of this module."""
        return energy_scan_step.__git_revision__

    @property
    def constraints(self):
        if self._constraints is None:
            self._constraints = self.parse_constraints()
        return self._constraints

    @property
    def step(self):
        """The calculation number in the scan."""
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    @property
    def working_configuration(self):
        """The configuration being worked on."""
        return self._working_configuration

    @property
    def working_directory(self):
        """The directory being worked on."""
        return self._working_directory

    def analyze(self, indent="", **kwargs):
        """Do any analysis of the output from this step.

        Also print important results to the local step.out file using
        "printer".

        Parameters
        ----------
        indent: str
            An extra indentation for the output
        """
        # Get the first real node
        node = self.subflowchart.get_node("1").next()

        # Loop over the subnodes, asking them to do their analysis
        while node is not None:
            for value in node.description:
                printer.important(value)
                printer.important(" ")

            node.analyze()

            node = node.next()

    def calculate_gradients(self, coordinates):
        """Given the new coordinates, calculate the energy and gradients.

        Parameters
        ----------
        coordinates : [3, n_atoms] array of coordinates
        """
        self.step = self.step + 1
        fmt = "05d"

        # Make the geometric output readable by removing the ANSI escape sequences
        logPath = Path("geomeTRIC.out")
        if logPath.exists():
            text = logPath.read_text()
            text = ansi_escape.sub("", text)
            logPath.write_text(text)

        n_atoms = self.working_configuration.n_atoms

        if self.logger.isEnabledFor(logging.DEBUG):
            print("\nnew coordinates")
            for i in range(n_atoms):
                print(
                    f"   {coordinates[i][0]:8.3f} {coordinates[i][1]:8.3f} "
                    f"{coordinates[i][2]:8.3f}"
                )

        # Set the coordinates in the configuration
        self.working_configuration.atoms.set_coordinates(coordinates, fractionals=False)

        # Find the handler for job.out and set the level up
        job_handler = None
        out_handler = None
        for handler in job.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and "job.out" in handler.baseFilename
            ):
                job_handler = handler
                job_level = job_handler.level
                job_handler.setLevel(printing.JOB)
            elif isinstance(handler, logging.StreamHandler):
                out_handler = handler
                out_level = out_handler.level
                out_handler.setLevel(printing.JOB)

        # Get the first real node
        first_node = self.subflowchart.get_node("1").next()

        # Ensure the nodes have their options
        node = first_node
        while node is not None:
            node.all_options = self.all_options
            node = node.next()

        # And the subflowchart has the executor
        self.subflowchart.executor = self.flowchart.executor

        # Direct most output to iteration.out
        step_id = f"step_{self.step:{fmt}}"
        step_dir = Path(self.working_directory) / step_id
        step_dir.mkdir(parents=True, exist_ok=True)

        # A handler for the file
        if self._file_handler is not None:
            self._file_handler.close()
            job.removeHandler(self._file_handler)
        path = step_dir / "Step.out"
        path.unlink(missing_ok=True)
        self._file_handler = logging.FileHandler(path)
        self._file_handler.setLevel(printing.NORMAL)
        formatter = logging.Formatter(fmt="{message:s}", style="{")
        self._file_handler.setFormatter(formatter)
        job.addHandler(self._file_handler)

        # Add the step to the ids so the directory structure is reasonable
        self.subflowchart.reset_visited()
        name = self.working_directory.name
        self.set_subids((*self._id, name, step_id))

        # Run through the steps in the loop body
        node = first_node
        try:
            while node is not None:
                node = node.run()
        except DeprecationWarning as e:
            printer.normal("\nDeprecation warning: " + str(e))
            traceback.print_exc(file=sys.stderr)
            traceback.print_exc(file=sys.stdout)
        except Exception as e:
            printer.job(f"Caught exception in step {self.step}: {str(e)}")
            with open(step_dir / "stderr.out", "a") as fd:
                traceback.print_exc(file=fd)
            raise
        self.logger.debug(f"End of step {self.step}")

        # Remove any redirection of printing.
        if self._file_handler is not None:
            self._file_handler.close()
            job.removeHandler(self._file_handler)
            self._file_handler = None
        if job_handler is not None:
            job_handler.setLevel(job_level)
        if out_handler is not None:
            out_handler.setLevel(out_level)

        # Get the energy and derivatives
        paths = sorted(step_dir.glob("**/Results.json"))

        if len(paths) == 0:
            raise RuntimeError(
                "There are no energy and gradients in properties.json for step "
                f"{self.step} in {step_dir}."
            )
        else:
            # Find the most recent and assume that is the one wanted
            newest_time = None
            for path in paths:
                with path.open() as fd:
                    data = json.load(fd)
                time = datetime.fromisoformat(data["iso time"])
                if newest_time is None:
                    newest = path
                    newest_time = time
                elif time > newest_time:
                    newest_time = time
                    newest = path
            with newest.open() as fd:
                data = json.load(fd)

        energy = data["energy"]
        if "energy,units" in data:
            units = data["energy,units"]
        else:
            units = "kJ/mol"
        energy *= Q_(1.0, units).to("E_h").magnitude

        gradients = data["gradients"]

        if self.logger.isEnabledFor(logging.DEBUG):
            logger.debug("\ngradients")
            for i in range(n_atoms):
                logger.debug(
                    f"   {gradients[i][0]:8.3f} {gradients[i][1]:8.3f} "
                    f"{gradients[i][2]:8.3f}"
                )

        if "gradients,units" in data:
            units = data["gradients,units"]
        else:
            units = "kJ/mol/Å"

        # Units!
        gradients = np.array(gradients) * Q_(1.0, units).to("E_h/a_0").magnitude

        return energy, gradients

    def create_parser(self):
        """Setup the command-line / config file parser"""
        parser_name = "energy-scan-step"
        parser = getParser()

        # Remember if the parser exists ... this type of step may have been
        # found before
        parser_exists = parser.exists(parser_name)

        # Create the standard options, e.g. log-level
        super().create_parser(name=parser_name)

        if not parser_exists:
            # Any options for the energy itself
            parser.add_argument(
                parser_name,
                "--html",
                action="store_true",
                help="whether to write out html files for graphs, etc.",
            )

        # Now need to walk through the steps in the subflowchart...
        self.subflowchart.reset_visited()
        node = self.subflowchart.get_node("1").next()
        while node is not None:
            node = node.create_parser()

        return self.next()

    def description_text(self, P=None, short=False):
        """Create the text description of what this step will do.
        The dictionary of control values is passed in as P so that
        the code can test values, etc.

        Parameters
        ----------
        P: dict
            An optional dictionary of the current values of the control
            parameters.
        Returns
        -------
        str
            A description of the current step.
        """
        if not P:
            P = self.parameters.values_to_dict()

        text = "Scan the following coordinates:"

        # Prepare the table of coordinates
        table = {
            "Name": [],
            "Operation": [],
            "Type": [],
            "SMARTS": [],
            "Which?": [],
            "Values": [],
            "Units": [],
            "Scan Type": [],
            "Direction": [],
        }
        for name, data in self.parameters["constraints"].value.items():
            table["Name"].append(name)
            table["Operation"].append(data["operation"])
            table["Type"].append(data["type"])
            table["SMARTS"].append(data["SMARTS"])
            table["Which?"].append(data["which"])
            table["Values"].append(data["values"])
            table["Units"].append(data["units"])
            if data["operation"] == "scan":
                table["Scan Type"].append(data["scan type"])
                table["Direction"].append(data["direction"])
            else:
                table["Scan Type"].append("")
                table["Direction"].append("")

        tmp = tabulate(
            table,
            headers="keys",
            tablefmt="simple_outline",
            disable_numparse=True,
            colalign=(
                "left",
                "left",
                "left",
                "left",
                "left",
                "left",
                "left",
                "left",
                "left",
            ),
        )
        length = len(tmp.splitlines()[0])

        text += "\n"
        text += 8 * " " + "Coordinates".center(length)
        text += "\n"
        for line in tmp.splitlines():
            text += 8 * " " + line + "\n"
        text += "\n"

        if not short:
            self.subflowchart.root_directory = self.flowchart.root_directory

            # Get the first real node
            node = self.subflowchart.get_node("1").next()

            while node is not None:
                try:
                    text += __(node.description_text(), indent=3 * " ").__str__()
                except Exception as e:
                    print(f"Error describing energy_scan flowchart: {e} in {node}")
                    self.logger.critical(
                        f"Error describing energy_scan flowchart: {e} in {node}"
                    )
                    raise
                except:  # noqa: E722
                    print(
                        "Unexpected error describing energy_scan flowchart: "
                        f"{sys.exc_info()[0]} in {str(node)}"
                    )
                    self.logger.critical(
                        "Unexpected error describing energy_scan flowchart: "
                        f"{sys.exc_info()[0]} in {str(node)}"
                    )
                    raise
                text += "\n"
                node = node.next()

        return self.header + "\n" + __(text, **P, indent=4 * " ").__str__()

    def parse_constraints(self):
        """Parse the given constraints.

        The constraints are stored as a dict of dicts::

            "distances": {...}
            "angles": {...}
            "dihedrals" {
                "constraints": {name1: data1, ...]  # see below
                "key atoms": {
                    (j, k): [(constraint_name, match_no), ....]
                    ...
                }

        where::

            data = {  # in constraints above
                "name": <unique name of constraint>,
                "operation": "<scan, freeze or set>",
                "type": "<distance, angle, dihedral>",
                "SMARTS": "<SMARTS string>",
                "value1": "<starting value>",
                "value2": "<end value for scan>",
                "step": "<step in Bohr or degrees>",
                "units": "<units for step>",
                "atoms": [ordered list of tuples of atom numbers],
                "matched atoms": [ordered list of tuples of all atoms matched],
                "mapping": [ordered indices of atoms in <matched atoms>]
            }
        """

        # Get the values of the parameters, dereferencing any variables
        P = self.parameters.current_values_to_dict(
            context=seamm.flowchart_variables._data
        )

        # And the molecule
        system, configuration = self.get_system_configuration(None)
        n_atoms = configuration.n_atoms

        # Find the coordinates of the constraint
        result = {
            "distances": {
                "constraints": {},
                "key atoms": {},
            },
            "angles": {
                "constraints": {},
                "key atoms": {},
            },
            "dihedrals": {
                "constraints": {},
                "key atoms": {},
            },
        }

        rdkMol = configuration.to_RDKMol()

        for name, original in P["constraints"].items():
            data = {**original}
            _type = data["type"]

            pattern = rdkit.Chem.MolFromSmarts(data["SMARTS"])

            ind_map = {}
            for atom in pattern.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num:
                    ind_map[map_num] = atom.GetIdx()
            map_list = [ind_map[x] for x in sorted(ind_map)]

            if _type == "distance":
                if len(map_list) != 2:
                    raise RuntimeError(
                        f"A distance constraint has {len(map_list)} atoms, not 2!\n"
                        f"      {name=}\n"
                        f"    {data['SMARTS']=}"
                    )
                type_data = result["distances"]
            elif _type == "angle":
                if len(map_list) != 3:
                    raise RuntimeError(
                        f"An angle constraint has {len(map_list)} atoms, not 3!\n"
                        f"      {name=}\n"
                        f"    {data['SMARTS']=}"
                    )
                type_data = result["angles"]
            elif _type == "dihedral":
                if len(map_list) != 4:
                    raise RuntimeError(
                        f"A dihedral constraint has {len(map_list)} atoms, not 4!\n"
                        f"      {name=}\n"
                        f"    {data['SMARTS']=}"
                    )
                type_data = result["dihedrals"]
            else:
                raise RuntimeError(f"Cannot handle constraint of type '{_type}'.")

            type_data["constraints"][name] = data
            data["atoms"] = []
            data["matched atoms"] = []
            data["mapping"] = map_list

            matches = rdkMol.GetSubstructMatches(pattern, maxMatches=12 * n_atoms)

            if len(matches) == 0:
                raise RuntimeError(
                    f"""no matches for SMARTS '{data["SMARTS"]} for {name}"""
                )

            for match_no, _match in enumerate(matches):
                self.logger.debug(f"{match_no}: {_match}")
                data["matched atoms"].append(_match)
                atoms = [_match[x] for x in map_list]
                data["atoms"].append(atoms)
                if _type == "distance":
                    i, j = atoms
                    key = (i, j) if i < j else (j, i)
                elif _type == "angle":
                    i, j, k = atoms
                    key = (i, j, k) if i < k else (k, j, i)
                elif _type == "dihedral":
                    j, k = atoms[1:3]
                    key = (j, k) if j < k else (k, j)
                else:
                    raise RuntimeError(f"Cannot handle constraint of type '{_type}'.")

                if key not in type_data["key atoms"]:
                    type_data["key atoms"][key] = []
                type_data["key atoms"][key].append((name, match_no))

        return result

    def run(self):
        """Run a Energy Scan step.

        Parameters
        ----------
        None

        Returns
        -------
        seamm.Node
            The next node object in the flowchart.
        """
        next_node = super().run(printer)

        # Get the values of the parameters, dereferencing any variables
        P = self.parameters.current_values_to_dict(
            context=seamm.flowchart_variables._data
        )

        # Have to fix formatting for printing...
        PP = dict(P)
        for key in PP:
            if isinstance(PP[key], units_class):
                PP[key] = "{:~P}".format(PP[key])

        # Print what we are doing
        printer.important(__(self.description_text(P, short=True), indent=self.indent))

        # Create the directory
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)

        system, starting_configuration = self.get_system_configuration(None)
        n_atoms = starting_configuration.n_atoms

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("initial coordinates")
            coordinates = starting_configuration.coordinates
            symbols = starting_configuration.atoms.symbols
            for i in range(n_atoms):
                self.logger.debug(
                    f"   {symbols[i]} {coordinates[i][0]:8.3f} {coordinates[i][1]:8.3f}"
                    f" {coordinates[i][2]:8.3f}"
                )
            self.logger.debug(starting_configuration.bonds)

        # Get the RDKit molecule for later use
        rdkMol = starting_configuration.to_RDKMol()
        rdkConfs = rdkMol.GetConformers()
        if len(rdkConfs) != 1:
            raise RuntimeError(f"There are {len(rdkConfs)} RDKit conformers")
        rdkConf = rdkConfs[0]

        data = self.setup_constraints(rdkConf)

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Data from setup_constraints")
            self.logger.debug(pprint.pformat(data))

        current = {}
        scans = [scan for scan in data["scans"]]
        current = {scan: 0 for scan in scans}
        sorted_points = [sorted(data["scans"][scan]["points"]) for scan in scans]

        # Setup a table for printing the results
        table = {scan: [] for scan in scans}
        table["Steps"] = []
        table["Energy"] = []

        all_data = {}

        # Loop over all the scans, incrementing each in turn
        finished = False
        self._working_configuration = starting_configuration
        while not finished:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("step coordinates")
                coordinates = self.working_configuration.coordinates
                symbols = self.working_configuration.atoms.symbols
                for i in range(n_atoms):
                    self.logger.debug(
                        f"   {symbols[i]} {coordinates[i][0]:8.3f} "
                        f"{coordinates[i][1]:8.3f} {coordinates[i][2]:8.3f}"
                    )
                self.logger.debug(self.working_configuration.bonds)

            # Set the coordinates to the current value
            rdkMol = self.working_configuration.to_RDKMol()
            rdkConf = rdkMol.GetConformers()[0]

            label = []
            pts = []
            constraint_text = []
            constraint_text.append("$freeze")
            for scan in scans:
                _type = data["scans"][scan]["type"]
                _atoms = data["scans"][scan]["atoms"]
                _point = data["scans"][scan]["points"][current[scan]]

                pts.append(_point)
                self.logger.info(
                    f"{_point=}  {data['scans'][scan]['points']} {current[scan]}"
                )
                label.append(f"{scan} {_point}")
                table[scan].append(_point)

                atom_string = " ".join([str(at + 1) for at in _atoms])
                constraint_text.append(f"{_type} {atom_string}")

                if _type == "distance":
                    iat, jat = _atoms
                    rdkit.Chem.rdMolTransforms.SetBondLength(rdkConf, iat, jat, _point)
                elif _type == "angle":
                    iat, jat, kat = _atoms
                    rdkit.Chem.rdMolTransforms.SetAngleDeg(
                        rdkConf, iat, jat, kat, _point
                    )
                elif _type == "dihedral":
                    iat, jat, kat, lat = _atoms
                    rdkit.Chem.rdMolTransforms.SetDihedralDeg(
                        rdkConf, iat, jat, kat, lat, _point
                    )
            for frozen in data["freeze"].keys():
                _type = data["freeze"][frozen]["type"]
                _atoms = data["freeze"][frozen]["atoms"]
                _value = data["freeze"][frozen]["value"]

                self.logger.info(f" freeze{_value=}")

                atom_string = " ".join([str(at + 1) for at in _atoms])
                constraint_text.append(f"{_type} {atom_string}")

                if _value != "current":
                    if _type == "distance":
                        iat, jat = _atoms
                        rdkit.Chem.rdMolTransforms.SetBondLength(
                            rdkConf, iat, jat, _value
                        )
                    elif _type == "angle":
                        iat, jat, kat = _atoms
                        rdkit.Chem.rdMolTransforms.SetAngleDeg(
                            rdkConf, iat, jat, kat, _value
                        )
                    elif _type == "dihedral":
                        iat, jat, kat, lat = _atoms
                        rdkit.Chem.rdMolTransforms.SetDihedralDeg(
                            rdkConf, iat, jat, kat, lat, _value
                        )

            self._working_directory = directory / "__".join(label)
            self.working_directory.mkdir(parents=True, exist_ok=True)

            label = " @ ".join(label)

            # Create a new configuration for this scan
            # closest = self._find_closest_configuration(pts, sorted_points, all_data)

            _, self._working_configuration = self.get_system_configuration(
                P={
                    "structure handling": "Create a new configuration",
                    "configuration name": label,
                    "system name": "keep current name",
                },
            )
            system.configuration = self.working_configuration

            # Set the coordinates to the updated ones for this scan
            self.working_configuration.coordinates_from_RDKMol(rdkMol)

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("updated coordinates")
                coordinates = self.working_configuration.coordinates
                symbols = self.working_configuration.atoms.symbols
                for i in range(n_atoms):
                    self.logger.debug(
                        f"   {symbols[i]} {coordinates[i][0]:8.3f} "
                        f"{coordinates[i][1]:8.3f} {coordinates[i][2]:8.3f}"
                    )
                self.logger.debug(self.working_configuration.bonds)

            coordinates = self.working_configuration.atoms.get_coordinates(
                fractionals=False, as_array=True
            )

            geoMol = geometric.molecule.Molecule()
            geoMol.elem = self.working_configuration.atoms.symbols
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("coordinates")
                for i in range(n_atoms):
                    self.logger.debug(
                        f"   {coordinates[i][0]:8.3f} {coordinates[i][1]:8.3f} "
                        f"{coordinates[i][2]:8.3f}"
                    )

            geoMol.xyzs = [coordinates]

            customengine = SEAMMEngine(self, geoMol)

            coordsys = P["coordinate system"].split(":")[0].lower()

            if P["max steps"] == "default":
                max_steps = 12 * self.working_configuration.n_atoms
            else:
                max_steps = P["max steps"]

            enforce = P["enforce"]
            units = str(enforce.u)
            if units == "deg":
                enforce_value = enforce.m_as("radians")
            else:
                enforce_value = enforce.m_as("bohr")

            kwargs = {
                "coordsys": coordsys,
                "maxiter": max_steps,
                "enforce": enforce_value,
            }

            constraints_path = self.working_directory / "constraints.txt"
            constraints_path.write_text("\n".join(constraint_text))

            self.step = 0
            logPath = self.working_directory / "geomeTRIC.out"
            logIni = self.working_directory / "log.ini"
            logIni.write_text(
                f"""\
# The default logging configuration file for geomeTRIC
# Modified to write to {logPath}

[loggers]
keys=root

[handlers]
keys=file_handler

[formatters]
keys=formatter

[logger_root]
level=INFO
handlers=file_handler

[handler_file_handler]
class=geometric.nifty.RawFileHandler
level=INFO
formatter=formatter
args=("{logPath}",)

[formatter_formatter]
format=%(message)s
#format=%(asctime)s %(name)-12s %(levelname)-8s %(message)s
"""
            )
            cc_logger = logging.getLogger("cclib")
            cc_logger.setLevel(logging.WARNING)

            converged = True
            with cd(self.working_directory):
                try:
                    m = geometric.optimize.run_optimizer(
                        logIni=str(logIni),
                        customengine=customengine,
                        input="optimization.txt",
                        constraints=str(constraints_path),
                        qdata=True,
                        **kwargs,
                    )
                except geometric.errors.GeomOptNotConvergedError:
                    converged = False

            # Make the geometric output readable by removing the ANSI escape sequences
            if logPath.exists():
                text = logPath.read_text()
                text = ansi_escape.sub("", text)
                logPath.write_text(text)

            # Get the optimized energy & geometry
            energy = m.qm_energies[-1] * Q_(1.0, "E_h").to("kJ/mol").magnitude
            coordinates = m.xyzs[-1].reshape(-1, 3)

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("optimized coordinates")
                for i in range(n_atoms):
                    self.logger.debug(
                        f"   {coordinates[i][0]:8.3f} {coordinates[i][1]:8.3f} "
                        f"{coordinates[i][2]:8.3f}"
                    )

            # Set the coordinates in the configuration
            self.working_configuration.atoms.set_coordinates(
                coordinates, fractionals=False
            )

            all_data[tuple(pts)] = {
                "energy": energy,
                "geometry": self.working_configuration,
                "label": label,
            }

            table["Steps"].append(self.step)
            table["Energy"].append(f"{energy:.6f}")

            tmp = tabulate(
                table,
                headers="keys",
                tablefmt="simple_outline",
                disable_numparse=False,
            )
            if len(table["Energy"]) == 1:
                length = len(tmp.splitlines()[0])
                text = ["\n" + 8 * " " + "Scans".center(length)]
                for line in tmp.splitlines()[:-1]:
                    text.append(8 * " " + line)
                text = "\n".join(text)
            else:
                text = 8 * " " + tmp.splitlines()[-2]
            footer = 8 * " " + tmp.splitlines()[-1] + "\n"

            printer.important(text)

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("step optimized coordinates")
                coordinates = self.working_configuration.coordinates
                symbols = self.working_configuration.atoms.symbols
                for i in range(n_atoms):
                    self.logger.debug(
                        f"   {symbols[i]} {coordinates[i][0]:8.3f} "
                        f"{coordinates[i][1]:8.3f} {coordinates[i][2]:8.3f}"
                    )
                self.logger.debug(self.working_configuration.bonds)

            if not converged:
                printer.important("\n")
                printer.important(self.indent + "Optimization did not converge!")
                printer.important("\n")
                raise RuntimeError("Energy Scan: Optimization did not converge!")

            for scan in scans:
                self.logger.info(f"{scan=}")
                self.logger.info(
                    f"if {current[scan]} + 1 < {len(data['scans'][scan]['points'])}:"
                )
                if current[scan] + 1 < len(data["scans"][scan]["points"]):
                    current[scan] += 1
                    self.logger.info(f"{current[scan]=}")
                    break
                else:
                    # Progress to next scan
                    current[scan] = 0
                    self.logger.info(f"{current[scan]=}")
                    if scan == scans[-1]:
                        # On last one!
                        finished = True
                        break
                self.logger.info("")

        printer.important(footer)

        # Output all the tables and graphs
        # Sort the points ...
        all_points = [*all_data.keys()]
        all_points.sort()
        energies = np.array([all_data[pt]["energy"] for pt in all_points])
        energies -= np.min(energies)

        n_points = [len(pts) for pts in sorted_points]

        energies.shape = n_points
        energies = energies.round(decimals=2).tolist()
        self.logger.debug(pprint.pformat(energies))

        types = [data["scans"][scan]["type"] for scan in scans]
        self._create_energy_graph(scans, types, sorted_points, energies)

        scan_configurations = [all_data[pt]["geometry"] for pt in all_points]
        self._create_sdf(scan_configurations)

        return next_node

    def set_id(self, node_id=()):
        """Sequentially number the subnodes"""
        self.logger.debug("Setting ids for subflowchart {}".format(self))
        if self.visited:
            return None
        else:
            self.visited = True
            self._id = node_id
            self.set_subids(self._id)
            return self.next()

    def set_subids(self, node_id=()):
        """Set the ids of the nodes in the subflowchart"""
        node = self.subflowchart.get_node("1").next()
        n = 1
        while node is not None:
            node = node.set_id((*node_id, str(n)))
            n += 1

    def setup_constraints(self, rdkConf):
        """Work through the given constraints, turning them into actionable data.

        This method takes the constraints as given by the user and turns them into the
        various forms that geomeTRIC takes. Also, rather than use geomeTRIC for scans,
        they will be performed in this plug-in with the scan coordinate turned into a
        "freeze" coordinate.

        Parameters
        ----------
        rdkConf : rdkit.Conformer()
            The RDKit conformer corresponding to the current conformer.
        """
        self.logger.debug("\n------------------ self.constriants -----------------\n")
        self.logger.debug(pprint.pformat(self.constraints))

        result = {}
        scan_data = result["scans"] = {}
        freeze_data = result["freeze"] = {}
        set_data = result["set"] = {}

        # Handle the types of constraints
        for _type in ("distances", "angles", "dihedrals"):
            type_data = self.constraints[_type]

            self.logger.debug(
                f"\n------------------ {_type[0:-1]} constraints -------------\n"
                + pprint.pformat(type_data)
            )

            if "key atoms" not in type_data or len(type_data["key atoms"]) == 0:
                continue

            constraints = type_data["constraints"]

            # find the current values of all the possibilities
            counts = {}
            for name in constraints:
                counts[name] = 0
                for key, key_data in type_data["key atoms"].items():
                    found = False
                    for tmp, match_no in key_data:
                        if tmp == name:
                            found = True
                            break
                    if not found:
                        continue

                    counts[name] += 1
                    count = counts[name]

                    values = []
                    indices = []
                    for tmp, match_no in key_data:
                        if tmp != name:
                            continue
                        indices.append(match_no)
                        data = constraints[name]
                        if _type == "distances":
                            i, j = data["atoms"][match_no]
                            r = rdkit.Chem.rdMolTransforms.GetBondLength(rdkConf, i, j)
                            self.logger.debug(f"{r=}")
                            values.append(r)
                            factor = Q_(1.0, data["units"]).m_as("Å")
                        elif _type == "angles":
                            i, j, k = data["atoms"][match_no]
                            theta = rdkit.Chem.rdMolTransforms.GetAngleDeg(
                                rdkConf, i, j, k
                            )
                            self.logger.debug(f"{theta=}")
                            values.append(theta)
                            factor = Q_(1.0, data["units"]).m_as("degree")
                        elif _type == "dihedrals":
                            i, j, k, lat = data["atoms"][match_no]
                            phi = rdkit.Chem.rdMolTransforms.GetDihedralDeg(
                                rdkConf, i, j, k, lat
                            )
                            self.logger.debug(f"{phi=}")
                            values.append(phi)
                            factor = Q_(1.0, data["units"]).m_as("degree")

                    if len(values) == 0:
                        continue

                    if data["which"] == "all":
                        pass
                    elif data["which"] == "first":
                        if count > 1:
                            break
                    else:
                        if count != int(data["which"]):
                            continue

                    if data["operation"] == "scan":
                        # Get the individual points in Angstrom
                        points = [
                            round(factor * v, 6)
                            for v in parse_list(data["values"], duplicates=False)
                        ]

                        # Work out an order for the points
                        if data["direction"] == "increasing":
                            points = sorted(points)

                            # Find instance closest to first point
                            tmp = np.abs(np.array(values) - points[0])
                            index = tmp.argmin()
                            closest = tmp.min()
                        elif data["direction"] == "decreasing":
                            points = sorted(points, reverse=True)

                            # Find instance closest to first point
                            tmp = np.abs(np.array(values) - points[0])
                            index = tmp.argmin()
                            closest = tmp.min()
                        else:
                            # Start as close as possible to current value
                            closest = None
                            index = None
                            for i, v in enumerate(values):
                                tmp = np.abs(np.array(points) - v)
                                if closest is None or tmp.min() < closest:
                                    pt = tmp.argmin()
                                    closest = tmp.min()
                                    index = i
                            points = points[pt:] + points[pt - 1 :: -1]

                        atoms = data["atoms"][indices[index]]

                        self.logger.debug(
                            f"{name}#{count} -- {index} --> {indices[index]} {closest}"
                        )
                        self.logger.debug(f"{points=}")
                        string = " ".join([str(x + 1) for x in atoms])
                        self.logger.debug(f"{string=}")

                        scan_data[f"{name}#{count}"] = {
                            "type": _type[0:-1],
                            "atoms": [*atoms],
                            "points": [*points],
                        }
                        self.logger.debug(f"Scan data for {name}#{count}:")
                        self.logger.debug(pprint.pformat(scan_data[f"{name}#{count}"]))
                        self.logger.debug("")
                    elif data["operation"] == "set":
                        # Find instance closest to first point
                        tmp = np.abs(np.array(values) - data["values"])
                        index = tmp.argmin()
                        closest = tmp.min()

                        self.logger.debug(f"{name}#{count=} -- {index} {closest}")
                        self.logger.debug(f"{data['values']=}")
                        string = " ".join([str(x + 1) for x in atoms])
                        self.logger.debug(f"{string=}")

                        set_data[f"{name}#{count}"] = {
                            "type": _type[0:-1],
                            "atoms": data["atoms"][index],
                            "value": data["values"],
                        }
                    elif data["operation"] == "freeze":
                        # Find instance closest to first point
                        tmp = np.abs(np.array(values) - data["values"])
                        index = tmp.argmin()
                        closest = tmp.min()

                        self.logger.debug(f"{name}#{count=} -- {index} {closest}")
                        self.logger.debug(f"{data['values']=}")
                        string = " ".join([str(x + 1) for x in atoms])
                        self.logger.debug(f"{string=}")

                        freeze_data[name] = {
                            "type": _type[0:-1],
                            "atoms": data["atoms"][index],
                            "value": data["values"],
                        }
                    else:
                        operation = data["operation"]
                        raise RuntimeError(
                            f"Type of constraint '{operation}' not recognized."
                        )

        return result

    def _create_energy_graph(self, scans, types, points, energies):
        """Create a simple xy plot of the energy along a scan.

        Parameters
        ----------
        points : [(float,...)]
            The points along the axes
        energies : [float]
            The energies at each point
        """
        n_dimensions = len(points)

        # Create graphs of the property

        if n_dimensions == 1:
            scan = scans[0]
            _type = types[0]
            pts = points[0]

            figure = self.create_figure(
                module_path=(self.__module__.split(".")[0], "seamm"),
                template="line.graph_template",
                title=scan,
            )
            name = scan.replace(" ", "_")
            plot = figure.add_plot(name)

            if _type == "distance":
                xlabel = "R (Å)"
                xunits = "Å"
            elif _type == "angle":
                xlabel = "\N{MATHEMATICAL ITALIC THETA SYMBOL} (º)"
                xunits = "º"
            else:
                xlabel = "\N{MATHEMATICAL ITALIC PHI SYMBOL} (º)"
                xunits = "º"

            x_axis = plot.add_axis("x", label=xlabel)
            y_axis = plot.add_axis("y", label="Energy (kJ/mol)", anchor=x_axis)
            x_axis.anchor = y_axis

            plot.add_trace(
                x_axis=x_axis,
                y_axis=y_axis,
                name=scan,
                x=pts,
                xlabel=xlabel,
                xunits=xunits,
                y=energies,
                ylabel="E",
                yunits="kJ/mol",
            )
        elif n_dimensions == 2:
            figure = self.create_figure(
                module_path=(self.__module__.split(".")[0], "seamm"),
                template="surface.graph_template",
                title="Energy Scan",
            )
            name = "EnergyScan"
            plot = figure.add_plot(name)

            # X axis
            scan = scans[0]
            _type = types[0]
            if _type == "distance":
                xlabel = f"R {scan} (Å)"
                xunits = "Å"
            elif _type == "angle":
                xlabel = f"\N{MATHEMATICAL ITALIC THETA SYMBOL} {scan} (º)"
                xunits = "º"
            else:
                xlabel = f"\N{MATHEMATICAL ITALIC PHI SYMBOL} {scan} (º)"
                xunits = "º"

            x_axis = plot.add_axis("x", label=xlabel)

            # Y axis
            scan = scans[1]
            _type = types[1]
            if _type == "distance":
                ylabel = f"R {scan} (Å)"
                yunits = "Å"
            elif _type == "angle":
                ylabel = f"\N{MATHEMATICAL ITALIC THETA SYMBOL} {scan} (º)"
                yunits = "º"
            else:
                ylabel = f"\N{MATHEMATICAL ITALIC PHI SYMBOL} {scan} (º)"
                yunits = "º"

            y_axis = plot.add_axis("y", label=ylabel)
            x_axis.anchor = y_axis

            z_axis = plot.add_axis("z", label="Energy (kJ/mol)", anchor=x_axis)

            plot.add_trace(
                x_axis=x_axis,
                y_axis=y_axis,
                z_axis=z_axis,
                name=scan,
                x=points[0],
                xlabel=xlabel,
                xunits=xunits,
                y=points[1],
                ylabel=ylabel,
                yunits=yunits,
                z=energies,
                zlabel="E",
                zunits="kJ/mol",
            )
        else:
            self.logger.warning("Energy Scan can't graph more than 2 dimensions")

        figure.grid_plots(name)

        node_path = Path(self.directory)
        node_path.mkdir(parents=True, exist_ok=True)
        path = node_path / "EnergyScan.graph"

        figure.dump(path)

        if "html" in self.options and self.options["html"]:
            if n_dimensions == 1:
                figure.template = "line.html_template"
            elif n_dimensions == 2:
                figure.template = "surface.html_template"

            figure.dump(path.with_suffix(".html"))

    def _create_sdf(self, configurations):
        """Write the configurations to an SDF file.

        Parameters
        ----------
        configurations : molsystem._Configuration
            The configurations to write.
        """
        read_structure_step.write(
            str(Path(self.directory) / "EnergyScan.sdf"),
            configurations,
            extension=".sdf",
            remove_hydrogens=False,
            printer=printer.important,
            references=self.references,
            bibliography=self._bibliography,
        )

    def _find_closest_configuration(self, pts, all_points, all_data):
        """Return the "closest" configuration to the given points.

        This is used to get the point in the scans that is closest to the
        current point, so that the configuration can be used as the starting
        cofiguration for the next calculation.

        Parameters
        ----------
        pts : [float]
            The points to find the closest configuration to.

        all_points : [[float], ...]
            The total set of points, sorted, with one list per scan direction.

        all_data : {(pts,...): {str: any}}
            The data from the calculations so far.

        Returns
        -------
        pts : (pts...)
            The tuple of the closest point.
        """
        if len(all_points) == 0:
            return None

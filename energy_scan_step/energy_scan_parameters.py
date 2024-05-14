# -*- coding: utf-8 -*-
"""
Control parameters for the Energy Scan step in a SEAMM flowchart
"""
import logging

import seamm

logger = logging.getLogger(__name__)


class EnergyScanParameters(seamm.Parameters):
    """
    The control parameters for Energy Scan.

    You need to replace the "time" entry in dictionary below these comments with the
    definitions of parameters to control this step. The keys are parameters for the
    current plugin,the values are dictionaries as outlined below.

    Examples
    --------
    ::

        parameters = {
            "time": {
                "default": 100.0,
                "kind": "float",
                "default_units": "ps",
                "enumeration": tuple(),
                "format_string": ".1f",
                "description": "Simulation time:",
                "help_text": ("The time to simulate in the dynamics run.")
            },
        }

    parameters : {str: {str: str}}
        A dictionary containing the parameters for the current step.
        Each key of the dictionary is a dictionary that contains the
        the following keys:

    parameters["default"] :
        The default value of the parameter, used to reset it.

    parameters["kind"] : enum()
        Specifies the kind of a variable. One of  "integer", "float", "string",
        "boolean", or "enum"

        While the "kind" of a variable might be a numeric value, it may still have
        enumerated custom values meaningful to the user. For instance, if the parameter
        is a convergence criterion for an optimizer, custom values like "normal",
        "precise", etc, might be adequate. In addition, any parameter can be set to a
        variable of expression, indicated by having "$" as the first character in the
        field. For example, $OPTIMIZER_CONV.

    parameters["default_units"] : str
        The default units, used for resetting the value.

    parameters["enumeration"]: tuple
        A tuple of enumerated values.

    parameters["format_string"]: str
        A format string for "pretty" output.

    parameters["description"]: str
        A short string used as a prompt in the GUI.

    parameters["help_text"]: str
        A longer string to display as help for the user.

    See Also
    --------
    EnergyScan, TkEnergyScan, EnergyScanParameters,
    EnergyScanStep
    """

    parameters = {
        "coordinate system": {
            "default": "TRIC: translation-rotation internal coordinates",
            "kind": "integer",
            "default_units": "",
            "enumeration": (
                "TRIC: translation-rotation internal coordinates",
                "TRIC-p: primitive (redundant) TRIC",
                "Cart: Cartesian coordinates",
                "DLC: delocalized internal coordinates",
                "HDLC: hybrid delocalized internal coordinates",
            ),
            "format_string": "",
            "description": "The coordinate system to use:",
            "help_text": "The coordinates system to use in the calculation.",
        },
        "max steps": {
            "default": "default",
            "kind": "integer",
            "default_units": "",
            "enumeration": ("default",),
            "format_string": "",
            "description": "Maximum number of optimization steps:",
            "help_text": (
                "The maximum number of optimization steps to take for any point in the "
                "scan."
            ),
        },
        "enforce": {
            "default": "5.0",
            "kind": "float",
            "default_units": "degree",
            "enumeration": tuple(),
            "format_string": "%.1f",
            "description": "Exactly enforce constraints:",
            "help_text": (
                "The maximum value of the deviation in degrees/angstroms at which to "
                "start exactly enforcing the constraints."
            ),
        },
        "constraints": {
            "default": {},
            "kind": "dictionary",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Coordinates to scan:",
            "help_text": "The coordinates to scan",
        },
    }

    def __init__(self, defaults={}, data=None):
        """
        Initialize the parameters, by default with the parameters defined above

        Parameters
        ----------
        defaults: dict
            A dictionary of parameters to initialize. The parameters
            above are used first and any given will override/add to them.
        data: dict
            A dictionary of keys and a subdictionary with value and units
            for updating the current, default values.

        Returns
        -------
        None
        """

        logger.debug("EnergyScanParameters.__init__")

        super().__init__(
            defaults={**EnergyScanParameters.parameters, **defaults}, data=data
        )

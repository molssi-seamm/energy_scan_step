# -*- coding: utf-8 -*-

"""The graphical part of a Energy Scan step"""

import pprint  # noqa: F401
import tkinter as tk
import tkinter.ttk as ttk

import Pmw

from .energy_scan_parameters import EnergyScanParameters
import seamm
from seamm_util import ureg, Q_, units_class  # noqa: F401
import seamm_widgets as sw


class TkEnergyScan(seamm.TkNode):
    """
    The graphical part of a Energy Scan step in a flowchart.

    Attributes
    ----------
    tk_flowchart : TkFlowchart = None
        The flowchart that we belong to.
    node : Node = None
        The corresponding node of the non-graphical flowchart
    canvas: tkCanvas = None
        The Tk Canvas to draw on
    dialog : Dialog
        The Pmw dialog object
    x : int = None
        The x-coordinate of the center of the picture of the node
    y : int = None
        The y-coordinate of the center of the picture of the node
    w : int = 200
        The width in pixels of the picture of the node
    h : int = 50
        The height in pixels of the picture of the node
    self[widget] : dict
        A dictionary of tk widgets built using the information
        contained in Energy Scan_parameters.py

    See Also
    --------
    EnergyScan, TkEnergyScan,
    EnergyScanParameters,
    """

    def __init__(
        self,
        tk_flowchart=None,
        node=None,
        namespace="org.molssi.seamm.tk",
        canvas=None,
        x=None,
        y=None,
        w=200,
        h=50,
    ):
        """
        Initialize a graphical node.

        Parameters
        ----------
        tk_flowchart: Tk_Flowchart
            The graphical flowchart that we are in.
        node: Node
            The non-graphical node for this step.
        namespace: str
            The stevedore namespace for finding sub-nodes.
        canvas: Canvas
           The Tk canvas to draw on.
        x: float
            The x position of the nodes center on the canvas.
        y: float
            The y position of the nodes cetner on the canvas.
        w: float
            The nodes graphical width, in pixels.
        h: float
            The nodes graphical height, in pixels.

        Returns
        -------
        None
        """
        self.namespace = namespace
        self.dialog = None

        super().__init__(
            tk_flowchart=tk_flowchart,
            node=node,
            canvas=canvas,
            x=x,
            y=y,
            w=w,
            h=h,
        )
        # Variables for handle the table of constraints
        self._constraints = {}  # temporary copy when editing
        self._new_constraint_dialog = None
        self._new = {}  # Widgets for the new constraints dialog
        self._edit_constraint_dialog = None
        self._edit = {}  # Widgets for the edit constraints dialog

        self.create_dialog()

    def create_dialog(self):
        """
        Create the dialog. A set of widgets will be chosen by default
        based on what is specified in the Energy Scan_parameters
        module.

        Parameters
        ----------
        None

        Returns
        -------
        None

        See Also
        --------
        TkEnergyScan.reset_dialog
        """

        frame = super().create_dialog(title="Energy Scan", widget="notebook")
        # make it large!
        screen_w = self.dialog.winfo_screenwidth()
        screen_h = self.dialog.winfo_screenheight()
        w = int(0.9 * screen_w)
        h = int(0.8 * screen_h)
        x = int(0.05 * screen_w / 2)
        y = int(0.1 * screen_h / 2)

        self.dialog.geometry(f"{w}x{h}+{x}+{y}")

        # Add a frame for the flowchart
        notebook = self["notebook"]
        flowchart_frame = ttk.Frame(notebook)
        self["flowchart frame"] = flowchart_frame
        notebook.add(flowchart_frame, text="Flowchart", sticky=tk.NSEW)

        self.tk_subflowchart = seamm.TkFlowchart(
            master=flowchart_frame,
            flowchart=self.node.subflowchart,
            namespace=self.namespace,
        )
        self.tk_subflowchart.draw()

        # Fill in the control parameters
        # Shortcut for parameters
        P = self.node.parameters

        # frame to isolate widgets
        frame = self["scan frame"] = ttk.LabelFrame(
            self["frame"],
            borderwidth=4,
            relief="sunken",
            text="Energy Scan Parameters",
            labelanchor="n",
            padding=10,
        )

        for key in EnergyScanParameters.parameters:
            self[key] = P[key].widget(frame)

        # frame to isolate constraints
        frame = self["constraint frame"] = ttk.LabelFrame(
            self["frame"],
            borderwidth=4,
            relief="sunken",
            text="Constraints",
            labelanchor="n",
            padding=10,
        )
        self["constraints"] = sw.ScrolledColumns(
            frame,
            columns=[
                "",
                "",
                "Name",
                "Operation",
                "Type",
                "SMARTS",
                "Which?",
                "Value(s)",
                "Units",
                "Scan Type",
                "Direction",
            ],
        )
        self["constraints"].grid(row=0, column=0, sticky=tk.NSEW)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # and binding to change as needed
        # self["approach"].combobox.bind("<<ComboboxSelected>>", self.reset_frame)

        # and lay them out
        self.reset_dialog()

    def edit(self):
        """Present a dialog for editing the Control Parameters input

        Parameters
        ----------
        None

        Returns
        -------
        None

        See Also
        --------
        TkControlParameters.right_click
        """

        if self.dialog is None:
            self.create_dialog()

        P = self.node.parameters
        self._constraints = P["constraints"].value

        self.reset_dialog()

        self.dialog.activate(geometry="centerscreenfirst")

    def handle_dialog(self, result):
        """Handle the closing of the edit dialog

        What to do depends on the button used to close the dialog. If
        the user closes it by clicking the 'x' of the dialog window,
        None is returned, which we take as equivalent to cancel.

        Parameters
        ----------
        result : None or str
            The value of this variable depends on what the button
            the user clicked.

        Returns
        -------
        None
        """
        if result is None or result == "Cancel":
            self.dialog.deactivate(result)
            self._constraints = {}
            return

        if result == "Help":
            # display help!!!
            return

        if result != "OK":
            self.dialog.deactivate(result)
            raise RuntimeError("Don't recognize dialog result '{}'".format(result))

        self.dialog.deactivate(result)
        # Shortcut for parameters
        P = self.node.parameters

        P["constraints"].value = self._constraints

        self._constraints = {}

        for key in P:
            if key != "constraints":
                P[key].set_from_widget()

    def reset_dialog(self, widget=None):
        """Layout the widgets in the dialog.

        The widgets are chosen by default from the information in
        energy scan parameters.

        This function simply lays them out row by row with
        aligned labels. You may wish a more complicated layout that
        is controlled by values of some of the control parameters.
        If so, edit or override this method

        Parameters
        ----------
        widget : Tk Widget = None

        Returns
        -------
        None

        See Also
        --------
        TkEnergyScan.create_dialog
        """

        frame = self["frame"]
        row = 0

        self["scan frame"].grid(row=row, column=0, sticky=tk.EW, pady=10)
        row += 1

        self["constraint frame"].grid(row=row, column=0, sticky=tk.EW, pady=10)
        frame.rowconfigure(row, weight=1)
        row += 1

        frame.columnconfigure(0, weight=1)

        self.reset_scan_frame()
        self.reset_constraints()

        return row

    def reset_scan_frame(self, widget=None):
        """Layout the widgets in the scan frame
        as needed for the current state"""

        frame = self["scan frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()

        row = 0

        # Main controls
        widgets = []
        widgets0 = []
        for key in ("coordinate system", "max steps", "enforce"):
            self[key].grid(row=row, column=0, columnspan=2, sticky=tk.W)
            widgets.append(self[key])
            row += 1

        self["enforce"].set_units(values=("degree", "Å", "pm", "bohr"))

        sw.align_labels(widgets, sticky=tk.E)
        if len(widgets0) > 0:
            sw.align_labels(widgets0, sticky=tk.E)

        frame.columnconfigure(0, minsize=50)

    def reset_constraints(self, widget=None):
        """Re-initialize the constraints."""
        table = self["constraints"]

        frame = table.table.interior()
        table.clear()

        row = 0
        for name, data in self._constraints.items():
            table[row, 0] = ttk.Button(
                frame,
                text="-",
                width=5,
                command=lambda nm=name: self._remove_constraint(nm),
                takefocus=True,
            )
            table[row, 1] = ttk.Button(
                frame,
                text="Edit",
                width=5,
                command=lambda nm=name: self._edit_constraint(nm),
                takefocus=True,
            )
            table[row, 2] = name
            table[row, 3] = data["operation"]
            table[row, 4] = data["type"]
            table[row, 5] = data["SMARTS"]
            table[row, 6] = data["which"]
            table[row, 7] = data["values"]
            table[row, 8] = data["units"]
            if data["operation"] == "scan":
                table[row, 9] = data["scan type"]
                table[row, 10] = data["direction"]
            else:
                table[row, 9] = ""
                table[row, 10] = ""
            row += 1

        # a button to add new constraints...
        table[row, 0] = ttk.Button(
            frame, text="+", width=5, command=self._add_constraint, takefocus=True
        )

        table.update_idletasks()

    def right_click(self, event):
        """
        Handles the right click event on the node.

        Parameters
        ----------
        event : Tk Event

        Returns
        -------
        None

        See Also
        --------
        TkEnergyScan.edit
        """

        super().right_click(event)
        self.popup_menu.add_command(label="Edit...", command=self.edit)

        self.popup_menu.tk_popup(event.x_root, event.y_root, 0)

    def _add_constraint(self):
        """Add a new constraint to the table."""
        if self._new_constraint_dialog is None:
            self._create_new_constraint_dialog()

        self._new["name"].set("C-C dihedral")
        self._new["operation"].set("scan")
        self._new["type"].set("dihedral")
        self._new["SMARTS"].set("[H:1][C:2][C:3][H:4]")
        self._new["which"].set("first")
        self._new["values"].set("0.0:180.0:10")
        self._new["units"].set("degree")
        self._new["scan type"].set("multidimensional")
        self._new["direction"].set("from current")

        # and lay them out
        self._reset_new_constraint_dialog()

        self._new_constraint_dialog.activate(geometry="centerscreenfirst")

    def _create_new_constraint_dialog(self):
        """
        Create a dialog for adding new constraints.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self._new_constraint_dialog is not None:
            return

        dialog = self._new_constraint_dialog = Pmw.Dialog(
            self.dialog.interior(),
            buttons=("OK", "Cancel"),
            defaultbutton="OK",
            title="Add Constraint",
            command=self._handle_new_constraint_dialog,
        )
        self._new_constraint_dialog.withdraw()

        # Create a frame to hold everything in the dialog
        frame = self._new["frame"] = ttk.Frame(dialog.interior())
        frame.pack(expand=tk.YES, fill=tk.BOTH)

        # Then create the widgets
        self._new["name"] = sw.LabeledEntry(frame, labeltext="Name")
        self._new["operation"] = sw.LabeledCombobox(
            frame,
            labeltext="Operation:",
            values=(
                "freeze",
                "set",
                "scan",
            ),
            state="readonly",
        )
        self._new["type"] = sw.LabeledCombobox(
            frame,
            labeltext="Type:",
            values=(
                "distance",
                "angle",
                "dihedral",
            ),
            state="readonly",
        )
        self._new["SMARTS"] = sw.LabeledEntry(frame, labeltext="SMARTS")
        self._new["which"] = sw.LabeledCombobox(
            frame,
            labeltext="Which?:",
            values=("all", "first"),
        )
        self._new["values"] = sw.LabeledEntry(frame, labeltext="Value 1")
        self._new["units"] = sw.LabeledCombobox(
            frame,
            labeltext="Units:",
            values=("degree", "Å", "pm", "bohr", "steps"),
            state="readonly",
        )
        self._new["scan type"] = sw.LabeledCombobox(
            frame,
            labeltext="Scan type:",
            values=("multidimensional", "sequential", "lockstep"),
            state="readonly",
        )
        self._new["direction"] = sw.LabeledCombobox(
            frame,
            labeltext="Direction:",
            values=("increasing", "decreasing", "from current"),
            state="readonly",
        )

        self._new["operation"].combobox.bind(
            "<<ComboboxSelected>>", self._reset_new_constraint_dialog
        )
        self._new["type"].combobox.bind(
            "<<ComboboxSelected>>", self._reset_new_constraint_dialog
        )

        # and lay them out
        self._reset_new_constraint_dialog()

    def _reset_new_constraint_dialog(self, widget=None):
        """Lay the dialog out based on the contents."""
        # Remove any widgets previously packed
        frame = self._new["frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()

        operation = self._new["operation"].get()
        _type = self._new["type"].get()

        row = 0
        widgets = []

        for key in ("name", "operation", "type", "SMARTS", "which", "values", "units"):
            self._new[key].grid(row=row, column=0, sticky=tk.EW)
            widgets.append(self._new[key])
            row += 1

        if operation == "scan":
            for key in ("scan type", "direction"):
                self._new[key].grid(row=row, column=0, sticky=tk.EW)
                widgets.append(self._new[key])
                row += 1

        units = self._new["units"].get()
        if _type == "distance":
            self._new["units"].config(values=("Å", "pm", "bohr", "steps"))
            if units not in ("Å", "pm", "bohr", "steps"):
                self._new["units"].set("Å")
        else:
            self._new["units"].config(values=("degree", "steps"))
            if units not in ("degree", "steps"):
                self._new["units"].set("degree")

        sw.align_labels(widgets)

    def _handle_new_constraint_dialog(self, result):
        """Handle the closing of the new constraint dialog

        What to do depends on the button used to close the dialog. If
        the user closes it by clicking the 'x' of the dialog window,
        None is returned, which we take as equivalent to cancel.

        Parameters
        ----------
        result : None or str
            The value of this constraint depends on what the button
            the user clicked.

        Returns
        -------
        None
        """

        if result is None or result == "Cancel":
            self._new_constraint_dialog.deactivate(result)
            return

        if result != "OK":
            self._new_constraint_dialog.deactivate(result)
            raise RuntimeError(
                f"Don't recognize new constraint dialog result '{result}'"
            )

        self._new_constraint_dialog.deactivate(result)

        name = self._new["name"].get()
        if name in self._constraints:
            raise KeyError(f"Duplicate name: '{name}'")

        data = self._constraints[name] = {}
        for key, w in self._new.items():
            if key not in ("frame",):
                data[key] = w.get()

        self.reset_constraints()

    def _edit_constraint(self, name):
        """Edit the values associated with a constraint."""
        # Post dialog to fill out the new constraint
        if self._edit_constraint_dialog is None:
            self._create_edit_constraint_dialog()

        self._edit_constraint_dialog.configure(
            command=lambda result, nm=name: self._handle_edit_constraint_dialog(
                nm, result
            )  # noqa: E501
        )

        data = self._constraints[name]
        for key, w in self._edit.items():
            if key == "name":
                w.set(name)
            elif key != "frame":
                w.set(data[key])

        # and lay them out
        self._reset_edit_constraint_dialog()

        self._edit_constraint_dialog.activate(geometry="centerscreenfirst")

    def _create_edit_constraint_dialog(self):
        """
        Create a dialog for adding edit constraints.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self._edit_constraint_dialog is not None:
            return

        dialog = self._edit_constraint_dialog = Pmw.Dialog(
            self.dialog.interior(),
            buttons=("OK", "Cancel"),
            defaultbutton="OK",
            title="Edit Constraint",
            command=lambda: self._handle_edit_constraint_dialog,
        )
        self._edit_constraint_dialog.withdraw()

        # Create a frame to hold everything in the dialog
        frame = self._edit["frame"] = ttk.Frame(dialog.interior())
        frame.pack(expand=tk.YES, fill=tk.BOTH)

        # Then create the widgets
        self._edit["name"] = sw.LabeledEntry(frame, labeltext="Name")
        self._edit["operation"] = sw.LabeledCombobox(
            frame,
            labeltext="Type:",
            values=(
                "freeze",
                "set",
                "scan",
            ),
            state="readonly",
        )
        self._edit["type"] = sw.LabeledCombobox(
            frame,
            labeltext="Type:",
            values=(
                "distance",
                "angle",
                "dihedral",
            ),
            state="readonly",
        )
        self._edit["SMARTS"] = sw.LabeledEntry(frame, labeltext="SMARTS")
        self._edit["which"] = sw.LabeledCombobox(
            frame,
            labeltext="Which?:",
            values=("all", "first"),
        )
        self._edit["values"] = sw.LabeledEntry(frame, labeltext="Value 1")
        self._edit["units"] = sw.LabeledCombobox(
            frame,
            labeltext="Units:",
            values=("º", "Å", "steps"),
            state="readonly",
        )
        self._edit["scan type"] = sw.LabeledCombobox(
            frame,
            labeltext="Scan type:",
            values=("multidimensional", "sequential", "lockstep"),
            state="readonly",
        )
        self._edit["direction"] = sw.LabeledCombobox(
            frame,
            labeltext="Direction:",
            values=("increasing", "decreasing", "from current"),
            state="readonly",
        )

        self._edit["operation"].combobox.bind(
            "<<ComboboxSelected>>", self._reset_edit_constraint_dialog
        )
        self._edit["type"].combobox.bind(
            "<<ComboboxSelected>>", self._reset_edit_constraint_dialog
        )

    def _reset_edit_constraint_dialog(self, widget=None):
        """Lay the dialog out based on the contents."""
        # Remove any widgets previously packed
        frame = self._edit["frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()

        operation = self._edit["operation"].get()
        _type = self._edit["type"].get()

        row = 0
        widgets = []

        for key in ("name", "operation", "type", "SMARTS", "which", "values", "units"):
            self._edit[key].grid(row=row, column=0, sticky=tk.EW)
            widgets.append(self._edit[key])
            row += 1

        if operation == "scan":
            for key in ("scan type", "direction"):
                self._edit[key].grid(row=row, column=0, sticky=tk.EW)
                widgets.append(self._edit[key])
                row += 1

        units = self._edit["units"].get()
        if _type == "distance":
            self._edit["units"].config(values=("Å", "pm", "bohr", "steps"))
            if units not in ("Å", "pm", "bohr", "steps"):
                self._edit["units"].set("Å")
        else:
            self._edit["units"].config(values=("degree", "steps"))
            if units not in ("degree", "steps"):
                self._edit["units"].set("degree")

        sw.align_labels(widgets)

    def _handle_edit_constraint_dialog(self, name, result):
        """Handle the closing of the edit constraint dialog

        What to do depends on the button used to close the dialog. If
        the user closes it by clicking the 'x' of the dialog window,
        None is returned, which we take as equivalent to cancel.

        Parameters
        ----------
        result : None or str
            The value of this constraint depends on what the button
            the user clicked.

        Returns
        -------
        None
        """

        if result is None or result == "Cancel":
            self._edit_constraint_dialog.deactivate(result)
            return

        if result != "OK":
            self._edit_constraint_dialog.deactivate(result)
            raise RuntimeError(
                f"Don't recognize edit constraint dialog result '{result}'"
            )

        self._edit_constraint_dialog.deactivate(result)

        new_name = self._edit["name"].get().lstrip("-")
        if new_name == name:
            data = self._constraints[name]
        else:
            del self._constraints[name]
            name = new_name
            data = self._constraints[name] = {}

        for key, w in self._edit.items():
            if key not in ("frame", "name"):
                data[key] = w.get()

        self.reset_constraints()

    def _remove_constraint(self, name):
        """Remove a constraint."""
        del self._constraints[name]
        self.reset_constraints()

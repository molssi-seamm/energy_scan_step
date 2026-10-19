#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `energy_scan_step` package."""

import pytest  # noqa: F401
import energy_scan_step  # noqa: F401


def test_construction():
    """Just create an object and test its type."""
    result = energy_scan_step.EnergyScan()
    assert str(type(result)) == "<class 'energy_scan_step.energy_scan.EnergyScan'>"


def _molecule(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    return mol, mol.GetConformer()


def _ring_atoms(mol):
    """O, C, C, O of the ring of ethylene carbonate: the CH2 carbons and the ring
    oxygen bonded to each."""
    ring = [r for r in mol.GetRingInfo().AtomRings() if len(r) == 5][0]
    ch2 = [
        a
        for a in ring
        if mol.GetAtomWithIdx(a).GetSymbol() == "C"
        and mol.GetAtomWithIdx(a).GetTotalNumHs(includeNeighbors=True) == 2
    ]
    result = []
    for c in ch2:
        o = [
            n.GetIdx()
            for n in mol.GetAtomWithIdx(c).GetNeighbors()
            if n.GetSymbol() == "O" and n.GetIdx() in ring
        ]
        result.append((o[0], c))
    (oa, c1), (ob, c2) = result
    return [oa, c1, c2, ob]


def test_dihedral_in_a_ring_is_left_to_the_optimizer():
    """RDKit cannot set a dihedral about a ring bond (the pucker of ethylene
    carbonate), so geomeTRIC is asked to drive it to the value ($set)."""
    import energy_scan_step

    node = energy_scan_step.EnergyScan()
    node._id = ("1",)
    mol, conf = _molecule("O=C1OCCO1")
    atoms = _ring_atoms(mol)
    frozen, to_set = [], []
    node._set_coordinate(mol, conf, "dihedral", atoms, 20.0, frozen, to_set)
    atom_string = " ".join(str(a + 1) for a in atoms)
    assert frozen == []
    assert to_set == [f"dihedral {atom_string} 20.0"]


def test_dihedral_in_a_chain_is_set_and_frozen():
    from rdkit.Chem import rdMolTransforms

    import energy_scan_step

    node = energy_scan_step.EnergyScan()
    node._id = ("1",)
    mol, conf = _molecule("CCCC")
    frozen, to_set = [], []
    node._set_coordinate(mol, conf, "dihedral", [0, 1, 2, 3], 60.0, frozen, to_set)
    assert to_set == [] and frozen == ["dihedral 1 2 3 4"]
    assert abs(rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3) - 60.0) < 1e-6

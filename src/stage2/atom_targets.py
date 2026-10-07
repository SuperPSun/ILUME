"""Compatibility exports for the unchanged Stage2 atom-mapping contract."""
from common.atom_targets import (
    AtomMappingResult, Mol2Atom, Mol2Bond, Mol2Graph,
    PARTIAL_CHARGE_MAPPING_CONTRACT, StructureManifestEntry,
    load_structure_manifest, load_verify_parse_and_map, map_partial_charges,
    parse_mol2, parse_mol2_text, verify_structure,
)

__all__ = [
    "AtomMappingResult", "Mol2Atom", "Mol2Bond", "Mol2Graph",
    "PARTIAL_CHARGE_MAPPING_CONTRACT", "StructureManifestEntry",
    "load_structure_manifest", "load_verify_parse_and_map", "map_partial_charges",
    "parse_mol2", "parse_mol2_text", "verify_structure",
]

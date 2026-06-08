"""Unified label space and cross-dataset category mapping.

Single source of truth: `unified_class_schema.json` (shipped alongside this
module). The model is trained in the **MOCS 13-class label space** (category_id
1..13). External predictions and GT are remapped into that space at eval time;
only shared classes are scored.

Resolution is by category **name** (normalised), not by id, so it is robust to
any id differences between the schema and your actual json files. External
categories whose unified class has no MOCS counterpart (e.g. CIS PC / PC-truck,
ACID backhoe_loader / grader) resolve to None and are dropped from eval.

ExtCon is not listed in the schema; it shares all 13 MOCS classes (a few via
synonyms), handled by EXTCON_SYNONYMS below.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "unified_class_schema.json")

# ExtCon synonym names -> canonical MOCS name (ExtCon is not in the schema).
EXTCON_SYNONYMS: Dict[str, str] = {
    "tower crane": "Static crane",
    "hanging hook": "Hanging head",
    "vehicle crane": "Crane",
    "pile driver": "Pile driving",
}


def _norm(name: str) -> str:
    """Lowercase, collapse non-alphanumerics to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _load_schema(path: str = _SCHEMA_PATH) -> List[dict]:
    with open(path) as f:
        return json.load(f)


# Fallback MOCS schema if the json is missing (keeps the package importable).
_FALLBACK_MOCS = [
    "Worker", "Static crane", "Hanging head", "Crane", "Roller", "Bulldozer",
    "Excavator", "Truck", "Loader", "Pump truck", "Concrete mixer",
    "Pile driving", "Other vehicle",
]


def _build_tables(schema: Optional[List[dict]]):
    """Return (mocs_id_to_name, name_to_mocs_id)."""
    mocs_id_to_name: Dict[int, str] = {}
    name_to_mocs_id: Dict[str, int] = {}

    if schema is None:
        for i, n in enumerate(_FALLBACK_MOCS):
            mocs_id_to_name[i + 1] = n
            name_to_mocs_id[_norm(n)] = i + 1
    else:
        for e in schema:
            mocs = e.get("MOCS")
            if not mocs:
                continue  # unified class with no MOCS counterpart -> not scored
            mid, mname = mocs["id"], mocs["name"]
            mocs_id_to_name[mid] = mname
            name_to_mocs_id[_norm(mname)] = mid          # MOCS name -> itself
            for ds in ("CIS", "ACID", "SODA"):
                v = e.get(ds)
                if not v:
                    continue
                items = v if isinstance(v, list) else [v]
                for it in items:
                    name_to_mocs_id[_norm(it["name"])] = mid

    # ExtCon synonyms (ExtCon not in schema)
    name_lookup = {_norm(n): i for i, n in mocs_id_to_name.items()}
    for syn, canon in EXTCON_SYNONYMS.items():
        if _norm(canon) in name_lookup:
            name_to_mocs_id[_norm(syn)] = name_lookup[_norm(canon)]

    return mocs_id_to_name, name_to_mocs_id


try:
    _SCHEMA = _load_schema()
except Exception:                                        # noqa: BLE001
    _SCHEMA = None

MOCS_ID_TO_NAME, _NAME_TO_MOCS_ID = _build_tables(_SCHEMA)
MOCS_CLASSES: List[str] = [MOCS_ID_TO_NAME[i] for i in sorted(MOCS_ID_TO_NAME)]
NUM_CLASSES = len(MOCS_CLASSES)                          # 13 foreground
MOCS_NAME_TO_ID: Dict[str, int] = {n: i for i, n in MOCS_ID_TO_NAME.items()}


def resolve_to_mocs(name: str) -> Optional[str]:
    """Resolve an external category name to a canonical MOCS class name (or None)."""
    mid = _NAME_TO_MOCS_ID.get(_norm(name))
    return MOCS_ID_TO_NAME.get(mid) if mid is not None else None


def build_external_to_mocs(categories: List[dict]) -> Dict[int, int]:
    """Given an external COCO `categories` list, build {ext_cat_id -> mocs_id}.

    Resolves by name; unmapped categories are omitted.
    """
    mapping: Dict[int, int] = {}
    for c in categories:
        mid = _NAME_TO_MOCS_ID.get(_norm(c["name"]))
        if mid is not None:
            mapping[c["id"]] = mid
    return mapping


def shared_mocs_ids(categories: List[dict]) -> List[int]:
    """Sorted set of MOCS class ids an external dataset covers (deduplicated)."""
    return sorted(set(build_external_to_mocs(categories).values()))

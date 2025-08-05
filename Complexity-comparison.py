# OCEL Complexity Comparison Module
# ----------------------------------
# This module analyzes and compares the structural complexity of OCEL (Object-Centric Event Logs)
# using original, relation-based sublogs, and flattened logs for each object type.

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import pm4py


def compute_ocel_complexity(ocel, alpha=0.5, verbose=False):
    """
    Compute the structural complexity of an OCEL log based on local diversity and global connectivity.

    Parameters:
    - ocel: OCEL object (with events, objects, and relations as pandas DataFrames)
    - alpha: Weighting factor between local and global complexity
    - verbose: Whether to print internal details

    Returns:
    - Complexity score (float between 0 and 1)
    """
    events_df = ocel.events.copy()
    objects_df = ocel.objects.copy()
    relations_df = ocel.relations.copy()

    OT = set(objects_df["ocel:type"].unique())
    ET = set(events_df["ocel:activity"].unique())

    obj_id_to_type = dict(zip(objects_df["ocel:oid"], objects_df["ocel:type"]))
    eid_to_activity = dict(zip(events_df["ocel:eid"], events_df["ocel:activity"]))

    E2O = defaultdict(list)
    for _, row in relations_df.iterrows():
        E2O[row["ocel:eid"]].append(row["ocel:oid"])

    E_ot = defaultdict(set)
    eventtypes_ot = defaultdict(set)
    coobjecttypes_ot = defaultdict(set)

    for e, oids in E2O.items():
        etype = eid_to_activity.get(e)
        involved_types = set(obj_id_to_type[oid] for oid in oids if oid in obj_id_to_type)
        for ot in involved_types:
            E_ot[ot].add(e)
            eventtypes_ot[ot].add(etype)
            coobjecttypes_ot[ot].update(t for t in involved_types if t != ot)

    total_events = sum(len(E_ot[ot]) for ot in OT)
    weights = {ot: len(E_ot[ot]) / total_events if total_events else 0.0 for ot in OT}

    local_components = {}
    for ot in OT:
        event_diversity = len(eventtypes_ot[ot]) / len(ET) if len(ET) > 0 else 0
        coobject_diversity = len(coobjecttypes_ot[ot]) / len(OT) if len(OT) > 0 else 0
        local_components[ot] = weights[ot] * (event_diversity + coobject_diversity)

    local_score = sum(local_components.values()) / 2.0

    G = nx.Graph()
    G.add_nodes_from(OT)
    for oids in E2O.values():
        involved_types = list(set(obj_id_to_type[oid] for oid in oids if oid in obj_id_to_type))
        for i in range(len(involved_types)):
            for j in range(i + 1, len(involved_types)):
                if involved_types[i] != involved_types[j]:
                    G.add_edge(involved_types[i], involved_types[j])

    rho = (2 * G.number_of_edges()) / (len(OT) * (len(OT) - 1)) if len(OT) > 1 else 0

    if verbose:
        print("Local score (normalized):", local_score)
        print("Graph density (rho):", rho)
        print("Local components per object type:", local_components)

    complexity = (1 - alpha) * local_score + alpha * rho
    return 1.0 - complexity


def compare_ocel_complexities(ocel, alpha=0.5):
    """
    Compare OCEL complexity score across:
    - Original OCEL log
    - Sublogs based on object types
    - Flattened logs using PM4Py

    Returns:
    - Dictionary with all complexity scores
    """
    original = compute_ocel_complexity(ocel, alpha, verbose=True)

    object_types = ocel.objects["ocel:type"].unique()
    obj_id_to_type = dict(zip(ocel.objects["ocel:oid"], ocel.objects["ocel:type"]))

    relation_scores = {}
    flattened_scores = {}

    for ot in object_types:
        try:
            flat = pm4py.ocel_flattening(ocel, ot)
            act_col = next((c for c in ["ocel:activity", "activity", "concept:name"] if c in flat.columns), None)
            case_col = next((c for c in ["ocel:oid", "case:concept:name"] if c in flat.columns), None)
            if act_col and case_col and not flat.empty:
                event_types = flat[act_col].nunique()
                object_ids = flat[case_col].nunique()
                flat_score = (event_types + object_ids) / (2 * max(event_types, object_ids, 1))
            else:
                flat_score = 0.0
        except:
            flat_score = 0.0
        flattened_scores[ot] = 1.0 - flat_score

        sub_eids = ocel.relations[ocel.relations["ocel:oid"].map(obj_id_to_type.get) == ot]["ocel:eid"].unique()
        sub_events = ocel.events[ocel.events["ocel:eid"].isin(sub_eids)]
        sub_rel = ocel.relations[ocel.relations["ocel:eid"].isin(sub_eids)]
        sub_oids = sub_rel["ocel:oid"].unique()
        sub_objs = ocel.objects[ocel.objects["ocel:oid"].isin(sub_oids)]

        class SubOCEL:
            def __init__(self, events, objects, relations):
                self.events = events
                self.objects = objects
                self.relations = relations

        sublog = SubOCEL(sub_events, sub_objs, sub_rel)
        relation_score = compute_ocel_complexity(sublog, alpha)
        relation_scores[ot] = min(max(relation_score, 0.0), 1.0)

    # Plotting comparison
    plt.figure(figsize=(12, 6))
    sorted_ots = sorted(object_types)
    rel_vals = [relation_scores[ot] for ot in sorted_ots]
    flat_vals = [flattened_scores[ot] for ot in sorted_ots]

    plt.plot(sorted_ots, rel_vals, marker='o', label='Relation-Based Sub OCELs')
    plt.plot(sorted_ots, flat_vals, marker='s', linestyle='--', label='Flattened')
    plt.axhline(original, color='red', linestyle=':', label='Original OCEL')
    plt.title("OCEL Complexity Comparison")
    plt.xlabel("Object Type")
    plt.ylabel("Complexity Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return {
        "original": original,
        "relation_based": relation_scores,
        "flattened": flattened_scores
    }

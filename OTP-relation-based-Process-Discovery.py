import pm4py
from pm4py.objects.ocel.obj import OCEL
from collections import defaultdict

def discover_object_perspective_models(ocel: OCEL):
    """
    Discover process models for each object type perspective in the OCEL log.

    Parameters:
    ocel (OCEL): An object-centric event log in OCEL format.

    Returns:
    dict: A dictionary where keys are object types and values are the corresponding discovered models.
    """
    # Copy OCEL dataframes
    events_df = ocel.events.copy()
    objects_df = ocel.objects.copy()
    objmap_df = ocel.relations.copy()

    # Define standard OCEL column names
    event_id_col = "ocel:eid"
    object_id_col = "ocel:oid"
    activity_col = "ocel:activity"
    object_type_col = "ocel:type"

    # Create lookup from object ID to its type
    obj_id_to_type = dict(zip(objects_df[object_id_col], objects_df[object_type_col]))

    # Map each event to associated object IDs
    event_to_objects = defaultdict(list)
    for _, row in objmap_df.iterrows():
        event_to_objects[row[event_id_col]].append(row[object_id_col])

    object_types = objects_df[object_type_col].unique()
    discovered_models = {}

    for focus_type in object_types:
        print(f"\n=== [DISCOVERY] Object Type Perspective: {focus_type} ===")

        # Select relevant events containing the object type
        relevant_eids = [eid for eid, oids in event_to_objects.items()
                         if any(obj_id_to_type.get(oid) == focus_type for oid in oids)]

        # Filter the OCEL components
        sub_events = events_df[events_df[event_id_col].isin(relevant_eids)]
        sub_relations = objmap_df[objmap_df[event_id_col].isin(relevant_eids)]
        related_oids = set(sub_relations[object_id_col])
        related_objects = objects_df[objects_df[object_id_col].isin(related_oids)]

        # Create a sub-OCEL log
        sub_ocel = OCEL(events=sub_events, objects=related_objects, relations=sub_relations)

        # Discover and visualize the OCDFG
        ocdfg = pm4py.discover_ocdfg(sub_ocel)
        pm4py.view_ocdfg(ocdfg)

        # Store the discovered model
        discovered_models[focus_type] = ocdfg

    return discovered_models

# Example usage:
# ocel = pm4py.read_ocel("your_log.jsonocel")
# models = discover_object_perspective_models(ocel)

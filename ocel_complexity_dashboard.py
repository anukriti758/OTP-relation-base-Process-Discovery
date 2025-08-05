# # OCEL Complexity Dashboard

# This Python module visualizes and analyzes the structural complexity of Object-Centric Event Logs (OCEL) using both local and global complexity metrics.

# ## Features
# - Computes event diversity and co-object diversity per object type
# - Visualizes local components and co-occurrence graph
# - Supports sub-log creation by object types
# - Outputs interpretable dashboards and complexity scores

# ## Usage
# ```python
# from ocel_complexity_dashboard import ocel_complexity_dashboard
# final_score, sub_scores = ocel_complexity_dashboard(ocel_log)



import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from matplotlib.patches import FancyBboxPatch

def ocel_complexity_dashboard(log, alpha=0.5):
    def compute_metrics(ocel):
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
        event_diversity_dict = {}
        coobject_diversity_dict = {}

        for ot in OT:
            event_diversity = len(eventtypes_ot[ot]) / len(ET) if len(ET) > 0 else 0
            coobject_diversity = len(coobjecttypes_ot[ot]) / len(OT) if len(OT) > 0 else 0
            event_diversity_dict[ot] = event_diversity
            coobject_diversity_dict[ot] = coobject_diversity
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
        complexity = (1 - alpha) * local_score + alpha * rho
        final_score = 1.0 - complexity

        return E_ot, event_diversity_dict, coobject_diversity_dict, weights, local_components, G, final_score

    def generate_sub_ocel_relation_based(log):
        events_df = log.events.copy()
        objects_df = log.objects.copy()
        relations_df = log.relations.copy()

        obj_id_to_type = dict(zip(objects_df["ocel:oid"], objects_df["ocel:type"]))
        event_to_objects = defaultdict(list)
        for _, row in relations_df.iterrows():
            event_to_objects[row["ocel:eid"]].append(row["ocel:oid"])

        sub_logs = {}
        for ot in objects_df["ocel:type"].unique():
            relevant_eids = [eid for eid, oids in event_to_objects.items()
                             if any(obj_id_to_type.get(oid) == ot for oid in oids)]

            sub_events = events_df[events_df["ocel:eid"].isin(relevant_eids)]
            sub_relations = relations_df[relations_df["ocel:eid"].isin(relevant_eids)]
            sub_oids = sub_relations["ocel:oid"].unique()
            sub_objects = objects_df[objects_df["ocel:oid"].isin(sub_oids)]

            sub_logs[ot] = type(log)(events=sub_events, objects=sub_objects, relations=sub_relations)

        return sub_logs

    def plot_dashboard(E_ot, event_diversity_dict, coobject_diversity_dict, weights, local_components, G, title, score):
        fig = plt.figure(figsize=(22, 16))
        fig.suptitle(title, fontsize=20, fontweight='bold')

        def add_subplot_bar(position, data, title, ylabel, color):
            ax = fig.add_subplot(position)
            bars = ax.bar(data.keys(), data.values(), color=color, edgecolor='black', linewidth=0.8)
            ax.set_title(title, fontsize=14, fontweight='semibold')
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.keys(), rotation=45, fontsize=10)
            ax.set_facecolor('#f7f9fc')
            ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, color='black')

        add_subplot_bar(231, {ot: len(E_ot[ot]) for ot in E_ot}, "#Events per Object Type", "Count", "#2E86AB")
        add_subplot_bar(232, event_diversity_dict, "Event Diversity per Object Type", "Diversity", "#58B368")
        add_subplot_bar(233, coobject_diversity_dict, "Co-object Diversity per Object Type", "Diversity", "#F4A261")
        add_subplot_bar(234, weights, "Weight per Object Type", "Weight", "#9B5DE5")
        add_subplot_bar(235, local_components, "Local Component per Object Type", "Score", "#FF6B6B")

        ax = fig.add_subplot(236)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, with_labels=True, node_size=1200, node_color='#00B4D8',
                font_size=10, edge_color='#264653', font_weight='bold')
        ax.set_title("Object Type Co-occurrence Graph", fontsize=14, fontweight='semibold')
        ax.set_facecolor('#f0f0f0')

        summary_text = (
            f"\u2022 Object-Centric Complexity Score: {score:.4f}\n"
            f"\u2022 Number of Object Types: {len(weights)}\n"
            f"\u2022 Max Local Component: {max(local_components.values()):.2f}\n"
            f"\u2022 Max Weight: {max(weights.values()):.2f}\n"
            f"\u2022 Avg Event Diversity: {sum(event_diversity_dict.values())/len(event_diversity_dict):.2f}\n"
            f"\u2022 Avg Co-object Diversity: {sum(coobject_diversity_dict.values())/len(coobject_diversity_dict):.2f}"
        )

        fig.text(0.73, 0.05, summary_text, fontsize=12, va='bottom', ha='left',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#e0f7fa", edgecolor="#00acc1"))

        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
        plt.show()

    print("[INFO] Calculating metrics for original OCEL...")
    E_ot, event_diversity_dict, coobject_diversity_dict, weights, local_components, G, final_score = compute_metrics(log)
    plot_dashboard(E_ot, event_diversity_dict, coobject_diversity_dict, weights, local_components, G,
                   "OCEL Complexity Dashboard (Original)", final_score)

    print("[INFO] Generating relation-based sub-OCELs...")
    sub_logs = generate_sub_ocel_relation_based(log)

    sub_final_scores = {}
    sub_metrics = {}
    for ot, sublog in sub_logs.items():
        print(f"[INFO] Generating dashboard for Sub-OCEL: {ot}...")
        sub_metrics[ot] = compute_metrics(sublog)
        E_ot_s, event_div_s, coobj_div_s, weights_s, local_s, G_s, score_s = sub_metrics[ot]
        plot_dashboard(E_ot_s, event_div_s, coobj_div_s, weights_s, local_s, G_s,
                       f"OCEL Complexity Dashboard (Sub-OCEL: {ot})", score_s)
        sub_final_scores[ot] = score_s

    print("\n[INFO] Final Complexity Score (Original OCEL):", round(final_score, 4))
    for ot, score in sub_final_scores.items():
        print(f"[INFO] Final Complexity Score (Sub-OCEL: {ot}): {round(score, 4)}")

    return final_score, sub_final_scores

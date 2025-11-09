import numpy as np
import os
import time as t
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import re
from collections import deque

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

#-------------------------------------------------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPANSION_FILE = os.path.join(BASE_DIR, "expantion_list.txt")
MATRIX_FILE = os.path.join(BASE_DIR, "matrix_holder.txt")


def parse_expansion_list_with_children(path):
    if not os.path.exists(path):
        print("No expantion_list.txt found at", path)
        return []

    expansions = []
    current_parent = None
    current_depth = None
    current_path_cost = None
    current_parent_state = None

    state_pattern = re.compile(r"state\s*:\s*([A-Za-z0-9_]+)")
    children_pattern = re.compile(r"children:\s*\[(.*?)\]")
    depth_pattern = re.compile(r"depth\s*:\s*(\d+)")
    path_cost_pattern = re.compile(r"path_cost\s*:\s*([\d.]+)")
    parent_pattern = re.compile(r"parent\s*:\s*([A-Za-z0-9_]+)")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m_state = state_pattern.search(line)
            if m_state:
                current_parent = m_state.group(1).strip()

            m_depth = depth_pattern.search(line)
            if m_depth:
                current_depth = int(m_depth.group(1))

            m_path_cost = path_cost_pattern.search(line)
            if m_path_cost:
                current_path_cost = float(m_path_cost.group(1))

            m_parent = parent_pattern.search(line)
            if m_parent:
                current_parent_state = m_parent.group(1).strip()

            m_children = children_pattern.search(line)
            if m_children:
                children_str = m_children.group(1)
                if children_str.strip():
                    children = [c.strip().strip("'\"") for c in children_str.split(",")]
                    current_children = [c for c in children if c]
                else:
                    current_children = []

                if current_parent:
                    expansions.append({
                        'state': current_parent,
                        'children': current_children[:],
                        'depth': current_depth,
                        'path_cost': current_path_cost,
                        'parent_state': current_parent_state
                    })
                    current_parent = None
                    current_depth = None
                    current_path_cost = None
                    current_parent_state = None

    return expansions


def animate():
    expansions = parse_expansion_list_with_children(EXPANSION_FILE)
    if not expansions:
        print("No expansions parsed. Make sure expantion_list.txt exists and contains the expansion data.")
        return

    # --- Read goal and path correctly ---
    PATH_FILE = os.path.join(BASE_DIR, "path_log.txt")
    goal_node = None
    final_path_text = ""
    no_path_found = False
    
    if os.path.exists(PATH_FILE):
        with open(PATH_FILE, "r", encoding="utf-8") as f:
            final_path_text = f.read().strip()
            
            # Check if no path was found
            if final_path_text == "" or "No path found" in final_path_text:
                no_path_found = True
                final_path_text = "No path found!"
            else:
                parts = [x.strip() for x in re.split(r"<---", final_path_text)]
                goal_node = parts[0] if parts else None
    else:
        final_path_text = "(path_log.txt not found)"

    # Extract path nodes for highlighting
    path_nodes = []
    if not no_path_found and "<---" in final_path_text:
        path_nodes = [x.strip() for x in final_path_text.split("<---")]

    # --- build frames ---
    # fig = plt.figure(figsize=(16, 8))
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()  

    import tkinter as tk

    # گرفتن اندازه واقعی نمایشگر
    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    #  ساخت شکل دقیقاً به اندازه مانیتور
    fig = plt.figure(figsize=(screen_w / 100, screen_h / 100))
    mng = plt.get_current_fig_manager()

    try:
        # فول‌اسکرین تمیز در ویندوز
        mng.window.state('zoomed')
    except Exception:
        # برای سایر سیستم‌ها
        mng.full_screen_toggle()


    # fig = plt.figure(figsize=(10, 6))  # اندازه منطقی پایه
    # mng = plt.get_current_fig_manager()

    # try:
    #     # برای ویندوز
    #     mng.window.state('zoomed')
    # except Exception:
    #     # برای سیستم‌های دیگر
    #     mng.full_screen_toggle()
  
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
    ax_tree = fig.add_subplot(gs[0, 0])
    right_gs = gs[0, 1].subgridspec(2, 1)
    ax_current = fig.add_subplot(right_gs[0])
    ax_info = fig.add_subplot(right_gs[1])
    plt.tight_layout(pad=3)

    start_time = t.time()
    G = nx.DiGraph()
    node_id = 0
    fringe = deque()
    node_info = {}

    root_state = expansions[0]['state']
    root_id = f"n{node_id}"
    node_id += 1
    G.add_node(root_id, label=root_state, state=root_state)
    node_info[root_id] = (root_state, None)
    fringe.append(root_id)

    frame_data = []
    for exp_idx, expansion in enumerate(expansions):
        parent_state = expansion['state']
        children_states = expansion['children']
        depth = expansion.get('depth', 0)
        path_cost = expansion.get('path_cost', 0)
        parent_state_of_node = expansion.get('parent_state', 'None')

        parent_node_id = None
        for node in list(fringe):
            if node_info[node][0] == parent_state:
                parent_node_id = node
                fringe.remove(node)
                break

        if parent_node_id is None:
            continue

        child_ids = []
        for child_state in children_states:
            child_id = f"n{node_id}"
            node_id += 1
            G.add_node(child_id, label=child_state, state=child_state)
            G.add_edge(parent_node_id, child_id)
            node_info[child_id] = (child_state, parent_node_id)
            fringe.append(child_id)
            child_ids.append(child_id)

        frame_data.append({
            'parent_id': parent_node_id,
            'parent_state': parent_state,
            'child_ids': child_ids,
            'children_states': children_states,
            'depth': depth,
            'path_cost': path_cost,
            'parent_of_node': parent_state_of_node,
            'nodes_so_far': list(G.nodes()),
            'edges_so_far': list(G.edges())
        })

    # FIX: Use the actual length of frame_data, not expansions
    frames = len(frame_data)
    
    if frames == 0:
        print("No frame data generated. Check expansion parsing.")
        return

    def hierarchical_pos(G, root):
        depths = {}
        queue = [(root, 0)]
        visited = {root}
        while queue:
            node, depth = queue.pop(0)
            depths[node] = depth
            for child in G.successors(node):
                if child not in visited:
                    visited.add(child)
                    queue.append((child, depth + 1))
        positions = {}
        x_counter = [0]
        max_depth = max(depths.values()) if depths else 0

        def assign_x(node):
            children = sorted(G.successors(node))
            if not children:
                positions[node] = x_counter[0]
                x_counter[0] += 1
            else:
                for c in children[:-1]:
                    assign_x(c)
                assign_x(children[-1])
                child_positions = [positions[c] for c in children]
                positions[node] = (min(child_positions) + max(child_positions)) / 2.0

        assign_x(root)
        if positions:
            min_x, max_x = min(positions.values()), max(positions.values())
            x_range = max_x - min_x if max_x > min_x else 1
            pos = {}
            for n in positions:
                x = 0.1 + 0.8 * (positions[n] - min_x) / x_range
                y = 1.0 - (depths[n] / max(1, max_depth + 1))
                pos[n] = (x, y)
            return pos
        return {root: (0.5, 1.0)}

    def update(frame):
        ax_tree.clear()
        ax_current.clear()
        ax_info.clear()

        data = frame_data[frame]
        current_state = data['parent_state']
        goal_reached = (current_state == goal_node)

        G_sub = nx.DiGraph()
        for n in data['nodes_so_far']:
            G_sub.add_node(n, label=G.nodes[n]['label'], state=G.nodes[n]['state'])
        for e in data['edges_so_far']:
            if e[0] in G_sub and e[1] in G_sub:
                G_sub.add_edge(e[0], e[1])

        root = list(G_sub.nodes())[0]
        pos = hierarchical_pos(G_sub, root)

        # --- Color nodes with state tracking ---
        colors = []
        labels = {}
        already_red_states = set()  # Track which states have been colored red
        
        for n in G_sub.nodes():
            s = G_sub.nodes[n]['state']
            labels[n] = s

            if frame == frames - 1 and not no_path_found:  # Only in last frame and if path exists
                if s in path_nodes and s not in already_red_states:
                    colors.append("red")
                    already_red_states.add(s)  # Mark this state as already red
                elif s == goal_node:
                    colors.append("lightgreen")
                elif n == data['parent_id']:
                    colors.append("orange")
                else:
                    colors.append("skyblue")
            else:
                if s == goal_node and not no_path_found:
                    colors.append("lightgreen")
                elif n == data['parent_id']:
                    colors.append("orange")
                else:
                    colors.append("skyblue")

        # --- Color edges with state tracking ---
        already_red_edge_states = set()  # Track which state pairs have been colored red
        edge_colors = []
        
        for (u, v) in G_sub.edges():
            u_state = G_sub.nodes[u]['state']
            v_state = G_sub.nodes[v]['state']
            edge_state_pair = (u_state, v_state)
            
            if frame == frames - 1 and not no_path_found:
                # Check if both states are in path and this edge hasn't been red yet
                if (u_state in path_nodes and v_state in path_nodes and 
                    edge_state_pair not in already_red_edge_states):
                    # Check if they are consecutive in the path
                    try:
                        i1 = path_nodes.index(u_state)
                        i2 = path_nodes.index(v_state)
                        if abs(i1 - i2) == 1:
                            edge_colors.append("red")
                            already_red_edge_states.add(edge_state_pair)
                        else:
                            edge_colors.append("gray")
                    except ValueError:
                        edge_colors.append("gray")
                else:
                    edge_colors.append("gray")
            else:
                edge_colors.append("gray")

        nx.draw(G_sub, pos, labels=labels, with_labels=True, arrows=True,
                node_color=colors, ax=ax_tree, node_size=800, font_size=10,
                font_weight='bold', edge_color=edge_colors, arrowsize=15, width=2)
        ax_tree.axis('off')

        current_text = f"Currently Expanding Node:\n━━━━━━━━━━━━━━━━━━━━━\nState: {current_state}\nDepth: {data.get('depth','N/A')}\nPath Cost: {data.get('path_cost','N/A')}\nParent: {data.get('parent_of_node','None')}\n━━━━━━━━━━━━━━━━━━━━━\nChildren: {data['children_states']}\n"
        ax_current.text(0.5, 0.5, current_text, ha="center", va="center", fontsize=11,
                        fontweight="bold", color="darkorange", family='monospace')
        ax_current.axis("off")

        ax_info.axis("off")
        if goal_reached or frame == frames - 1:
            if no_path_found:
                info_text = f"NO PATH FOUND!\n\nThe goal state is not reachable\nfrom the initial state."
                ax_info.text(0.5, 0.5, info_text, ha="center", va="center",
                             fontsize=14, color="red", fontweight="bold",
                             family="monospace")
            else:
                info_text = f"GOAL REACHED!\n\n{final_path_text}"
                ax_info.text(0.5, 0.5, info_text, ha="center", va="center",
                             fontsize=12, color="green", fontweight="bold",
                             family="monospace")
            if goal_reached:
                ani.event_source.stop()
        else:
            elapsed = t.time() - start_time
            ax_info.text(0.5, 0.5,
                         f"Searching...\nTime: {elapsed:.2f}s\nExpansion: {frame+1}/{frames}",
                         ha="center", va="center", fontsize=12, color="blue")

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=800, repeat=False)
    plt.show()


animate()
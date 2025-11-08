
import numpy as np
import os
import time as t
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import re

#-------------------------------------------------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPANSION_FILE = os.path.join(BASE_DIR, "expantion_list.txt")
MATRIX_FILE = os.path.join(BASE_DIR, "matrix_holder.txt")


def parse_expansion_list(path):
    """Parse expantion_list.txt to extract the sequence of expanded node states."""
    if not os.path.exists(path):
        print("No expantion_list.txt found at", path)
        return []

    states = []
    state_pattern = re.compile(r"state\s*:\s*([A-Za-z0-9]+)")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = state_pattern.search(line)
            if m:
                states.append(m.group(1).strip())
    return states


def load_matrix_labels(matrix_path):
    """Loads the adjacency matrix and creates labels A, B, C, ..."""
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"matrix_holder.txt not found at {matrix_path}")

    mat = np.loadtxt(matrix_path, delimiter=",")
    n = mat.shape[0]
    labels = [chr(65 + i) for i in range(n)]
    return mat, labels


def adjacency_from_matrix(mat, labels):
    """Build adjacency list from numeric adjacency matrix."""
    edges = []
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            if mat[i, j] > 0:
                edges.append((labels[i], labels[j]))
    return edges


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.25, vert_loc=0, xcenter=0.5):
    """Create hierarchical positions for a tree."""
    if not nx.is_tree(G):
        raise TypeError("hierarchy_pos requires a tree graph")

    if root is None:
        root = next(iter(nx.topological_sort(G)))

    def _hierarchy_pos(G, root, left, right, vert_loc, vert_gap, pos=None):
        if pos is None:
            pos = {root: ((left + right) / 2.0, vert_loc)}
        else:
            pos[root] = ((left + right) / 2.0, vert_loc)
        children = list(G.successors(root))
        if children:
            dx = (right - left) / len(children)
            next_left = left
            for child in children:
                next_right = next_left + dx
                _hierarchy_pos(G, child, next_left, next_right, vert_loc - vert_gap, vert_gap, pos)
                next_left = next_right
        return pos

    return _hierarchy_pos(G, root, 0.0, width, vert_loc, vert_gap, None)


def build_tree_from_expansions(expansion_order, adjacency_edges):
    """Build a SEARCH TREE representation from expansion order."""
    G = nx.DiGraph()
    if not expansion_order:
        return G

    root = expansion_order[0]
    G.add_node(root)

    # adjacency map for quick lookup
    adj_map = {}
    for u, v in adjacency_edges:
        adj_map.setdefault(u, []).append(v)

    added = set([root])

    # Iterate through expansions and add newly discovered children
    for parent in expansion_order:
        children = adj_map.get(parent, [])
        for ch in children:
            if ch not in added:
                G.add_node(ch)
                G.add_edge(parent, ch)
                added.add(ch)

    return G


def animate():
    expansions = parse_expansion_list(EXPANSION_FILE)
    if not expansions:
        print("No expansions parsed. Make sure expantion_list.txt exists and contains the 'state :' lines.")
        return

    mat, labels = load_matrix_labels(MATRIX_FILE)
    adj_edges = adjacency_from_matrix(mat, labels)

    frames = len(expansions)

    # Build full tree once
    full_tree = build_tree_from_expansions(expansions, adj_edges)
    if full_tree.number_of_nodes() == 0:
        print("Built empty tree - nothing to show.")
        return

    root = expansions[0]
    # precompute hierarchical positions for the full tree
    try:
        pos = hierarchy_pos(full_tree, root=root, width=max(2.0, full_tree.number_of_nodes() * 0.5), vert_gap=0.35)
    except Exception:
        pos = nx.spring_layout(full_tree, seed=42)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
    ax_tree = fig.add_subplot(gs[0, 0])
    right_gs = gs[0, 1].subgridspec(2, 1)
    ax_current = fig.add_subplot(right_gs[0])
    ax_info = fig.add_subplot(right_gs[1])
    plt.tight_layout(pad=3)

    start_time = t.time()

    # For progressive drawing, compute which nodes to show per frame
    nodes_shown = []
    discovered = set()
    discovered.add(root)
    nodes_shown.append(set(discovered))

    adj_map = {}
    for u, v in adj_edges:
        adj_map.setdefault(u, []).append(v)

    for idx in range(1, frames):
        parent = expansions[idx - 1]
        current = set(nodes_shown[-1])
        for ch in adj_map.get(parent, []):
            if ch not in current:
                current.add(ch)
        nodes_shown.append(current)

    if len(nodes_shown) < frames:
        while len(nodes_shown) < frames:
            nodes_shown.append(set(nodes_shown[-1]))

    goal_node = expansions[-1]

    # Extract solution path from path_log.txt
    path_log_path = os.path.join(BASE_DIR, "path_log.txt")
    solution_path = []
    if os.path.exists(path_log_path):
        with open(path_log_path, "r", encoding="utf-8") as f:
            path_str = f.read().strip()
            solution_path = [x.strip() for x in path_str.split("<---") if x.strip()]

    def update(frame):
        ax_tree.clear()
        ax_current.clear()
        ax_info.clear()

        # nodes and edges visible this frame
        visible_nodes = nodes_shown[frame]
        visible_sub = full_tree.subgraph(visible_nodes).copy()

        # Current expanding node
        current_node = expansions[min(frame, frames - 1)]
        goal_reached = (current_node == goal_node)

        # node color mapping
        colors = []
        for n in visible_sub.nodes():
            # Goal node is green ONLY when it's being expanded (reached)
            if n == goal_node and goal_reached:
                colors.append("lightgreen")
            # Solution path nodes in gold (only after goal is reached)
            elif goal_reached and n in solution_path:
                colors.append("gold")
            # Currently expanding node in orange
            elif n == current_node:
                colors.append("orange")
            else:
                colors.append("skyblue")

        # draw tree portion
        nx.draw(visible_sub, pos=pos, with_labels=True, arrows=True, node_color=colors, ax=ax_tree, 
                node_size=800, font_size=12, font_weight='bold')
        ax_tree.set_title("Tree Search Visualization", fontsize=14, fontweight='bold')

        # Current expanding node display
        current_text = f"Currently Expanding:\n{current_node}"
        current_color = "darkgreen" if goal_reached else "darkorange"
        ax_current.text(0.5, 0.5, current_text, ha="center", va="center", fontsize=15,
                        fontweight="bold", color=current_color)
        ax_current.axis("off")

        elapsed = t.time() - start_time
        status = "✅ GOAL REACHED!" if goal_reached else "🔍 Searching..."
        status_color = "green" if goal_reached else "blue"
        ax_info.text(0.5, 0.5, f"Expanded Nodes: {frame + 1}\nTime: {elapsed:.2f}s\nStatus: {status}",
                     ha="center", va="center", fontsize=13, color=status_color,
                     fontweight="bold" if goal_reached else "normal")
        ax_info.axis("off")

        if goal_reached:
            ani.event_source.stop()

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=900, repeat=False)
    plt.show()
#-------------------------------------------------------------------------------------------------------------------------------
animate()
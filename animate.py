
# # # # # # # Ignore emoji/font warnings
# # # # # # warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# # # # # # base_dir = os.path.dirname(os.path.abspath(__file__))
# # # # # # steps_log_path = os.path.join(base_dir, "steps_log.txt")
# # # # # # path_log_path = os.path.join(base_dir, "path_log.txt")

# # # # # # # --- Helper: parse the logged steps ---
# # # # # # def parse_steps(file_path):
# # # # # #     steps = []
# # # # # #     with open(file_path, "r", encoding="utf-8") as f:
# # # # # #         for line in f:
# # # # # #             if "->" in line:
# # # # # #                 parent, children_str = line.strip().split("->")
# # # # # #                 parent = parent.strip()
# # # # # #                 try:
# # # # # #                     children = eval(children_str.strip())
# # # # # #                 except Exception:
# # # # # #                     children = []
# # # # # #                 steps.append((parent, children))
# # # # # #     return steps

# # # # # # # --- Hierarchical layout (tree-like) ---
# # # # # # def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.25, vert_loc=0, xcenter=0.5):
# # # # # #     """
# # # # # #     Recursively positions nodes in a hierarchy for NetworkX graphs.
# # # # # #     """
# # # # # #     if not nx.is_tree(G):
# # # # # #         raise TypeError("Cannot use hierarchy_pos on a graph that is not a tree")

# # # # # #     if root is None:
# # # # # #         if isinstance(G, nx.DiGraph):
# # # # # #             root = next(iter(nx.topological_sort(G)))
# # # # # #         else:
# # # # # #             root = list(G.nodes)[0]

# # # # # #     def _hierarchy_pos(G, root, left, right, vert_loc, vert_gap, pos=None, parent=None):
# # # # # #         if pos is None:
# # # # # #             pos = {root: (xcenter, vert_loc)}
# # # # # #         else:
# # # # # #             pos[root] = ((left + right) / 2, vert_loc)
# # # # # #         neighbors = list(G.successors(root))
# # # # # #         if neighbors:
# # # # # #             dx = (right - left) / len(neighbors)
# # # # # #             nextx = left
# # # # # #             for neighbor in neighbors:
# # # # # #                 nextx += dx
# # # # # #                 pos = _hierarchy_pos(G, neighbor, nextx - dx, nextx, vert_loc - vert_gap, vert_gap, pos, root)
# # # # # #         return pos

# # # # # #     return _hierarchy_pos(G, root, 0, width, vert_loc, vert_gap)

# # # # # # # --- Main animation ---
# # # # # # def animate_tree():
# # # # # #     steps = parse_steps(steps_log_path)
# # # # # #     if not steps:
# # # # # #         print("No steps found in log file.")
# # # # # #         return

# # # # # #     # Read the path (goal sequence)
# # # # # #     path = ""
# # # # # #     if os.path.exists(path_log_path):
# # # # # #         with open(path_log_path, "r", encoding="utf-8") as f:
# # # # # #             path = f.read().strip()

# # # # # #     goal_nodes = [x.strip() for x in path.split("<---") if x.strip()]

# # # # # #     G = nx.DiGraph()
# # # # # #     fig = plt.figure(figsize=(12, 6))
# # # # # #     gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])

# # # # # #     ax_tree = fig.add_subplot(gs[0, 0])
# # # # # #     right_gs = gs[0, 1].subgridspec(2, 1)
# # # # # #     ax_current = fig.add_subplot(right_gs[0])
# # # # # #     ax_info = fig.add_subplot(right_gs[1])

# # # # # #     plt.tight_layout(pad=3)

# # # # # #     start_time = t.time()
# # # # # #     expanded_count = 0

# # # # # #     def update(frame):
# # # # # #         nonlocal expanded_count
# # # # # #         ax_tree.clear()
# # # # # #         ax_current.clear()
# # # # # #         ax_info.clear()

# # # # # #         parent, children = steps[frame]
# # # # # #         expanded_count += 1
# # # # # #         for child in children:
# # # # # #             G.add_edge(parent, child)

# # # # # #         # Hierarchical layout
# # # # # #         try:
# # # # # #             pos = hierarchy_pos(G, list(G.nodes)[0])
# # # # # #         except Exception:
# # # # # #             pos = nx.spring_layout(G)

# # # # # #         # Color goal node differently if found
# # # # # #         node_colors = []
# # # # # #         for n in G.nodes():
# # # # # #             if n in goal_nodes:
# # # # # #                 node_colors.append("lightgreen")
# # # # # #             elif n == parent:
# # # # # #                 node_colors.append("orange")
# # # # # #             else:
# # # # # #                 node_colors.append("skyblue")

# # # # # #         nx.draw(G, pos, with_labels=True, arrows=True, node_color=node_colors, ax=ax_tree)
# # # # # #         ax_tree.set_title("Tree Search Visualization")

# # # # # #         ax_current.text(0.5, 0.5, f"Currently Expanding:\n{parent}",
# # # # # #                         ha="center", va="center", fontsize=15, fontweight="bold")
# # # # # #         ax_current.axis("off")

# # # # # #         elapsed = t.time() - start_time
# # # # # #         goal_reached = parent in goal_nodes
# # # # # #         status = "✅ Goal Found!" if goal_reached else "⏳ Searching..."
# # # # # #         ax_info.text(0.5, 0.5,
# # # # # #                      f"Expanded Nodes: {expanded_count}\nTime: {elapsed:.2f}s\nStatus: {status}",
# # # # # #                      ha="center", va="center", fontsize=13)
# # # # # #         ax_info.axis("off")

# # # # # #         # Stop animation once goal reached
# # # # # #         if goal_reached:
# # # # # #             ani.event_source.stop()

# # # # # #     ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=1200, repeat=False)
# # # # # #     plt.show()
# # # # # #-------------------------------------------------------------------------------------------------------------------------------


# # # # # import numpy as np
# # # # # import os
# # # # # import time as t
# # # # # import warnings
# # # # # import matplotlib.pyplot as plt
# # # # # import matplotlib.animation as animation
# # # # # import networkx as nx
# # # # # import re

# # # # # #-------------------------------------------------------------------------------------------------------------------------------
# # # # # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # # # # EXPANSION_FILE = os.path.join(BASE_DIR, "expantion_list.txt")
# # # # # MATRIX_FILE = os.path.join(BASE_DIR, "matrix_holder.txt")


# # # # # def parse_expansion_list(path):
# # # # #     """Parse expantion_list.txt to extract the sequence of expanded node states."""
# # # # #     if not os.path.exists(path):
# # # # #         print("No expantion_list.txt found at", path)
# # # # #         return []

# # # # #     states = []
# # # # #     state_pattern = re.compile(r"state\s*:\s*([A-Za-z0-9]+)")
# # # # #     with open(path, "r", encoding="utf-8") as f:
# # # # #         for line in f:
# # # # #             m = state_pattern.search(line)
# # # # #             if m:
# # # # #                 states.append(m.group(1).strip())
# # # # #     return states


# # # # # def load_matrix_labels(matrix_path):
# # # # #     """Loads the adjacency matrix and creates labels A, B, C, ..."""
# # # # #     if not os.path.exists(matrix_path):
# # # # #         raise FileNotFoundError(f"matrix_holder.txt not found at {matrix_path}")

# # # # #     mat = np.loadtxt(matrix_path, delimiter=",")
# # # # #     n = mat.shape[0]
# # # # #     labels = [chr(65 + i) for i in range(n)]
# # # # #     return mat, labels


# # # # # def adjacency_from_matrix(mat, labels):
# # # # #     """Build adjacency list from numeric adjacency matrix."""
# # # # #     edges = []
# # # # #     n = mat.shape[0]
# # # # #     for i in range(n):
# # # # #         for j in range(n):
# # # # #             if mat[i, j] > 0:
# # # # #                 edges.append((labels[i], labels[j]))
# # # # #     return edges


# # # # # def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.25, vert_loc=0, xcenter=0.5):
# # # # #     """Create hierarchical positions for a tree."""
# # # # #     if not nx.is_tree(G):
# # # # #         raise TypeError("hierarchy_pos requires a tree graph")

# # # # #     if root is None:
# # # # #         root = next(iter(nx.topological_sort(G)))

# # # # #     def _hierarchy_pos(G, root, left, right, vert_loc, vert_gap, pos=None):
# # # # #         if pos is None:
# # # # #             pos = {root: ((left + right) / 2.0, vert_loc)}
# # # # #         else:
# # # # #             pos[root] = ((left + right) / 2.0, vert_loc)
# # # # #         children = list(G.successors(root))
# # # # #         if children:
# # # # #             dx = (right - left) / len(children)
# # # # #             next_left = left
# # # # #             for child in children:
# # # # #                 next_right = next_left + dx
# # # # #                 _hierarchy_pos(G, child, next_left, next_right, vert_loc - vert_gap, vert_gap, pos)
# # # # #                 next_left = next_right
# # # # #         return pos

# # # # #     return _hierarchy_pos(G, root, 0.0, width, vert_loc, vert_gap, None)


# # # # # def build_tree_from_expansions(expansion_order, adjacency_edges):
# # # # #     """Build a SEARCH TREE representation from expansion order."""
# # # # #     G = nx.DiGraph()
# # # # #     if not expansion_order:
# # # # #         return G

# # # # #     root = expansion_order[0]
# # # # #     G.add_node(root)

# # # # #     # adjacency map for quick lookup
# # # # #     adj_map = {}
# # # # #     for u, v in adjacency_edges:
# # # # #         adj_map.setdefault(u, []).append(v)

# # # # #     added = set([root])

# # # # #     # Iterate through expansions and add newly discovered children
# # # # #     for parent in expansion_order:
# # # # #         children = adj_map.get(parent, [])
# # # # #         for ch in children:
# # # # #             if ch not in added:
# # # # #                 G.add_node(ch)
# # # # #                 G.add_edge(parent, ch)
# # # # #                 added.add(ch)

# # # # #     return G


# # # # # def animate():
# # # # #     expansions = parse_expansion_list(EXPANSION_FILE)
# # # # #     if not expansions:
# # # # #         print("No expansions parsed. Make sure expantion_list.txt exists and contains the 'state :' lines.")
# # # # #         return

# # # # #     mat, labels = load_matrix_labels(MATRIX_FILE)
# # # # #     adj_edges = adjacency_from_matrix(mat, labels)

# # # # #     frames = len(expansions)

# # # # #     # Build full tree once
# # # # #     full_tree = build_tree_from_expansions(expansions, adj_edges)
# # # # #     if full_tree.number_of_nodes() == 0:
# # # # #         print("Built empty tree - nothing to show.")
# # # # #         return

# # # # #     root = expansions[0]
# # # # #     # precompute hierarchical positions for the full tree
# # # # #     try:
# # # # #         pos = hierarchy_pos(full_tree, root=root, width=max(2.0, full_tree.number_of_nodes() * 0.5), vert_gap=0.35)
# # # # #     except Exception:
# # # # #         pos = nx.spring_layout(full_tree, seed=42)

# # # # #     fig = plt.figure(figsize=(12, 6))
# # # # #     gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
# # # # #     ax_tree = fig.add_subplot(gs[0, 0])
# # # # #     right_gs = gs[0, 1].subgridspec(2, 1)
# # # # #     ax_current = fig.add_subplot(right_gs[0])
# # # # #     ax_info = fig.add_subplot(right_gs[1])
# # # # #     plt.tight_layout(pad=3)

# # # # #     start_time = t.time()

# # # # #     # For progressive drawing, compute which nodes to show per frame
# # # # #     nodes_shown = []
# # # # #     discovered = set()
# # # # #     discovered.add(root)
# # # # #     nodes_shown.append(set(discovered))

# # # # #     adj_map = {}
# # # # #     for u, v in adj_edges:
# # # # #         adj_map.setdefault(u, []).append(v)

# # # # #     for idx in range(1, frames):
# # # # #         parent = expansions[idx - 1]
# # # # #         current = set(nodes_shown[-1])
# # # # #         for ch in adj_map.get(parent, []):
# # # # #             if ch not in current:
# # # # #                 current.add(ch)
# # # # #         nodes_shown.append(current)

# # # # #     if len(nodes_shown) < frames:
# # # # #         while len(nodes_shown) < frames:
# # # # #             nodes_shown.append(set(nodes_shown[-1]))

# # # # #     goal_node = expansions[-1]

# # # # #     # Extract solution path from path_log.txt
# # # # #     path_log_path = os.path.join(BASE_DIR, "path_log.txt")
# # # # #     solution_path = []
# # # # #     if os.path.exists(path_log_path):
# # # # #         with open(path_log_path, "r", encoding="utf-8") as f:
# # # # #             path_str = f.read().strip()
# # # # #             solution_path = [x.strip() for x in path_str.split("<---") if x.strip()]

# # # # #     def update(frame):
# # # # #         ax_tree.clear()
# # # # #         ax_current.clear()
# # # # #         ax_info.clear()

# # # # #         # nodes and edges visible this frame
# # # # #         visible_nodes = nodes_shown[frame]
# # # # #         visible_sub = full_tree.subgraph(visible_nodes).copy()

# # # # #         # Current expanding node
# # # # #         current_node = expansions[min(frame, frames - 1)]
# # # # #         goal_reached = (current_node == goal_node)

# # # # #         # node color mapping
# # # # #         colors = []
# # # # #         for n in visible_sub.nodes():
# # # # #             # Goal node is green ONLY when it's being expanded (reached)
# # # # #             if n == goal_node and goal_reached:
# # # # #                 colors.append("lightgreen")
# # # # #             # Solution path nodes in gold (only after goal is reached)
# # # # #             elif goal_reached and n in solution_path:
# # # # #                 colors.append("gold")
# # # # #             # Currently expanding node in orange
# # # # #             elif n == current_node:
# # # # #                 colors.append("orange")
# # # # #             else:
# # # # #                 colors.append("skyblue")

# # # # #         # draw tree portion
# # # # #         nx.draw(visible_sub, pos=pos, with_labels=True, arrows=True, node_color=colors, ax=ax_tree, 
# # # # #                 node_size=800, font_size=12, font_weight='bold')
# # # # #         ax_tree.set_title("Tree Search Visualization", fontsize=14, fontweight='bold')

# # # # #         # Current expanding node display
# # # # #         current_text = f"Currently Expanding:\n{current_node}"
# # # # #         current_color = "darkgreen" if goal_reached else "darkorange"
# # # # #         ax_current.text(0.5, 0.5, current_text, ha="center", va="center", fontsize=15,
# # # # #                         fontweight="bold", color=current_color)
# # # # #         ax_current.axis("off")

# # # # #         elapsed = t.time() - start_time
# # # # #         status = "✅ GOAL REACHED!" if goal_reached else "🔍 Searching..."
# # # # #         status_color = "green" if goal_reached else "blue"
# # # # #         ax_info.text(0.5, 0.5, f"Expanded Nodes: {frame + 1}\nTime: {elapsed:.2f}s\nStatus: {status}",
# # # # #                      ha="center", va="center", fontsize=13, color=status_color,
# # # # #                      fontweight="bold" if goal_reached else "normal")
# # # # #         ax_info.axis("off")

# # # # #         if goal_reached:
# # # # #             ani.event_source.stop()

# # # # #     ani = animation.FuncAnimation(fig, update, frames=frames, interval=900, repeat=False)
# # # # #     plt.show()
# # # # # #-------------------------------------------------------------------------------------------------------------------------------
# # # # # animate()

# # # # import numpy as np
# # # # import os
# # # # import time as t
# # # # import warnings
# # # # import matplotlib.pyplot as plt
# # # # import matplotlib.animation as animation
# # # # import networkx as nx
# # # # import re

# # # # # Silence matplotlib font emoji warnings
# # # # warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# # # # #-------------------------------------------------------------------------------------------------------------------------------
# # # # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # # # EXPANSION_FILE = os.path.join(BASE_DIR, "expantion_list.txt")
# # # # MATRIX_FILE = os.path.join(BASE_DIR, "matrix_holder.txt")


# # # # def parse_expansion_list_with_children(path):
# # # #     """
# # # #     Parse expantion_list.txt to extract ALL expansions with their children.
# # # #     Returns list of tuples: [(parent_state, [children_states]), ...]
# # # #     This includes repeated expansions!
# # # #     """
# # # #     if not os.path.exists(path):
# # # #         print("No expantion_list.txt found at", path)
# # # #         return []

# # # #     expansions = []
# # # #     current_parent = None
# # # #     current_children = []
    
# # # #     state_pattern = re.compile(r"state\s*:\s*([A-Za-z0-9]+)")
# # # #     children_pattern = re.compile(r"children:\s*\[(.*?)\]")
    
# # # #     with open(path, "r", encoding="utf-8") as f:
# # # #         for line in f:
# # # #             # Check for parent state
# # # #             m_state = state_pattern.search(line)
# # # #             if m_state:
# # # #                 current_parent = m_state.group(1).strip()
            
# # # #             # Check for children
# # # #             m_children = children_pattern.search(line)
# # # #             if m_children:
# # # #                 children_str = m_children.group(1)
# # # #                 if children_str.strip():
# # # #                     # Parse children list: ['A', 'B', 'C'] -> [A, B, C]
# # # #                     children = [c.strip().strip("'\"") for c in children_str.split(",")]
# # # #                     current_children = [c for c in children if c]
# # # #                 else:
# # # #                     current_children = []
                
# # # #                 # Save this expansion
# # # #                 if current_parent:
# # # #                     expansions.append((current_parent, current_children[:]))
# # # #                     current_parent = None
# # # #                     current_children = []
    
# # # #     return expansions


# # # # def load_matrix_labels(matrix_path):
# # # #     """Loads the adjacency matrix and creates labels A, B, C, ..."""
# # # #     if not os.path.exists(matrix_path):
# # # #         raise FileNotFoundError(f"matrix_holder.txt not found at {matrix_path}")

# # # #     mat = np.loadtxt(matrix_path, delimiter=",")
# # # #     n = mat.shape[0]
# # # #     labels = [chr(65 + i) for i in range(n)]
# # # #     return mat, labels


# # # # def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.25, vert_loc=0, xcenter=0.5):
# # # #     """Create hierarchical positions for a tree."""
# # # #     if not nx.is_tree(G):
# # # #         # For graphs with cycles, use spring layout
# # # #         return None

# # # #     if root is None:
# # # #         root = next(iter(nx.topological_sort(G)))

# # # #     def _hierarchy_pos(G, root, left, right, vert_loc, vert_gap, pos=None):
# # # #         if pos is None:
# # # #             pos = {root: ((left + right) / 2.0, vert_loc)}
# # # #         else:
# # # #             pos[root] = ((left + right) / 2.0, vert_loc)
# # # #         children = list(G.successors(root))
# # # #         if children:
# # # #             dx = (right - left) / len(children)
# # # #             next_left = left
# # # #             for child in children:
# # # #                 next_right = next_left + dx
# # # #                 _hierarchy_pos(G, child, next_left, next_right, vert_loc - vert_gap, vert_gap, pos)
# # # #                 next_left = next_right
# # # #         return pos

# # # #     return _hierarchy_pos(G, root, 0.0, width, vert_loc, vert_gap, None)


# # # # def animate():
# # # #     expansions = parse_expansion_list_with_children(EXPANSION_FILE)
# # # #     if not expansions:
# # # #         print("No expansions parsed. Make sure expantion_list.txt exists and contains the expansion data.")
# # # #         return

# # # #     frames = len(expansions)
    
# # # #     # Get the goal node (last expanded node)
# # # #     goal_node = expansions[-1][0]
    
# # # #     # Extract solution path from path_log.txt
# # # #     path_log_path = os.path.join(BASE_DIR, "path_log.txt")
# # # #     solution_path = []
# # # #     if os.path.exists(path_log_path):
# # # #         with open(path_log_path, "r", encoding="utf-8") as f:
# # # #             path_str = f.read().strip()
# # # #             solution_path = [x.strip() for x in path_str.split("<---") if x.strip()]

# # # #     fig = plt.figure(figsize=(14, 7))
# # # #     gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
# # # #     ax_tree = fig.add_subplot(gs[0, 0])
# # # #     right_gs = gs[0, 1].subgridspec(2, 1)
# # # #     ax_current = fig.add_subplot(right_gs[0])
# # # #     ax_info = fig.add_subplot(right_gs[1])
# # # #     plt.tight_layout(pad=3)

# # # #     start_time = t.time()

# # # #     # Build the tree progressively
# # # #     G = nx.DiGraph()
    
# # # #     # For handling multiple instances of same node, we'll use unique IDs
# # # #     node_counter = {}
    
# # # #     def get_node_id(state):
# # # #         """Create unique node ID for visualization (handles repeated nodes)"""
# # # #         if state not in node_counter:
# # # #             node_counter[state] = 0
# # # #             return state
# # # #         else:
# # # #             node_counter[state] += 1
# # # #             return f"{state}_{node_counter[state]}"

# # # #     def update(frame):
# # # #         ax_tree.clear()
# # # #         ax_current.clear()
# # # #         ax_info.clear()

# # # #         # Add edges for current frame
# # # #         parent, children = expansions[frame]
        
# # # #         # Add parent if not exists
# # # #         if parent not in G.nodes():
# # # #             G.add_node(parent)
        
# # # #         # Add children and edges
# # # #         for child in children:
# # # #             # For repeated nodes in tree, create unique ID but display original label
# # # #             child_id = get_node_id(child) if child in G.nodes() else child
# # # #             G.add_node(child_id, label=child)  # Store original label
# # # #             G.add_edge(parent, child_id)

# # # #         # Current expanding node
# # # #         current_node = parent
# # # #         goal_reached = (current_node == goal_node)

# # # #         # Try hierarchical layout, fall back to spring layout
# # # #         try:
# # # #             root = list(G.nodes())[0]
# # # #             pos = hierarchy_pos(G, root=root, width=max(3.0, G.number_of_nodes() * 0.4), vert_gap=0.3)
# # # #         except:
# # # #             pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        
# # # #         if pos is None:
# # # #             pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

# # # #         # Node color mapping
# # # #         colors = []
# # # #         labels_dict = {}
# # # #         for n in G.nodes():
# # # #             # Get original label
# # # #             original_label = G.nodes[n].get('label', n)
# # # #             labels_dict[n] = original_label
            
# # # #             # Goal node is green ONLY when it's being expanded (reached)
# # # #             if original_label == goal_node and goal_reached:
# # # #                 colors.append("lightgreen")
# # # #             # Solution path nodes in gold (only after goal is reached)
# # # #             elif goal_reached and original_label in solution_path:
# # # #                 colors.append("gold")
# # # #             # Currently expanding node in orange
# # # #             elif n == current_node or original_label == current_node:
# # # #                 colors.append("orange")
# # # #             else:
# # # #                 colors.append("skyblue")

# # # #         # Draw tree
# # # #         nx.draw(G, pos, labels=labels_dict, with_labels=True, arrows=True, 
# # # #                 node_color=colors, ax=ax_tree, node_size=700, font_size=11, 
# # # #                 font_weight='bold', edge_color='gray', arrowsize=15)
# # # #         ax_tree.set_title("Complete Tree Search (All Expansions)", fontsize=14, fontweight='bold')

# # # #         # Current expanding node display
# # # #         current_text = f"Currently Expanding:\n{current_node}\n\nChildren: {children}"
# # # #         current_color = "darkgreen" if goal_reached else "darkorange"
# # # #         ax_current.text(0.5, 0.5, current_text, ha="center", va="center", fontsize=13,
# # # #                         fontweight="bold", color=current_color)
# # # #         ax_current.axis("off")

# # # #         elapsed = t.time() - start_time
# # # #         status = "✅ GOAL REACHED!" if goal_reached else "🔍 Searching..."
# # # #         status_color = "green" if goal_reached else "blue"
# # # #         ax_info.text(0.5, 0.5, 
# # # #                      f"Expansion #{frame + 1} / {frames}\nTotal Nodes: {G.number_of_nodes()}\nTime: {elapsed:.2f}s\n\nStatus: {status}",
# # # #                      ha="center", va="center", fontsize=13, color=status_color,
# # # #                      fontweight="bold" if goal_reached else "normal")
# # # #         ax_info.axis("off")

# # # #         if goal_reached:
# # # #             ani.event_source.stop()

# # # #     ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000, repeat=False)
# # # #     plt.show()

# # # # animate()

# # # import numpy as np
# # # import os
# # # import time as t
# # # import warnings
# # # import matplotlib.pyplot as plt
# # # import matplotlib.animation as animation
# # # import networkx as nx
# # # import re

# # # # Silence matplotlib font emoji warnings
# # # warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# # # #-------------------------------------------------------------------------------------------------------------------------------
# # # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # # EXPANSION_FILE = os.path.join(BASE_DIR, "expantion_list.txt")
# # # MATRIX_FILE = os.path.join(BASE_DIR, "matrix_holder.txt")


# # # def parse_expansion_list_with_children(path):
# # #     """Parse expantion_list.txt to extract ALL expansions with their children."""
# # #     if not os.path.exists(path):
# # #         print("No expantion_list.txt found at", path)
# # #         return []

# # #     expansions = []
# # #     current_parent = None
    
# # #     state_pattern = re.compile(r"state\s*:\s*([A-Za-z0-9]+)")
# # #     children_pattern = re.compile(r"children:\s*\[(.*?)\]")
    
# # #     with open(path, "r", encoding="utf-8") as f:
# # #         for line in f:
# # #             m_state = state_pattern.search(line)
# # #             if m_state:
# # #                 current_parent = m_state.group(1).strip()
            
# # #             m_children = children_pattern.search(line)
# # #             if m_children:
# # #                 children_str = m_children.group(1)
# # #                 if children_str.strip():
# # #                     children = [c.strip().strip("'\"") for c in children_str.split(",")]
# # #                     current_children = [c for c in children if c]
# # #                 else:
# # #                     current_children = []
                
# # #                 if current_parent:
# # #                     expansions.append((current_parent, current_children[:]))
# # #                     current_parent = None
    
# # #     return expansions


# # # def draw_tree_hierarchical(G, root, ax, colors, labels_dict):
# # #     """Draw tree in perfect hierarchical layout using manual positioning."""
    
# # #     # Calculate levels (depth) for each node
# # #     levels = {}
    
# # #     def assign_levels(node, level=0):
# # #         levels[node] = level
# # #         for child in G.successors(node):
# # #             assign_levels(child, level + 1)
    
# # #     assign_levels(root)
    
# # #     # Group nodes by level
# # #     nodes_by_level = {}
# # #     for node, level in levels.items():
# # #         nodes_by_level.setdefault(level, []).append(node)
    
# # #     # Calculate positions
# # #     pos = {}
# # #     max_level = max(levels.values()) if levels else 0
    
# # #     for level, nodes in nodes_by_level.items():
# # #         y = 1.0 - (level / (max_level + 1))  # Top to bottom
# # #         num_nodes = len(nodes)
        
# # #         if num_nodes == 1:
# # #             pos[nodes[0]] = (0.5, y)
# # #         else:
# # #             # Spread nodes evenly across x-axis
# # #             spacing = 1.0 / (num_nodes + 1)
# # #             for i, node in enumerate(nodes):
# # #                 x = spacing * (i + 1)
# # #                 pos[node] = (x, y)
    
# # #     # Draw the tree
# # #     nx.draw(G, pos, labels=labels_dict, with_labels=True, arrows=True,
# # #             node_color=colors, ax=ax, node_size=800, font_size=12,
# # #             font_weight='bold', edge_color='gray', arrowsize=20,
# # #             arrowstyle='->', connectionstyle='arc3,rad=0.1')
    
# # #     ax.set_xlim(-0.1, 1.1)
# # #     ax.set_ylim(-0.1, 1.1)


# # # def animate():
# # #     expansions = parse_expansion_list_with_children(EXPANSION_FILE)
# # #     if not expansions:
# # #         print("No expansions parsed. Make sure expantion_list.txt exists and contains the expansion data.")
# # #         return

# # #     frames = len(expansions)
# # #     goal_node = expansions[-1][0]
    
# # #     # Extract solution path
# # #     path_log_path = os.path.join(BASE_DIR, "path_log.txt")
# # #     solution_path = []
# # #     if os.path.exists(path_log_path):
# # #         with open(path_log_path, "r", encoding="utf-8") as f:
# # #             path_str = f.read().strip()
# # #             solution_path = [x.strip() for x in path_str.split("<---") if x.strip()]

# # #     fig = plt.figure(figsize=(16, 8))
# # #     gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
# # #     ax_tree = fig.add_subplot(gs[0, 0])
# # #     right_gs = gs[0, 1].subgridspec(2, 1)
# # #     ax_current = fig.add_subplot(right_gs[0])
# # #     ax_info = fig.add_subplot(right_gs[1])
# # #     plt.tight_layout(pad=3)

# # #     start_time = t.time()

# # #     # Build complete tree structure first - each expansion creates NEW nodes
# # #     all_frames_data = []
# # #     G = nx.DiGraph()
# # #     node_counter = 0
# # #     pending_parents = {}  # Maps state -> list of node_ids that can be expanded
    
# # #     for frame_idx, (parent_state, children_states) in enumerate(expansions):
# # #         # Find which node ID to expand
# # #         if frame_idx == 0:
# # #             # Root node
# # #             parent_id = f"n{node_counter}"
# # #             node_counter += 1
# # #             G.add_node(parent_id, label=parent_state, state=parent_state)
# # #             pending_parents[parent_state] = [parent_id]
# # #         else:
# # #             # Get the first available parent node with this state
# # #             if parent_state in pending_parents and pending_parents[parent_state]:
# # #                 parent_id = pending_parents[parent_state].pop(0)
# # #             else:
# # #                 print(f"Warning: No pending parent for {parent_state}")
# # #                 continue
        
# # #         # Create children
# # #         child_ids = []
# # #         for child_state in children_states:
# # #             child_id = f"n{node_counter}"
# # #             node_counter += 1
# # #             G.add_node(child_id, label=child_state, state=child_state)
# # #             G.add_edge(parent_id, child_id)
# # #             child_ids.append(child_id)
            
# # #             # Add to pending parents
# # #             pending_parents.setdefault(child_state, []).append(child_id)
        
# # #         all_frames_data.append({
# # #             'parent_id': parent_id,
# # #             'parent_state': parent_state,
# # #             'child_ids': child_ids,
# # #             'children_states': children_states
# # #         })

# # #     # Now animate by showing progressive subgraphs
# # #     def update(frame):
# # #         ax_tree.clear()
# # #         ax_current.clear()
# # #         ax_info.clear()

# # #         # Build subgraph up to this frame
# # #         nodes_to_show = set()
# # #         edges_to_show = []
        
# # #         for i in range(frame + 1):
# # #             frame_data = all_frames_data[i]
# # #             nodes_to_show.add(frame_data['parent_id'])
# # #             for child_id in frame_data['child_ids']:
# # #                 nodes_to_show.add(child_id)
# # #                 edges_to_show.append((frame_data['parent_id'], child_id))
        
# # #         G_sub = G.subgraph(nodes_to_show).copy()
        
# # #         # Current frame info
# # #         current_data = all_frames_data[frame]
# # #         current_node = current_data['parent_state']
# # #         goal_reached = (current_node == goal_node)

# # #         # Node colors
# # #         colors = []
# # #         labels_dict = {}
# # #         for n in G_sub.nodes():
# # #             node_label = G.nodes[n]['label']
# # #             node_state = G.nodes[n]['state']
# # #             labels_dict[n] = node_label
            
# # #             # Color based on state
# # #             if node_state == goal_node and goal_reached:
# # #                 colors.append("lightgreen")
# # #             elif goal_reached and node_state in solution_path:
# # #                 colors.append("gold")
# # #             elif n == current_data['parent_id']:
# # #                 colors.append("orange")
# # #             else:
# # #                 colors.append("skyblue")

# # #         # Draw with hierarchical layout
# # #         root = list(G_sub.nodes())[0]
# # #         draw_tree_hierarchical(G_sub, root, ax_tree, colors, labels_dict)
# # #         ax_tree.set_title("Tree Search - Hierarchical View", fontsize=16, fontweight='bold')
# # #         ax_tree.axis('off')

# # #         # Current expanding node display
# # #         current_text = f"Expanding: {current_node}\n\nChildren: {current_data['children_states']}"
# # #         current_color = "darkgreen" if goal_reached else "darkorange"
# # #         ax_current.text(0.5, 0.5, current_text, ha="center", va="center", fontsize=14,
# # #                         fontweight="bold", color=current_color, wrap=True)
# # #         ax_current.axis("off")

# # #         # Info panel
# # #         elapsed = t.time() - start_time
# # #         status = "✅ GOAL REACHED!" if goal_reached else "🔍 Searching..."
# # #         status_color = "green" if goal_reached else "blue"
# # #         info_text = f"Expansion: {frame + 1} / {frames}\n"
# # #         info_text += f"Total Nodes: {len(nodes_to_show)}\n"
# # #         info_text += f"Time: {elapsed:.2f}s\n\n"
# # #         info_text += f"{status}"
        
# # #         ax_info.text(0.5, 0.5, info_text,
# # #                      ha="center", va="center", fontsize=14, color=status_color,
# # #                      fontweight="bold" if goal_reached else "normal")
# # #         ax_info.axis("off")

# # #         if goal_reached:
# # #             ani.event_source.stop()

# # #     ani = animation.FuncAnimation(fig, update, frames=frames, interval=1200, repeat=False)
# # #     plt.show()


# # # animate()

# # import numpy as np
# # import os
# # import time as t
# # import warnings
# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation
# # import networkx as nx
# # import re

# # # Silence matplotlib font emoji warnings
# # warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# # #-------------------------------------------------------------------------------------------------------------------------------
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # EXPANSION_FILE = os.path.join(BASE_DIR, "expantion_list.txt")
# # MATRIX_FILE = os.path.join(BASE_DIR, "matrix_holder.txt")


# # def parse_expansion_list_with_children(path):
# #     """Parse expantion_list.txt to extract ALL expansions with their children."""
# #     if not os.path.exists(path):
# #         print("No expantion_list.txt found at", path)
# #         return []

# #     expansions = []
# #     current_parent = None
    
# #     state_pattern = re.compile(r"state\s*:\s*([A-Za-z0-9]+)")
# #     children_pattern = re.compile(r"children:\s*\[(.*?)\]")
    
# #     with open(path, "r", encoding="utf-8") as f:
# #         for line in f:
# #             m_state = state_pattern.search(line)
# #             if m_state:
# #                 current_parent = m_state.group(1).strip()
            
# #             m_children = children_pattern.search(line)
# #             if m_children:
# #                 children_str = m_children.group(1)
# #                 if children_str.strip():
# #                     children = [c.strip().strip("'\"") for c in children_str.split(",")]
# #                     current_children = [c for c in children if c]
# #                 else:
# #                     current_children = []
                
# #                 if current_parent:
# #                     expansions.append((current_parent, current_children[:]))
# #                     current_parent = None
    
# #     return expansions


# # def draw_tree_hierarchical(G_sub, root, ax, colors, labels_dict):
# #     """Draw tree in perfect hierarchical layout using manual positioning."""
    
# #     # Calculate levels (depth) for each node using BFS
# #     levels = {root: 0}
# #     queue = [root]
    
# #     while queue:
# #         node = queue.pop(0)
# #         current_level = levels[node]
# #         for child in G_sub.successors(node):
# #             if child not in levels:  # Avoid cycles
# #                 levels[child] = current_level + 1
# #                 queue.append(child)
    
# #     # Group nodes by level
# #     nodes_by_level = {}
# #     for node in G_sub.nodes():
# #         level = levels.get(node, 0)
# #         nodes_by_level.setdefault(level, []).append(node)
    
# #     # Calculate positions for ALL nodes in G_sub
# #     pos = {}
# #     max_level = max(levels.values()) if levels else 0
    
# #     for level, nodes in nodes_by_level.items():
# #         y = 1.0 - (level / (max_level + 1))  # Top to bottom
# #         num_nodes = len(nodes)
        
# #         if num_nodes == 1:
# #             pos[nodes[0]] = (0.5, y)
# #         else:
# #             # Spread nodes evenly across x-axis
# #             spacing = 1.0 / (num_nodes + 1)
# #             for i, node in enumerate(nodes):
# #                 x = spacing * (i + 1)
# #                 pos[node] = (x, y)
    
# #     # Ensure ALL nodes in G_sub have positions
# #     for node in G_sub.nodes():
# #         if node not in pos:
# #             pos[node] = (0.5, 0.5)  # Fallback position
    
# #     # Draw the tree - use G_sub for both graph and positions
# #     nx.draw(G_sub, pos, labels=labels_dict, with_labels=True, arrows=True,
# #             node_color=colors, ax=ax, node_size=800, font_size=12,
# #             font_weight='bold', edge_color='gray', arrowsize=20,
# #             arrowstyle='->', connectionstyle='arc3,rad=0.1')
    
# #     ax.set_xlim(-0.1, 1.1)
# #     ax.set_ylim(-0.1, 1.1)


# # def animate():
# #     expansions = parse_expansion_list_with_children(EXPANSION_FILE)
# #     if not expansions:
# #         print("No expansions parsed. Make sure expantion_list.txt exists and contains the expansion data.")
# #         return

# #     frames = len(expansions)
# #     goal_node = expansions[-1][0]
    
# #     # Extract solution path
# #     path_log_path = os.path.join(BASE_DIR, "path_log.txt")
# #     solution_path = []
# #     if os.path.exists(path_log_path):
# #         with open(path_log_path, "r", encoding="utf-8") as f:
# #             path_str = f.read().strip()
# #             solution_path = [x.strip() for x in path_str.split("<---") if x.strip()]

# #     fig = plt.figure(figsize=(16, 8))
# #     gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
# #     ax_tree = fig.add_subplot(gs[0, 0])
# #     right_gs = gs[0, 1].subgridspec(2, 1)
# #     ax_current = fig.add_subplot(right_gs[0])
# #     ax_info = fig.add_subplot(right_gs[1])
# #     plt.tight_layout(pad=3)

# #     start_time = t.time()

# #     # Build complete tree structure first - each expansion creates NEW nodes
# #     all_frames_data = []
# #     G = nx.DiGraph()
# #     node_counter = 0
# #     pending_parents = {}  # Maps state -> list of node_ids that can be expanded
    
# #     for frame_idx, (parent_state, children_states) in enumerate(expansions):
# #         # Find which node ID to expand
# #         if frame_idx == 0:
# #             # Root node
# #             parent_id = f"n{node_counter}"
# #             node_counter += 1
# #             G.add_node(parent_id, label=parent_state, state=parent_state)
# #             pending_parents[parent_state] = [parent_id]
# #         else:
# #             # Get the first available parent node with this state
# #             if parent_state in pending_parents and pending_parents[parent_state]:
# #                 parent_id = pending_parents[parent_state].pop(0)
# #             else:
# #                 print(f"Warning: No pending parent for {parent_state}")
# #                 continue
        
# #         # Create children
# #         child_ids = []
# #         for child_state in children_states:
# #             child_id = f"n{node_counter}"
# #             node_counter += 1
# #             G.add_node(child_id, label=child_state, state=child_state)
# #             G.add_edge(parent_id, child_id)
# #             child_ids.append(child_id)
            
# #             # Add to pending parents
# #             pending_parents.setdefault(child_state, []).append(child_id)
        
# #         all_frames_data.append({
# #             'parent_id': parent_id,
# #             'parent_state': parent_state,
# #             'child_ids': child_ids,
# #             'children_states': children_states
# #         })

# #     # Now animate by showing progressive subgraphs
# #     def update(frame):
# #         ax_tree.clear()
# #         ax_current.clear()
# #         ax_info.clear()

# #         # Build subgraph up to this frame
# #         nodes_to_show = set()
        
# #         for i in range(frame + 1):
# #             frame_data = all_frames_data[i]
# #             nodes_to_show.add(frame_data['parent_id'])
# #             for child_id in frame_data['child_ids']:
# #                 nodes_to_show.add(child_id)
        
# #         # Create a proper subgraph with only the nodes we want
# #         G_sub = nx.DiGraph()
# #         for node in nodes_to_show:
# #             G_sub.add_node(node, label=G.nodes[node]['label'], state=G.nodes[node]['state'])
        
# #         for parent, child in G.edges():
# #             if parent in nodes_to_show and child in nodes_to_show:
# #                 G_sub.add_edge(parent, child)
        
# #         # Current frame info
# #         current_data = all_frames_data[frame]
# #         current_node = current_data['parent_state']
# #         goal_reached = (current_node == goal_node)

# #         # Node colors
# #         colors = []
# #         labels_dict = {}
# #         for n in G_sub.nodes():
# #             node_label = G_sub.nodes[n]['label']
# #             node_state = G_sub.nodes[n]['state']
# #             labels_dict[n] = node_label
            
# #             # Color based on state
# #             if node_state == goal_node and goal_reached:
# #                 colors.append("lightgreen")
# #             elif goal_reached and node_state in solution_path:
# #                 colors.append("gold")
# #             elif n == current_data['parent_id']:
# #                 colors.append("orange")
# #             else:
# #                 colors.append("skyblue")

# #         # Draw with hierarchical layout
# #         if G_sub.number_of_nodes() > 0:
# #             root = list(G_sub.nodes())[0]
# #             draw_tree_hierarchical(G_sub, root, ax_tree, colors, labels_dict)
# #         ax_tree.set_title("Tree Search - Hierarchical View", fontsize=16, fontweight='bold')
# #         ax_tree.axis('off')

# #         # Current expanding node display
# #         current_text = f"Expanding: {current_node}\n\nChildren: {current_data['children_states']}"
# #         current_color = "darkgreen" if goal_reached else "darkorange"
# #         ax_current.text(0.5, 0.5, current_text, ha="center", va="center", fontsize=14,
# #                         fontweight="bold", color=current_color, wrap=True)
# #         ax_current.axis("off")

# #         # Info panel
# #         elapsed = t.time() - start_time
# #         status = "✅ GOAL REACHED!" if goal_reached else "🔍 Searching..."
# #         status_color = "green" if goal_reached else "blue"
# #         info_text = f"Expansion: {frame + 1} / {frames}\n"
# #         info_text += f"Total Nodes: {len(nodes_to_show)}\n"
# #         info_text += f"Time: {elapsed:.2f}s\n\n"
# #         info_text += f"{status}"
        
# #         ax_info.text(0.5, 0.5, info_text,
# #                      ha="center", va="center", fontsize=14, color=status_color,
# #                      fontweight="bold" if goal_reached else "normal")
# #         ax_info.axis("off")

# #         if goal_reached:
# #             ani.event_source.stop()

# #     ani = animation.FuncAnimation(fig, update, frames=frames, interval=800, repeat=False)  # Faster: 800ms
# #     plt.show()


# # animate()

# import numpy as np
# import os
# import time as t
# import warnings
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import networkx as nx
# import re
# from collections import deque

# # Silence matplotlib font emoji warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# #-------------------------------------------------------------------------------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# EXPANSION_FILE = os.path.join(BASE_DIR, "expantion_list.txt")
# MATRIX_FILE = os.path.join(BASE_DIR, "matrix_holder.txt")


# def parse_expansion_list_with_children(path):
#     """Parse expantion_list.txt to extract ALL expansions with their children."""
#     if not os.path.exists(path):
#         print("No expantion_list.txt found at", path)
#         return []

#     expansions = []
#     current_parent = None
    
#     state_pattern = re.compile(r"state\s*:\s*([A-Za-z0-9]+)")
#     children_pattern = re.compile(r"children:\s*\[(.*?)\]")
    
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             m_state = state_pattern.search(line)
#             if m_state:
#                 current_parent = m_state.group(1).strip()
            
#             m_children = children_pattern.search(line)
#             if m_children:
#                 children_str = m_children.group(1)
#                 if children_str.strip():
#                     children = [c.strip().strip("'\"") for c in children_str.split(",")]
#                     current_children = [c for c in children if c]
#                 else:
#                     current_children = []
                
#                 if current_parent:
#                     expansions.append((current_parent, current_children[:]))
#                     current_parent = None
    
#     return expansions


# def animate():
#     expansions = parse_expansion_list_with_children(EXPANSION_FILE)
#     if not expansions:
#         print("No expansions parsed. Make sure expantion_list.txt exists and contains the expansion data.")
#         return

#     frames = len(expansions)
#     goal_node = expansions[-1][0]
    
#     # Extract solution path
#     path_log_path = os.path.join(BASE_DIR, "path_log.txt")
#     solution_path = []
#     if os.path.exists(path_log_path):
#         with open(path_log_path, "r", encoding="utf-8") as f:
#             path_str = f.read().strip()
#             solution_path = [x.strip() for x in path_str.split("<---") if x.strip()]

#     fig = plt.figure(figsize=(16, 8))
#     gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
#     ax_tree = fig.add_subplot(gs[0, 0])
#     right_gs = gs[0, 1].subgridspec(2, 1)
#     ax_current = fig.add_subplot(right_gs[0])
#     ax_info = fig.add_subplot(right_gs[1])
#     plt.tight_layout(pad=3)

#     start_time = t.time()

#     # NEW APPROACH: Build tree by simulating the actual search
#     # Each node in fringe can potentially be expanded
#     G = nx.DiGraph()
#     node_id = 0
#     fringe = deque()  # Queue of node_ids that can be expanded
#     node_info = {}  # node_id -> (state, parent_id)
    
#     # Create root
#     root_state = expansions[0][0]
#     root_id = f"n{node_id}"
#     node_id += 1
#     G.add_node(root_id, label=root_state, state=root_state)
#     node_info[root_id] = (root_state, None)
#     fringe.append(root_id)
    
#     # Process each expansion
#     frame_data = []
    
#     for exp_idx, (parent_state, children_states) in enumerate(expansions):
#         # Find node to expand - the FIRST node in fringe with matching state
#         parent_node_id = None
        
#         for node in list(fringe):
#             if node_info[node][0] == parent_state:
#                 parent_node_id = node
#                 fringe.remove(node)
#                 break
        
#         if parent_node_id is None:
#             print(f"ERROR: Cannot find node to expand for state {parent_state}")
#             continue
        
#         # Create children
#         child_ids = []
#         for child_state in children_states:
#             child_id = f"n{node_id}"
#             node_id += 1
#             G.add_node(child_id, label=child_state, state=child_state)
#             G.add_edge(parent_node_id, child_id)
#             node_info[child_id] = (child_state, parent_node_id)
#             fringe.append(child_id)
#             child_ids.append(child_id)
        
#         frame_data.append({
#             'parent_id': parent_node_id,
#             'parent_state': parent_state,
#             'child_ids': child_ids,
#             'children_states': children_states,
#             'nodes_so_far': list(G.nodes()),
#             'edges_so_far': list(G.edges())
#         })
    
#     print(f"Built tree with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

#     def hierarchical_pos(G, root):
#         """Calculate positions using proper BFS levels."""
#         levels = {}
#         queue = [(root, 0)]
#         visited = {root}
        
#         while queue:
#             node, level = queue.pop(0)
#             levels[node] = level
            
#             for child in G.successors(node):
#                 if child not in visited:
#                     visited.add(child)
#                     queue.append((child, level + 1))
        
#         # Group by level
#         by_level = {}
#         for node, level in levels.items():
#             by_level.setdefault(level, []).append(node)
        
#         # Sort for consistency
#         for level in by_level:
#             by_level[level].sort()
        
#         # Calculate positions
#         pos = {}
#         max_level = max(levels.values()) if levels else 0
        
#         for level, nodes in by_level.items():
#             y = 1.0 - (level / max(1, max_level + 1))
#             n = len(nodes)
            
#             for i, node in enumerate(nodes):
#                 if n == 1:
#                     x = 0.5
#                 else:
#                     x = 0.1 + 0.8 * i / (n - 1)
#                 pos[node] = (x, y)
        
#         return pos

#     def update(frame):
#         ax_tree.clear()
#         ax_current.clear()
#         ax_info.clear()

#         # Get current frame data
#         data = frame_data[frame]
#         current_state = data['parent_state']
#         goal_reached = (current_state == goal_node)
        
#         # Build subgraph up to this frame
#         G_sub = nx.DiGraph()
#         for node in data['nodes_so_far']:
#             G_sub.add_node(node, label=G.nodes[node]['label'], state=G.nodes[node]['state'])
#         for edge in data['edges_so_far']:
#             if edge[0] in G_sub and edge[1] in G_sub:
#                 G_sub.add_edge(edge[0], edge[1])
        
#         # Calculate positions
#         root = list(G_sub.nodes())[0]
#         pos = hierarchical_pos(G_sub, root)
        
#         # Colors
#         colors = []
#         labels = {}
#         for node in G_sub.nodes():
#             state = G_sub.nodes[node]['state']
#             labels[node] = G_sub.nodes[node]['label']
            
#             if state == goal_node and goal_reached:
#                 colors.append("lightgreen")
#             elif goal_reached and state in solution_path:
#                 colors.append("gold")
#             elif node == data['parent_id']:
#                 colors.append("orange")
#             else:
#                 colors.append("skyblue")
        
#         # Draw
#         nx.draw(G_sub, pos, labels=labels, with_labels=True, arrows=True,
#                 node_color=colors, ax=ax_tree, node_size=800, font_size=12,
#                 font_weight='bold', edge_color='gray', arrowsize=15, width=2)
        
#         ax_tree.set_title("Tree Search - Hierarchical View", fontsize=16, fontweight='bold')
#         ax_tree.axis('off')
#         ax_tree.set_xlim(-0.05, 1.05)
#         ax_tree.set_ylim(-0.05, 1.05)

#         # Info panels
#         current_text = f"Expanding: {current_state}\n\nChildren: {data['children_states']}"
#         current_color = "darkgreen" if goal_reached else "darkorange"
#         ax_current.text(0.5, 0.5, current_text, ha="center", va="center", fontsize=14,
#                         fontweight="bold", color=current_color)
#         ax_current.axis("off")

#         elapsed = t.time() - start_time
#         status = "✅ GOAL REACHED!" if goal_reached else "🔍 Searching..."
#         status_color = "green" if goal_reached else "blue"
#         info_text = f"Expansion: {frame + 1} / {frames}\n"
#         info_text += f"Total Nodes: {len(data['nodes_so_far'])}\n"
#         info_text += f"Time: {elapsed:.2f}s\n\n{status}"
        
#         ax_info.text(0.5, 0.5, info_text, ha="center", va="center", fontsize=14,
#                      color=status_color, fontweight="bold" if goal_reached else "normal")
#         ax_info.axis("off")

#         if goal_reached:
#             ani.event_source.stop()

#     ani = animation.FuncAnimation(fig, update, frames=frames, interval=800, repeat=False)
#     plt.show()


# animate()

import numpy as np
import os
import time as t
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import re
from collections import deque

# Silence matplotlib font emoji warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

#-------------------------------------------------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPANSION_FILE = os.path.join(BASE_DIR, "expantion_list.txt")
MATRIX_FILE = os.path.join(BASE_DIR, "matrix_holder.txt")


def parse_expansion_list_with_children(path):
    """Parse expantion_list.txt to extract ALL expansions with their children."""
    if not os.path.exists(path):
        print("No expantion_list.txt found at", path)
        return []

    expansions = []
    current_parent = None
    
    state_pattern = re.compile(r"state\s*:\s*([A-Za-z0-9]+)")
    children_pattern = re.compile(r"children:\s*\[(.*?)\]")
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m_state = state_pattern.search(line)
            if m_state:
                current_parent = m_state.group(1).strip()
            
            m_children = children_pattern.search(line)
            if m_children:
                children_str = m_children.group(1)
                if children_str.strip():
                    children = [c.strip().strip("'\"") for c in children_str.split(",")]
                    current_children = [c for c in children if c]
                else:
                    current_children = []
                
                if current_parent:
                    expansions.append((current_parent, current_children[:]))
                    current_parent = None
    
    return expansions


def animate():
    expansions = parse_expansion_list_with_children(EXPANSION_FILE)
    if not expansions:
        print("No expansions parsed. Make sure expantion_list.txt exists and contains the expansion data.")
        return

    frames = len(expansions)
    goal_node = expansions[-1][0]
    
    # Extract solution path
    path_log_path = os.path.join(BASE_DIR, "path_log.txt")
    solution_path = []
    if os.path.exists(path_log_path):
        with open(path_log_path, "r", encoding="utf-8") as f:
            path_str = f.read().strip()
            solution_path = [x.strip() for x in path_str.split("<---") if x.strip()]

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
    ax_tree = fig.add_subplot(gs[0, 0])
    right_gs = gs[0, 1].subgridspec(2, 1)
    ax_current = fig.add_subplot(right_gs[0])
    ax_info = fig.add_subplot(right_gs[1])
    plt.tight_layout(pad=3)

    start_time = t.time()

    # NEW APPROACH: Build tree by simulating the actual search
    # Each node in fringe can potentially be expanded
    G = nx.DiGraph()
    node_id = 0
    fringe = deque()  # Queue of node_ids that can be expanded
    node_info = {}  # node_id -> (state, parent_id)
    
    # Create root
    root_state = expansions[0][0]
    root_id = f"n{node_id}"
    node_id += 1
    G.add_node(root_id, label=root_state, state=root_state)
    node_info[root_id] = (root_state, None)
    fringe.append(root_id)
    
    # Process each expansion
    frame_data = []
    
    for exp_idx, (parent_state, children_states) in enumerate(expansions):
        # Find node to expand - the FIRST node in fringe with matching state
        parent_node_id = None
        
        for node in list(fringe):
            if node_info[node][0] == parent_state:
                parent_node_id = node
                fringe.remove(node)
                break
        
        if parent_node_id is None:
            print(f"ERROR: Cannot find node to expand for state {parent_state}")
            continue
        
        # Create children
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
            'nodes_so_far': list(G.nodes()),
            'edges_so_far': list(G.edges())
        })
    
    print(f"Built tree with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    def hierarchical_pos(G, root):
        """Calculate positions using proper tree layout with subtree-aware spacing."""
        
        # Calculate depth for each node
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
        
        # Calculate subtree sizes (number of leaves under each node)
        def count_leaves(node):
            children = list(G.successors(node))
            if not children:
                return 1
            return sum(count_leaves(child) for child in children)
        
        subtree_sizes = {}
        for node in G.nodes():
            subtree_sizes[node] = count_leaves(node)
        
        # Assign x positions based on in-order traversal
        positions = {}
        x_counter = [0]  # Use list to make it mutable in nested function
        max_depth = max(depths.values()) if depths else 0
        
        def assign_x(node):
            """Assign x-coordinate using in-order traversal."""
            children = sorted(G.successors(node))
            
            if not children:
                # Leaf node - assign next x position
                positions[node] = x_counter[0]
                x_counter[0] += 1
            else:
                # Process left children
                for child in children[:-1]:
                    assign_x(child)
                
                # Position this node at center of its children
                if len(children) > 0:
                    # Process last child
                    assign_x(children[-1])
                    
                    # Position parent at center of all children
                    child_positions = [positions[c] for c in children]
                    positions[node] = (min(child_positions) + max(child_positions)) / 2.0
        
        assign_x(root)
        
        # Normalize x positions to [0, 1] range
        if positions:
            min_x = min(positions.values())
            max_x = max(positions.values())
            x_range = max_x - min_x if max_x > min_x else 1
            
            pos = {}
            for node in positions:
                normalized_x = 0.1 + 0.8 * (positions[node] - min_x) / x_range
                y = 1.0 - (depths[node] / max(1, max_depth + 1))
                pos[node] = (normalized_x, y)
            
            return pos
        
        return {root: (0.5, 1.0)}

    def update(frame):
        ax_tree.clear()
        ax_current.clear()
        ax_info.clear()

        # Get current frame data
        data = frame_data[frame]
        current_state = data['parent_state']
        goal_reached = (current_state == goal_node)
        
        # Build subgraph up to this frame
        G_sub = nx.DiGraph()
        for node in data['nodes_so_far']:
            G_sub.add_node(node, label=G.nodes[node]['label'], state=G.nodes[node]['state'])
        for edge in data['edges_so_far']:
            if edge[0] in G_sub and edge[1] in G_sub:
                G_sub.add_edge(edge[0], edge[1])
        
        # Calculate positions
        root = list(G_sub.nodes())[0]
        pos = hierarchical_pos(G_sub, root)
        
        # Colors
        colors = []
        labels = {}
        for node in G_sub.nodes():
            state = G_sub.nodes[node]['state']
            labels[node] = G_sub.nodes[node]['label']
            
            if state == goal_node and goal_reached:
                colors.append("lightgreen")
            elif goal_reached and state in solution_path:
                colors.append("gold")
            elif node == data['parent_id']:
                colors.append("orange")
            else:
                colors.append("skyblue")
        
        # Draw
        nx.draw(G_sub, pos, labels=labels, with_labels=True, arrows=True,
                node_color=colors, ax=ax_tree, node_size=800, font_size=12,
                font_weight='bold', edge_color='gray', arrowsize=15, width=2)
        
        ax_tree.set_title("Tree Search - Hierarchical View", fontsize=16, fontweight='bold')
        ax_tree.axis('off')
        ax_tree.set_xlim(-0.05, 1.05)
        ax_tree.set_ylim(-0.05, 1.05)

        # Info panels
        current_text = f"Expanding: {current_state}\n\nChildren: {data['children_states']}"
        current_color = "darkgreen" if goal_reached else "darkorange"
        ax_current.text(0.5, 0.5, current_text, ha="center", va="center", fontsize=14,
                        fontweight="bold", color=current_color)
        ax_current.axis("off")

        elapsed = t.time() - start_time
        status = "✅ GOAL REACHED!" if goal_reached else "🔍 Searching..."
        status_color = "green" if goal_reached else "blue"
        info_text = f"Expansion: {frame + 1} / {frames}\n"
        info_text += f"Total Nodes: {len(data['nodes_so_far'])}\n"
        info_text += f"Time: {elapsed:.2f}s\n\n{status}"
        
        ax_info.text(0.5, 0.5, info_text, ha="center", va="center", fontsize=14,
                     color=status_color, fontweight="bold" if goal_reached else "normal")
        ax_info.axis("off")

        if goal_reached:
            ani.event_source.stop()

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=800, repeat=False)
    plt.show()


animate()
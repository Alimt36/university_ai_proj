# import numpy as np
# import os
# import time as t
# import warnings
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import networkx as nx
# import re
# from collections import deque

# warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# #-------------------------------------------------------------------------------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# EXPANSION_FILE = os.path.join(BASE_DIR, "expantion_list.txt")
# MATRIX_FILE = os.path.join(BASE_DIR, "matrix_holder.txt")


# def parse_expansion_list_with_children(path):
#     if not os.path.exists(path):
#         print("No expantion_list.txt found at", path)
#         return []

#     expansions = []
#     current_parent = None
#     current_depth = None
#     current_path_cost = None
#     current_parent_state = None

#     state_pattern = re.compile(r"state\s*:\s*([A-Za-z0-9]+)")
#     children_pattern = re.compile(r"children:\s*\[(.*?)\]")
#     depth_pattern = re.compile(r"depth\s*:\s*(\d+)")
#     path_cost_pattern = re.compile(r"path_cost\s*:\s*([\d.]+)")
#     parent_pattern = re.compile(r"parent\s*:\s*([A-Za-z0-9]+)")

#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             m_state = state_pattern.search(line)
#             if m_state:
#                 current_parent = m_state.group(1).strip()

#             m_depth = depth_pattern.search(line)
#             if m_depth:
#                 current_depth = int(m_depth.group(1))

#             m_path_cost = path_cost_pattern.search(line)
#             if m_path_cost:
#                 current_path_cost = float(m_path_cost.group(1))

#             m_parent = parent_pattern.search(line)
#             if m_parent:
#                 current_parent_state = m_parent.group(1).strip()

#             m_children = children_pattern.search(line)
#             if m_children:
#                 children_str = m_children.group(1)
#                 if children_str.strip():
#                     children = [c.strip().strip("'\"") for c in children_str.split(",")]
#                     current_children = [c for c in children if c]
#                 else:
#                     current_children = []

#                 if current_parent:
#                     expansions.append({
#                         'state': current_parent,
#                         'children': current_children[:],
#                         'depth': current_depth,
#                         'path_cost': current_path_cost,
#                         'parent_state': current_parent_state
#                     })
#                     current_parent = None
#                     current_depth = None
#                     current_path_cost = None
#                     current_parent_state = None

#     return expansions


# def animate():
#     expansions = parse_expansion_list_with_children(EXPANSION_FILE)
#     if not expansions:
#         print("No expansions parsed. Make sure expantion_list.txt exists and contains the expansion data.")
#         return

#     frames = len(expansions)

#     # --- Read goal and path correctly ---
#     PATH_FILE = os.path.join(BASE_DIR, "path_log.txt")
#     goal_node = None
#     final_path_text = ""
#     if os.path.exists(PATH_FILE):
#         with open(PATH_FILE, "r", encoding="utf-8") as f:
#             final_path_text = f.read().strip()
#             parts = [x.strip() for x in re.split(r"<---", final_path_text)]
#             goal_node = parts[0] if parts else None
#     else:
#         final_path_text = "(path_log.txt not found)"

#     # Extract path nodes for highlighting
#     path_nodes = []
#     if "<---" in final_path_text:
#         path_nodes = [x.strip() for x in final_path_text.split("<---")]

#     # --- build frames ---
#     fig = plt.figure(figsize=(16, 8))
#     gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
#     ax_tree = fig.add_subplot(gs[0, 0])
#     right_gs = gs[0, 1].subgridspec(2, 1)
#     ax_current = fig.add_subplot(right_gs[0])
#     ax_info = fig.add_subplot(right_gs[1])
#     plt.tight_layout(pad=3)

#     start_time = t.time()
#     G = nx.DiGraph()
#     node_id = 0
#     fringe = deque()
#     node_info = {}

#     root_state = expansions[0]['state']
#     root_id = f"n{node_id}"
#     node_id += 1
#     G.add_node(root_id, label=root_state, state=root_state)
#     node_info[root_id] = (root_state, None)
#     fringe.append(root_id)

#     frame_data = []
#     for exp_idx, expansion in enumerate(expansions):
#         parent_state = expansion['state']
#         children_states = expansion['children']
#         depth = expansion.get('depth', 0)
#         path_cost = expansion.get('path_cost', 0)
#         parent_state_of_node = expansion.get('parent_state', 'None')

#         parent_node_id = None
#         for node in list(fringe):
#             if node_info[node][0] == parent_state:
#                 parent_node_id = node
#                 fringe.remove(node)
#                 break

#         if parent_node_id is None:
#             continue

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
#             'depth': depth,
#             'path_cost': path_cost,
#             'parent_of_node': parent_state_of_node,
#             'nodes_so_far': list(G.nodes()),
#             'edges_so_far': list(G.edges())
#         })

#     def hierarchical_pos(G, root):
#         depths = {}
#         queue = [(root, 0)]
#         visited = {root}
#         while queue:
#             node, depth = queue.pop(0)
#             depths[node] = depth
#             for child in G.successors(node):
#                 if child not in visited:
#                     visited.add(child)
#                     queue.append((child, depth + 1))
#         positions = {}
#         x_counter = [0]
#         max_depth = max(depths.values()) if depths else 0

#         def assign_x(node):
#             children = sorted(G.successors(node))
#             if not children:
#                 positions[node] = x_counter[0]
#                 x_counter[0] += 1
#             else:
#                 for c in children[:-1]:
#                     assign_x(c)
#                 assign_x(children[-1])
#                 child_positions = [positions[c] for c in children]
#                 positions[node] = (min(child_positions) + max(child_positions)) / 2.0

#         assign_x(root)
#         if positions:
#             min_x, max_x = min(positions.values()), max(positions.values())
#             x_range = max_x - min_x if max_x > min_x else 1
#             pos = {}
#             for n in positions:
#                 x = 0.1 + 0.8 * (positions[n] - min_x) / x_range
#                 y = 1.0 - (depths[n] / max(1, max_depth + 1))
#                 pos[n] = (x, y)
#             return pos
#         return {root: (0.5, 1.0)}

#     def update(frame):
#         ax_tree.clear()
#         ax_current.clear()
#         ax_info.clear()

#         data = frame_data[frame]
#         current_state = data['parent_state']
#         goal_reached = (current_state == goal_node)

#         G_sub = nx.DiGraph()
#         for n in data['nodes_so_far']:
#             G_sub.add_node(n, label=G.nodes[n]['label'], state=G.nodes[n]['state'])
#         for e in data['edges_so_far']:
#             if e[0] in G_sub and e[1] in G_sub:
#                 G_sub.add_edge(e[0], e[1])

#         root = list(G_sub.nodes())[0]
#         pos = hierarchical_pos(G_sub, root)

#         # --- رنگ گره‌ها (اصلاح‌شده) ---
#         colors = []
#         labels = {}
#         colored_path_nodes = set()  # برای جلوگیری از رنگ تکراری‌ها

#         for n in G_sub.nodes():
#             s = G_sub.nodes[n]['state']
#             labels[n] = G_sub.nodes[n]['label']

#             if s == goal_node:
#                 colors.append("lightgreen")
#             elif s in path_nodes and s not in colored_path_nodes:
#                 colors.append("#FF6666")  # مسیر نهایی فقط برای اولین occurrence
#                 colored_path_nodes.add(s)
#             elif n == data['parent_id']:
#                 colors.append("orange")
#             else:
#                 colors.append("skyblue")

#         # --- رنگ یال‌ها ---
#         edge_colors = []
#         for (u, v) in G_sub.edges():
#             u_state = G_sub.nodes[u]['state']
#             v_state = G_sub.nodes[v]['state']
#             if u_state in path_nodes and v_state in path_nodes:
#                 i1, i2 = path_nodes.index(u_state), path_nodes.index(v_state)
#                 if abs(i1 - i2) == 1:
#                     edge_colors.append("red")
#                 else:
#                     edge_colors.append("gray")
#             else:
#                 edge_colors.append("gray")

#         nx.draw(G_sub, pos, labels=labels, with_labels=True, arrows=True,
#                 node_color=colors, ax=ax_tree, node_size=800, font_size=12,
#                 font_weight='bold', edge_color=edge_colors, arrowsize=15, width=2)
#         ax_tree.axis('off')

#         current_text = f"Currently Expanding Node:\n━━━━━━━━━━━━━━━━━━━━━\nState: {current_state}\nDepth: {data.get('depth','N/A')}\nPath Cost: {data.get('path_cost','N/A')}\nParent: {data.get('parent_of_node','None')}\n━━━━━━━━━━━━━━━━━━━━━\nChildren: {data['children_states']}\n"
#         ax_current.text(0.5, 0.5, current_text, ha="center", va="center", fontsize=11,
#                         fontweight="bold", color="darkorange", family='monospace')
#         ax_current.axis("off")

#         ax_info.axis("off")
#         if goal_reached:
#             info_text = f"GOAL REACHED!\n\n{final_path_text}"
#             ax_info.text(0.5, 0.5, info_text, ha="center", va="center",
#                          fontsize=12, color="green", fontweight="bold",
#                          family="monospace")
#             ani.event_source.stop()
#         else:
#             elapsed = t.time() - start_time
#             ax_info.text(0.5, 0.5,
#                          f"Searching...\nTime: {elapsed:.2f}s\nExpansion: {frame+1}/{frames}",
#                          ha="center", va="center", fontsize=12, color="blue")

#     ani = animation.FuncAnimation(fig, update, frames=frames, interval=800, repeat=False)
#     plt.show()


# animate()

# # animate.py (به‌روز شده مطابق درخواست شما)
# import os
# import re
# import time as t
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import networkx as nx

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# EXPANSION_FILE = os.path.join(BASE_DIR, "expantion_list.txt")
# PATH_FILE = os.path.join(BASE_DIR, "path_log.txt")

# def parse_expansion_list_with_children(path):
#     """
#     پارس کردن فایل expantion_list.txt
#     خروجی: لیستی از دیکشنری‌ها با کلیدهای: state, children, depth, path_cost, parent_state
#     """
#     if not os.path.exists(path):
#         print("No expantion_list.txt found at", path)
#         return []

#     expansions = []
#     current_parent = None
#     current_depth = None
#     current_path_cost = None
#     current_parent_state = None

#     state_pattern = re.compile(r"state\s*:\s*([A-Za-z0-9]+)")
#     children_pattern = re.compile(r"children:\s*\[(.*?)\]")
#     depth_pattern = re.compile(r"depth\s*:\s*(\d+)")
#     path_cost_pattern = re.compile(r"path_cost\s*:\s*([\d.]+)")
#     parent_pattern = re.compile(r"parent\s*:\s*([A-Za-z0-9]+)")

#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             m_state = state_pattern.search(line)
#             if m_state:
#                 current_parent = m_state.group(1).strip()

#             m_depth = depth_pattern.search(line)
#             if m_depth:
#                 current_depth = int(m_depth.group(1))

#             m_path_cost = path_cost_pattern.search(line)
#             if m_path_cost:
#                 current_path_cost = float(m_path_cost.group(1))

#             m_parent = parent_pattern.search(line)
#             if m_parent:
#                 current_parent_state = m_parent.group(1).strip()

#             m_children = children_pattern.search(line)
#             if m_children:
#                 children_str = m_children.group(1)
#                 if children_str.strip():
#                     children = [c.strip().strip("'\"") for c in children_str.split(",")]
#                     current_children = [c for c in children if c]
#                 else:
#                     current_children = []

#                 if current_parent:
#                     expansions.append({
#                         'state': current_parent,
#                         'children': current_children[:],
#                         'depth': current_depth,
#                         'path_cost': current_path_cost,
#                         'parent_state': current_parent_state
#                     })
#                     current_parent = None
#                     current_depth = None
#                     current_path_cost = None
#                     current_parent_state = None

#     return expansions

# def read_final_path(path_file):
#     """
#     خواندن path_log.txt و استخراج مسیر نودها به صورت لیست ['A','B','C',...]
#     """
#     if not os.path.exists(path_file):
#         return []
#     with open(path_file, "r", encoding="utf-8") as f:
#         txt = f.read().strip()
#         if not txt:
#             return []
#         nodes = [x.strip() for x in txt.split("<---")]
#         return nodes

# def animate():
#     expansions = parse_expansion_list_with_children(EXPANSION_FILE)
#     if not expansions:
#         print("No expansions parsed. Make sure expantion_list.txt exists and contains the expansion data.")
#         return

#     final_path_nodes = read_final_path(PATH_FILE)  # مسیر نهایی از فایل path_log.txt
#     goal_node = final_path_nodes[-1] if final_path_nodes else None

#     # --- تنظیمات شکل ---
#     frames = len(expansions) + 1  # +1 برای قاب نهایی که درخت کامل و مسیر را نشان می‌دهد
#     fig = plt.figure(figsize=(16, 8))
#     gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
#     ax_tree = fig.add_subplot(gs[0, 0])
#     right_gs = gs[0, 1].subgridspec(2, 1)
#     ax_current = fig.add_subplot(right_gs[0])
#     ax_info = fig.add_subplot(right_gs[1])
#     plt.tight_layout(pad=3)

#     # --- داده‌های ساخت گراف (حفظ نگاشت state -> node_id برای جلوگیری از تکرار) ---
#     G = nx.DiGraph()
#     state_to_nodeid = {}   # mapping e.g. 'A' -> 'n3'
#     node_info = {}         # 'n3' -> (state, parent_nodeid)
#     edges_set = set()      # مجموعهٔ ایج‌ها برای جلوگیری از duplicate (parentid, childid)

#     node_counter = 0
#     fringe_order = []  # فقط برای ترتیبِ بصری (نه برای منطق)

#     # --- build frames data (but ensure nodes/edges unique) ---
#     frame_data = []
#     for expansion in expansions:
#         parent_state = expansion['state']
#         children_states = expansion['children']
#         depth = expansion.get('depth', None)
#         path_cost = expansion.get('path_cost', None)
#         parent_of_node = expansion.get('parent_state', None)

#         # اگر parent از قبل وجود ندارد، بساز
#         if parent_state not in state_to_nodeid:
#             nid = f"n{node_counter}"
#             node_counter += 1
#             state_to_nodeid[parent_state] = nid
#             G.add_node(nid, label=parent_state, state=parent_state)
#             node_info[nid] = (parent_state, None)
#             fringe_order.append(nid)

#         parent_id = state_to_nodeid[parent_state]

#         added_child_ids = []
#         added_children_states = []

#         for cstate in children_states:
#             # اگر child از قبل وجود دارد، از همان node_id استفاده کن
#             if cstate in state_to_nodeid:
#                 child_id = state_to_nodeid[cstate]
#             else:
#                 child_id = f"n{node_counter}"
#                 node_counter += 1
#                 state_to_nodeid[cstate] = child_id
#                 G.add_node(child_id, label=cstate, state=cstate)
#                 node_info[child_id] = (cstate, parent_id)
#                 fringe_order.append(child_id)

#             # اگر این ایج قبلاً ساخته نشده، اضافه‌ش کن؛ در غیر این صورت نادیده بگیر
#             if (parent_id, child_id) not in edges_set:
#                 G.add_edge(parent_id, child_id)
#                 edges_set.add((parent_id, child_id))
#                 added_child_ids.append(child_id)
#                 added_children_states.append(cstate)
#             else:
#                 # مسیر قبلاً ساخته شده — نساختن دوباره
#                 pass

#         frame_data.append({
#             'parent_id': parent_id,
#             'parent_state': parent_state,
#             'added_child_ids': added_child_ids,
#             'added_children_states': added_children_states,
#             'depth': depth,
#             'path_cost': path_cost,
#             'parent_of_node': parent_of_node,
#             'nodes_so_far': list(G.nodes()),
#             'edges_so_far': list(G.edges())
#         })

#     # --- تابع تعیین موقعیت سلسله‌مراتبی (hierarchical) ---
#     def hierarchical_pos(G_sub, root):
#         depths = {}
#         queue = [(root, 0)]
#         visited = {root}
#         while queue:
#             node, depth = queue.pop(0)
#             depths[node] = depth
#             for child in G_sub.successors(node):
#                 if child not in visited:
#                     visited.add(child)
#                     queue.append((child, depth + 1))
#         positions = {}
#         x_counter = [0]
#         max_depth = max(depths.values()) if depths else 0

#         def assign_x(node):
#             children = sorted(list(G_sub.successors(node)))
#             if not children:
#                 positions[node] = x_counter[0]
#                 x_counter[0] += 1
#             else:
#                 for c in children[:-1]:
#                     assign_x(c)
#                 assign_x(children[-1])
#                 child_positions = [positions[c] for c in children]
#                 positions[node] = (min(child_positions) + max(child_positions)) / 2.0

#         assign_x(root)
#         if positions:
#             min_x, max_x = min(positions.values()), max(positions.values())
#             x_range = max_x - min_x if max_x > min_x else 1
#             pos = {}
#             for n in positions:
#                 x = 0.05 + 0.9 * (positions[n] - min_x) / x_range
#                 y = 1.0 - (depths[n] / max(1, max_depth + 1))
#                 pos[n] = (x, y)
#             return pos
#         return {root: (0.5, 1.0)}

#     start_time = t.time()

#     def update(frame):
#         ax_tree.clear()
#         ax_current.clear()
#         ax_info.clear()

#         # اگر frame < len(frame_data) => نمایش مرحله‌ای تا وقتی که جست‌وجو تموم نشده
#         # اگر frame == len(frame_data) => قاب نهایی: نمایش درخت کامل + رنگ مسیر نهایی
#         is_final_frame = (frame == len(frame_data))

#         if not is_final_frame:
#             data = frame_data[frame]
#             nodes_in_frame = data['nodes_so_far']
#             edges_in_frame = data['edges_so_far']
#             # ساخت گراف فرعی برای رسم این فریم
#             G_sub = nx.DiGraph()
#             for n in nodes_in_frame:
#                 G_sub.add_node(n, label=G.nodes[n]['label'], state=G.nodes[n]['state'])
#             for e in edges_in_frame:
#                 if e[0] in G_sub and e[1] in G_sub:
#                     G_sub.add_edge(e[0], e[1])

#             root = list(G_sub.nodes())[0]
#             pos = hierarchical_pos(G_sub, root)

#             # رنگ‌ها: گرهِ که الان expand شده (parent) نارنجی، بقیه آبی روشن، مسیر نهایی هنوز خاکستری
#             colors = []
#             labels = {}
#             for n in G_sub.nodes():
#                 s = G_sub.nodes[n]['state']
#                 labels[n] = G_sub.nodes[n]['label']
#                 if n == data['parent_id']:
#                     colors.append("orange")
#                 else:
#                     colors.append("skyblue")

#             # رنگ یال‌ها به صورت پیش‌فرض خاکستری؛ اگر یالی جزو مسیر نهایی بود در فریم نهایی قرمز خواهد شد
#             edge_colors = ["gray" for _ in G_sub.edges()]

#             nx.draw(G_sub, pos, labels=labels, with_labels=True, arrows=True,
#                     node_color=colors, ax=ax_tree, node_size=800, font_size=12,
#                     font_weight='bold', edge_color=edge_colors, arrowsize=15, width=2)
#             ax_tree.axis('off')

#             # اطلاعات کنار
#             current_text = (f"Currently Expanding Node:\n━━━━━━━━━━━━━━━━━━━━━\n"
#                             f"State: {data['parent_state']}\nDepth: {data.get('depth','N/A')}\n"
#                             f"Path Cost: {data.get('path_cost','N/A')}\nParent: {data.get('parent_of_node','None')}\n"
#                             f"━━━━━━━━━━━━━━━━━━━━━\nChildren added: {data['added_children_states']}\n")
#             ax_current.text(0.5, 0.5, current_text, ha="center", va="center", fontsize=11,
#                             fontweight="bold", color="darkorange", family='monospace')
#             ax_current.axis("off")

#             elapsed = t.time() - start_time
#             ax_info.axis("off")
#             ax_info.text(0.5, 0.5,
#                          f"Searching...\nTime: {elapsed:.2f}s\nExpansion: {frame+1}/{len(frame_data)}",
#                          ha="center", va="center", fontsize=12, color="blue")
#         else:
#             # قاب نهایی: درخت کامل + رنگ مسیر نهایی
#             nodes_in_frame = list(G.nodes())
#             edges_in_frame = list(G.edges())
#             G_sub = nx.DiGraph()
#             for n in nodes_in_frame:
#                 G_sub.add_node(n, label=G.nodes[n]['label'], state=G.nodes[n]['state'])
#             for e in edges_in_frame:
#                 G_sub.add_edge(e[0], e[1])

#             root = list(G_sub.nodes())[0]
#             pos = hierarchical_pos(G_sub, root)

#             # آماده‌سازی رنگ‌ها: مسیر نهایی قرمز (nodes & edges)، goal سبز، بقیه آبی
#             node_colors = []
#             labels = {}
#             path_state_set = set(final_path_nodes)

#             for n in G_sub.nodes():
#                 s = G_sub.nodes[n]['state']
#                 labels[n] = G_sub.nodes[n]['label']
#                 if s == goal_node:
#                     node_colors.append("lightgreen")
#                 elif s in path_state_set:
#                     node_colors.append("#FF6666")  # مسیر نهایی
#                 else:
#                     node_colors.append("skyblue")

#             edge_colors = []
#             # برای هر یال، اگر هر دو سر یال در مسیر نهایی و ترتیبشان در مسیر مجاور است -> قرمز
#             for (u, v) in G_sub.edges():
#                 u_state = G_sub.nodes[u]['state']
#                 v_state = G_sub.nodes[v]['state']
#                 if u_state in path_state_set and v_state in path_state_set:
#                     # چک ترتیب در path
#                     try:
#                         i1 = final_path_nodes.index(u_state)
#                         i2 = final_path_nodes.index(v_state)
#                         if i2 - i1 == 1:
#                             edge_colors.append("red")
#                         else:
#                             edge_colors.append("gray")
#                     except ValueError:
#                         edge_colors.append("gray")
#                 else:
#                     edge_colors.append("gray")

#             nx.draw(G_sub, pos, labels=labels, with_labels=True, arrows=True,
#                     node_color=node_colors, ax=ax_tree, node_size=800, font_size=12,
#                     font_weight='bold', edge_color=edge_colors, arrowsize=15, width=2)
#             ax_tree.axis('off')

#             # کنار ابزارک: نمایش مسیر نهایی
#             path_text = " <--- ".join(final_path_nodes) if final_path_nodes else "(no path)"
#             ax_current.text(0.5, 0.5, f"Search Finished\n\nFinal path:\n{path_text}",
#                             ha="center", va="center", fontsize=12, fontweight="bold", family='monospace')
#             ax_current.axis("off")

#             elapsed = t.time() - start_time
#             ax_info.axis("off")
#             ax_info.text(0.5, 0.5, f"Finished in {elapsed:.2f}s\nNodes: {len(G_sub.nodes())}\nEdges: {len(G_sub.edges())}",
#                          ha="center", va="center", fontsize=12, color="green", fontweight="bold")

#     ani = animation.FuncAnimation(fig, update, frames=frames, interval=650, repeat=False)
#     plt.show()

# if __name__ == "__main__":
#     animate()
#----------------------------------------------------------------


# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import networkx as nx

# # فایل‌ها را بخوان
# with open("expantion_list.txt", "r") as f:
#     expansion_list = [line.strip() for line in f.readlines() if line.strip()]

# with open("path_log.txt", "r") as f:
#     path_nodes = [node.strip() for node in f.readlines() if node.strip()]

# # گراف خالی بساز
# G = nx.DiGraph()

# # همه گره‌ها را از expansion_list و path_nodes اضافه کن
# all_nodes = set(expansion_list + path_nodes)
# for node in all_nodes:
#     G.add_node(node)

# # برای زیبایی، یال‌ها را از ترتیب expansion_list بساز
# for i in range(len(expansion_list) - 1):
#     G.add_edge(expansion_list[i], expansion_list[i + 1])

# # موقعیت گره‌ها را تنظیم کن
# pos = nx.spring_layout(G, seed=42)

# # تنظیمات اولیه‌ی نمودار
# fig, ax = plt.subplots(figsize=(8, 6))
# plt.axis("off")

# def update(frame):
#     ax.clear()
#     plt.axis("off")
#     plt.title(f"Step {frame + 1}/{len(expansion_list)}", fontsize=14)

#     # گره‌ها و یال‌های قبلی را رسم کن
#     expanded = expansion_list[:frame + 1]
#     nx.draw_networkx_nodes(G, pos, nodelist=expanded, node_color='skyblue', node_size=600, ax=ax)
#     nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), edge_color='gray', width=1, ax=ax)
#     nx.draw_networkx_labels(G, pos, font_weight='bold', ax=ax)

#     # اگر به آخرین فریم رسیدیم، مسیر نهایی را قرمز کن
#     if frame == len(expansion_list) - 1:
#         nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_color='red', node_size=600, ax=ax)
#         # یال‌های مسیر نهایی را هم قرمز کن
#         path_edges = list(zip(path_nodes, path_nodes[1:]))
#         nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3, ax=ax)

# ani = animation.FuncAnimation(fig, update, frames=len(expansion_list), interval=1000, repeat=False)
# plt.show()

#-------------------------------------------------------------------------------------------

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

    state_pattern = re.compile(r"state\s*:\s*([A-Za-z0-9]+)")
    children_pattern = re.compile(r"children:\s*\[(.*?)\]")
    depth_pattern = re.compile(r"depth\s*:\s*(\d+)")
    path_cost_pattern = re.compile(r"path_cost\s*:\s*([\d.]+)")
    parent_pattern = re.compile(r"parent\s*:\s*([A-Za-z0-9]+)")

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

    frames = len(expansions)

    # --- Read goal and path correctly ---
    PATH_FILE = os.path.join(BASE_DIR, "path_log.txt")
    goal_node = None
    final_path_text = ""
    if os.path.exists(PATH_FILE):
        with open(PATH_FILE, "r", encoding="utf-8") as f:
            final_path_text = f.read().strip()
            parts = [x.strip() for x in re.split(r"<---", final_path_text)]
            goal_node = parts[0] if parts else None
    else:
        final_path_text = "(path_log.txt not found)"

    # Extract path nodes for highlighting
    path_nodes = []
    if "<---" in final_path_text:
        path_nodes = [x.strip() for x in final_path_text.split("<---")]

    # --- build frames ---
    fig = plt.figure(figsize=(16, 8))
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

    # ==============================================================
    # ✨ بخش اصلی اصلاح‌شده برای رنگ مسیر نهایی در آخرین فریم
    # ==============================================================
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

        # --- رنگ گره‌ها ---
        colors = []
        labels = {}
        for n in G_sub.nodes():
            s = G_sub.nodes[n]['state']
            labels[n] = G_sub.nodes[n]['label']

            if frame == frames - 1:  # فقط در آخرین فریم مسیر قرمز شود
                if s in path_nodes:
                    colors.append("red")
                elif s == goal_node:
                    colors.append("lightgreen")
                elif n == data['parent_id']:
                    colors.append("orange")
                else:
                    colors.append("skyblue")
            else:
                if s == goal_node:
                    colors.append("lightgreen")
                elif n == data['parent_id']:
                    colors.append("orange")
                else:
                    colors.append("skyblue")

        # --- رنگ یال‌ها ---
        edge_colors = []
        for (u, v) in G_sub.edges():
            u_state = G_sub.nodes[u]['state']
            v_state = G_sub.nodes[v]['state']
            if frame == frames - 1 and u_state in path_nodes and v_state in path_nodes:
                i1, i2 = path_nodes.index(u_state), path_nodes.index(v_state)
                if abs(i1 - i2) == 1:
                    edge_colors.append("red")
                else:
                    edge_colors.append("gray")
            else:
                edge_colors.append("gray")

        nx.draw(G_sub, pos, labels=labels, with_labels=True, arrows=True,
                node_color=colors, ax=ax_tree, node_size=800, font_size=12,
                font_weight='bold', edge_color=edge_colors, arrowsize=15, width=2)
        ax_tree.axis('off')

        current_text = f"Currently Expanding Node:\n━━━━━━━━━━━━━━━━━━━━━\nState: {current_state}\nDepth: {data.get('depth','N/A')}\nPath Cost: {data.get('path_cost','N/A')}\nParent: {data.get('parent_of_node','None')}\n━━━━━━━━━━━━━━━━━━━━━\nChildren: {data['children_states']}\n"
        ax_current.text(0.5, 0.5, current_text, ha="center", va="center", fontsize=11,
                        fontweight="bold", color="darkorange", family='monospace')
        ax_current.axis("off")

        ax_info.axis("off")
        if goal_reached or frame == frames - 1:
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

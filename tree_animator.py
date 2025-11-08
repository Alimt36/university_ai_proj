# import matplotlib.pyplot as plt
# import networkx as nx
# import matplotlib.animation as animation
# import os

# # ---------- خواندن گام‌ها ----------
# steps = []
# base_dir = os.path.dirname(os.path.abspath(__file__))
# steps_file = os.path.join(base_dir, "steps_log.txt")
# with open(steps_file, encoding="utf-8") as f:
#     for line in f:
#         if "->" in line:
#             parent, children = line.strip().split(" -> ")
#             children = eval(children)
#             steps.append((parent, children))

# # ---------- خواندن مسیر اصلی ----------
# path_file = os.path.join(base_dir, "path_log.txt")
# with open(path_file, encoding="utf-8") as f:
#     path = f.read().strip().split(" <--- ")

# start_node = path[-1]  # مسیر برگشت داده شده است
# goal_node = path[0]

# fig, ax = plt.subplots(figsize=(12, 8))

# # ---------- تابع layout سلسله‌مراتبی ----------
# def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
#     if pos is None:
#         pos = {root: (xcenter, vert_loc)}
#     else:
#         pos[root] = (xcenter, vert_loc)
#     children = list(G.neighbors(root))
#     if parent is not None:
#         children = [c for c in children if c != parent]
#     if len(children) != 0:
#         dx = width / len(children)
#         nextx = xcenter - width/2 - dx/2
#         for child in children:
#             nextx += dx
#             pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
#                                 vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root)
#     return pos

# # ---------- اجرای انیمیشن ----------
# G = nx.Graph()
# max_steps = len(steps)
# step_counter = 0  # برای جلو رفتن مرحله‌ای

# def update(frame):
#     global step_counter
#     ax.clear()

#     # فقط یک گام جدید اضافه شود
#     if step_counter < max_steps:
#         parent, children = steps[step_counter]
#         for child in children:
#             G.add_edge(parent, child)
#         step_counter += 1  # رفتن به مرحله بعد در گام بعدی

#     pos = hierarchy_pos(G, root=start_node)

#     # رنگ گره‌ها
#     node_colors = []
#     for node in G.nodes():
#         if node == start_node:
#             node_colors.append("green")
#         elif node == goal_node:
#             node_colors.append("blue")
#         elif node in path:
#             node_colors.append("yellow")
#         else:
#             node_colors.append("lightgreen")

#     # رنگ یال‌ها
#     edge_colors = []
#     for edge in G.edges():
#         if edge[0] in path and edge[1] in path:
#             idx0 = path.index(edge[0])
#             idx1 = path.index(edge[1])
#             if abs(idx0 - idx1) == 1:
#                 edge_colors.append("red")
#             else:
#                 edge_colors.append("gray")
#         else:
#             edge_colors.append("gray")

#     nx.draw(
#         G, pos,
#         with_labels=True,
#         node_color=node_colors,
#         edge_color=edge_colors,
#         node_size=1000,
#         font_size=12,
#         font_weight="bold",
#         ax=ax
#     )

#     ax.set_title(f"Step {step_counter}/{max_steps}", color="white", fontsize=14)
#     fig.patch.set_facecolor("#222")
#     ax.set_facecolor("#333")

# # ---------- تعداد frame کمتر، مرحله‌ای ----------
# ani = animation.FuncAnimation(fig, update, frames=max_steps + 10, interval=1000, repeat=False)
# plt.show()

# ----------------------------------------------------------------------------------------------------------
 
# import networkx as nx
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import os

# # فایل‌های خروجی از main
# base_dir = os.path.dirname(os.path.abspath(__file__))
# steps_log_path = os.path.join(base_dir, "steps_log.txt")
# path_file = os.path.join(base_dir, "path_log.txt")

# # خواندن مسیر اصلی
# with open(path_file, "r", encoding="utf-8") as f:
#     solution_path = f.read().strip().split(" <--- ")
# solution_path = [node.strip() for node in solution_path if node.strip()]

# # ساخت گراف از فایل steps_log
# G = nx.DiGraph()
# with open(steps_log_path, "r", encoding="utf-8") as f:
#     for line in f:
#         if '->' in line:
#             parent, children = line.strip().split("->")
#             parent = parent.strip()
#             children = children.strip().replace("[","").replace("]","").replace("'","").split(",")
#             for child in children:
#                 child = child.strip()
#                 if child:
#                     G.add_edge(parent, child)

# # استفاده از graphviz_layout برای مرتب‌سازی سلسله‌مراتبی
# try:
#     pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
# except:
#     # fallback اگر pygraphviz نصب نباشد
#     pos = nx.spring_layout(G)

# fig, ax = plt.subplots(figsize=(12,8))
# nodes = list(G.nodes)
# node_colors = ["lightblue" if n not in solution_path else "red" for n in nodes]

# nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1000, arrowsize=20, ax=ax)

# # تابع انیمیشن مسیر
# path_edges = [(solution_path[i+1], solution_path[i]) for i in range(len(solution_path)-1)]
# edge_colors = ["red" if edge in path_edges else "black" for edge in G.edges]

# def update(num):
#     ax.clear()
#     current_edges = path_edges[:num+1]
#     colors = ["red" if edge in current_edges else "black" for edge in G.edges]
#     nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1000, arrowsize=20, edge_color=colors, ax=ax)
#     ax.set_title(f"Step {num+1}/{len(path_edges)}")

# ani = FuncAnimation(fig, update, frames=len(path_edges), interval=800, repeat=False)
# plt.show()


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
steps_log_path = os.path.join(base_dir, "steps_log.txt")
path_file = os.path.join(base_dir, "path_log.txt")

# خواندن مسیر حل
with open(path_file, "r", encoding="utf-8") as f:
    solution_path = f.read().strip().split(" <--- ")
solution_path = [n.strip() for n in solution_path if n.strip()]

# ساخت درخت از steps_log (درخت واقعی)
G = nx.DiGraph()
with open(steps_log_path, "r", encoding="utf-8") as f:
    for line in f:
        if '->' in line:
            parent, children = line.strip().split("->")
            parent = parent.strip()
            children = children.strip().replace("[","").replace("]","").replace("'","").split(",")
            for child in children:
                child = child.strip()
                if child:
                    G.add_edge(parent, child)

# فقط نودهایی که در مسیر ساخته شده‌اند
nodes_in_tree = list(G.nodes)
node_colors = ["lightblue" if n not in solution_path else "red" for n in nodes_in_tree]

# مرتب‌سازی سلسله‌مراتبی
try:
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
except:
    pos = nx.spring_layout(G)

fig, ax = plt.subplots(figsize=(12,8))

# انیمیشن مسیر حل
path_edges = [(solution_path[i+1], solution_path[i]) for i in range(len(solution_path)-1)]

def update(num):
    ax.clear()
    current_edges = path_edges[:num+1]
    colors = ["red" if edge in current_edges else "black" for edge in G.edges]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1000, arrowsize=20, edge_color=colors, ax=ax)
    ax.set_title(f"Step {num+1}/{len(path_edges)}")

ani = FuncAnimation(fig, update, frames=len(path_edges), interval=800, repeat=False)
plt.show()


import numpy as np
import os
import time as t
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import re

import subprocess
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
label = []
already_expanded = []
# global exp_index 
exp_index = 1
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
class Node :
    def __init__ (self , state=None , action="move" , path_cost=0 , depth=None ,  parent=None ) -> None :
        self.state = state
        self.action = action 
        self.path_cost = path_cost
        self.parent = parent

        if self.parent != None :
        
            self.depth = self.parent.depth + 1
        else :
            self.depth = 0
        

    
    def __str__ (self) -> str : 
        if self.parent != None :
            return ( f" state     : {self.state} \n action    : {self.action} \n path_cost : {self.path_cost} \n depth     : {self.depth} \n\n parent    : {self.parent.state} \n ------------------\n")
        else :
            return ( f" state     : {self.state} \n action    : {self.action} \n path_cost : {self.path_cost} \n depth     : {self.depth} \n\n parent    : None \n ------------------\n")


# temp00 = Node("A" , None , 0 , 0 , None)
# temp01 = Node("B" , "Left" , 10 , 1 , temp00)
# temp10 = Node("B" , "Right" , 15 , 1 , temp00)
# print(temp00.__str__())
# print(temp01.__str__())
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
def tree_search ( initial_state , goal_state , adjncy_matrix , fringe=None  ) -> Node :
    # temp = [initial_state , fringe] 
    # fringe = np.array(temp)
    # fringe = np.array([initial_state])
    x = 0
    for i in range (0 , len(adjncy_matrix)):
        if label[i] == initial_state:
            x = i
    temp = Node(state=initial_state , action=None)
    fringe = np.array([temp])

    initial_itr = False

    while (True) :
        # if (fringe.size == 0):
        #     print('0')
        # else :
        #     print('1')

        # if (fringe.size == 0 and initial_itr) :
        #     return None
        if (fringe.size == 0 and initial_itr) :
            print("No path found.")
            return None, ""

        else :
            initial_itr = True
            node_in_action , fringe  = select_a_node( fringe , algo_type )
            # already_expanded.append([node_in_action])
            already_expanded.append(node_in_action.state)

            if node_in_action.state == goal_state :
                return solution(node_in_action)
            
            fringe = np.append(fringe, np.array(expand(node_in_action, adjncy_matrix)))
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
def select_a_node( fringe , type="ucs" ) :
    if type == "ucs":
        min_ucs = fringe[0].path_cost
        min_index = 0
        # min_ucs_obj = None
        min_ucs_obj = fringe[0]
        # i = 0
        for i in range(1 , fringe.size):
            if fringe[i].path_cost < min_ucs :
                min_ucs = fringe[i].path_cost
                min_ucs_obj = fringe[i]
                min_index = i

        fringe = np.delete(fringe , min_index)

        return min_ucs_obj , fringe
    
    elif type == "bfs":
        bfs_obj = fringe[0]
        fringe = np.delete(fringe, 0) 
       
        return bfs_obj, fringe
    
    elif type == "dfs":
        return
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
def solution( node ) ->  (Node , str):
    # print("?")
    node_main = node
    temp = ""
    temp_path_cost = 0
    while node is not None :
        # if node == None :
        #     return " Empty "
        # else :
        if temp == "" :
            temp = node.state
        else :
            temp = temp + " <--- " + node.state
        node = node.parent

    return  node_main , temp
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
def node_row_finder( node:Node , matrix ) -> int :
    for i in range (0 , len(matrix)):
        if label[i] == node.state :
            return i
#-------------------------------------------------------------------------------------------------------------------------------
        
#-------------------------------------------------------------------------------------------------------------------------------
def expand( node , matrix ) -> list :
    # i = 0
    # for i in range( 0 , len(matrix[0])):
    #     if node.state = 
    # print("?")

    row_x = node_row_finder(node , matrix)

    temp = []
    # for i in range( 0 , len(matrix)):
    #     if matrix[row_x][i] > 0 :
    #         # if node.parent != None :
    #         #     if label[i] != node.parent.state :
    #             if label[i] not in already_expanded :
    #                 temp.append(Node( state=label[i] , path_cost=float(node.path_cost+matrix[row_x][i]) , parent=node ))

    for i in range( 0 , len(matrix)):
        if matrix[row_x][i] > 0 :
            temp.append(Node( state=label[i] , path_cost=float(node.path_cost+matrix[row_x][i]) , parent=node ))

    global exp_index

    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, "expantion_list.txt")

    with open(log_path, "a") as f:
        f.write(f"expand count : {exp_index} \nexpanded node: {node.__str__()}\n")
    exp_index += 1

    ## file for UI
    # steps_log_path = os.path.join(base_dir, "steps_log.txt")
    # with open(steps_log_path, "a", encoding="utf-8") as log:
    #     children_states = [child.state for child in temp]
    #     log.write(f"{node.state} -> {children_states}\n")


    return temp
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
def label_maker ( matrix ) -> None :
    lbl = 65
    for i in range(0 , len(matrix)):
        label.append(chr(lbl))
        lbl = lbl+1
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
def matrix_extractor() :
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "matrix_holder.txt")

    return np.loadtxt(file_path, delimiter=",")
#-------------------------------------------------------------------------------------------------------------------------------





# # Ignore emoji/font warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# base_dir = os.path.dirname(os.path.abspath(__file__))
# steps_log_path = os.path.join(base_dir, "steps_log.txt")
# path_log_path = os.path.join(base_dir, "path_log.txt")

# # --- Helper: parse the logged steps ---
# def parse_steps(file_path):
#     steps = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             if "->" in line:
#                 parent, children_str = line.strip().split("->")
#                 parent = parent.strip()
#                 try:
#                     children = eval(children_str.strip())
#                 except Exception:
#                     children = []
#                 steps.append((parent, children))
#     return steps

# # --- Hierarchical layout (tree-like) ---
# def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.25, vert_loc=0, xcenter=0.5):
#     """
#     Recursively positions nodes in a hierarchy for NetworkX graphs.
#     """
#     if not nx.is_tree(G):
#         raise TypeError("Cannot use hierarchy_pos on a graph that is not a tree")

#     if root is None:
#         if isinstance(G, nx.DiGraph):
#             root = next(iter(nx.topological_sort(G)))
#         else:
#             root = list(G.nodes)[0]

#     def _hierarchy_pos(G, root, left, right, vert_loc, vert_gap, pos=None, parent=None):
#         if pos is None:
#             pos = {root: (xcenter, vert_loc)}
#         else:
#             pos[root] = ((left + right) / 2, vert_loc)
#         neighbors = list(G.successors(root))
#         if neighbors:
#             dx = (right - left) / len(neighbors)
#             nextx = left
#             for neighbor in neighbors:
#                 nextx += dx
#                 pos = _hierarchy_pos(G, neighbor, nextx - dx, nextx, vert_loc - vert_gap, vert_gap, pos, root)
#         return pos

#     return _hierarchy_pos(G, root, 0, width, vert_loc, vert_gap)

# # --- Main animation ---
# def animate_tree():
#     steps = parse_steps(steps_log_path)
#     if not steps:
#         print("No steps found in log file.")
#         return

#     # Read the path (goal sequence)
#     path = ""
#     if os.path.exists(path_log_path):
#         with open(path_log_path, "r", encoding="utf-8") as f:
#             path = f.read().strip()

#     goal_nodes = [x.strip() for x in path.split("<---") if x.strip()]

#     G = nx.DiGraph()
#     fig = plt.figure(figsize=(12, 6))
#     gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])

#     ax_tree = fig.add_subplot(gs[0, 0])
#     right_gs = gs[0, 1].subgridspec(2, 1)
#     ax_current = fig.add_subplot(right_gs[0])
#     ax_info = fig.add_subplot(right_gs[1])

#     plt.tight_layout(pad=3)

#     start_time = t.time()
#     expanded_count = 0

#     def update(frame):
#         nonlocal expanded_count
#         ax_tree.clear()
#         ax_current.clear()
#         ax_info.clear()

#         parent, children = steps[frame]
#         expanded_count += 1
#         for child in children:
#             G.add_edge(parent, child)

#         # Hierarchical layout
#         try:
#             pos = hierarchy_pos(G, list(G.nodes)[0])
#         except Exception:
#             pos = nx.spring_layout(G)

#         # Color goal node differently if found
#         node_colors = []
#         for n in G.nodes():
#             if n in goal_nodes:
#                 node_colors.append("lightgreen")
#             elif n == parent:
#                 node_colors.append("orange")
#             else:
#                 node_colors.append("skyblue")

#         nx.draw(G, pos, with_labels=True, arrows=True, node_color=node_colors, ax=ax_tree)
#         ax_tree.set_title("Tree Search Visualization")

#         ax_current.text(0.5, 0.5, f"Currently Expanding:\n{parent}",
#                         ha="center", va="center", fontsize=15, fontweight="bold")
#         ax_current.axis("off")

#         elapsed = t.time() - start_time
#         goal_reached = parent in goal_nodes
#         status = "✅ Goal Found!" if goal_reached else "⏳ Searching..."
#         ax_info.text(0.5, 0.5,
#                      f"Expanded Nodes: {expanded_count}\nTime: {elapsed:.2f}s\nStatus: {status}",
#                      ha="center", va="center", fontsize=13)
#         ax_info.axis("off")

#         # Stop animation once goal reached
#         if goal_reached:
#             ani.event_source.stop()

#     ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=1200, repeat=False)
#     plt.show()
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
def main () -> None : 
    # matrix = [ 
    #     [0 , 7 , 0] ,
    #     [7 , 0 , 1] ,
    #     [0 , 1 , 0] ]
    # matrix = [
    # [0, 2, 0, 0, 0, 0, 0],  
    # [2, 0, 3, 4, 0, 0, 0],  
    # [0, 3, 0, 0, 6, 0, 0],  
    # [0, 4, 0, 0, 5, 3, 0],  
    # [0, 0, 6, 5, 0, 2, 8],  
    # [0, 0, 0, 3, 2, 0, 1],  
    # [0, 0, 0, 0, 8, 1, 0] ]
    # matrix = [
    # [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],  
    # [2, 0, 3, 8, 0, 0, 0, 0, 0, 0],  
    # [0, 3, 0, 1, 0, 0, 0, 0, 0, 0],  
    # [0, 8, 1, 0, 2, 7, 0, 0, 0, 0],  
    # [0, 0, 0, 2, 0, 2, 0, 0, 0, 0],  
    # [0, 0, 0, 7, 2, 0, 3, 9, 0, 0],  
    # [0, 0, 0, 0, 0, 3, 0, 2, 0, 0],  
    # [0, 0, 0, 0, 0, 9, 2, 0, 4, 0],  
    # [0, 0, 0, 0, 0, 0, 0, 4, 0, 1],  
    # [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] ]
    # matrix = 
    
    matrix = matrix_extractor()

    label_maker(matrix)

    # print( tree_search('A' , 'G' , matrix) )


    initial_state_ = str(input("input the initial_state :")).upper()
    # temp = chr(initial_state_)
    goal_state_ = str(input("input the goal_state :")).upper()

    global algo_type 
    algo_type = str(input("input type of the algorithm you want to use :")).lower()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    files_to_clear = [
        os.path.join(base_dir, "expantion_list.txt"),
        os.path.join(base_dir, "path_log.txt"),
        os.path.join(base_dir, "steps_log.txt")
    ]
    
    for file_path in files_to_clear:
        try:
            with open(file_path, "w") as f:
                f.write("")  # Clear the file
            print(f"Cleared: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Could not clear {os.path.basename(file_path)}: {e}")

    initial_time = t.time()
    # node , path = tree_search('A' , 'J' , matrix)
    node , path = tree_search(initial_state_[0] , goal_state_[0] , matrix)
    ending_time = t.time()

    time_cost = ending_time - initial_time

    print(f"Path :\n" ,path)
    print(f"Path cost : {node.path_cost}")
    print(f"Time needed : {time_cost:.6f} s")

    animate_file = os.path.join(base_dir, "animate.py")
    subprocess.run(["python", animate_file], check=True)
    # animate()

main()
#-------------------------------------------------------------------------------------------------------------------------------





import numpy as np
import os
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
class Node :
    def __init__ (self , state=None , action="move" , path_cost=0 , depth=None , x=-1 ,  parent=None ) -> None :
        self.state = state
        self.action = action 
        self.path_cost = path_cost
        self.parent = parent
        self.x =  x
        # self.y =  y

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
    temp = Node(state=initial_state , action=None , x=x)
    fringe = np.array([temp])

    initial_itr = False

    while (True) :
        # if (fringe.size == 0):
        #     print('0')
        # else :
        #     print('1')

        if (fringe.size == 0 and initial_itr) :
            return None
        else :
            initial_itr = True
            node_in_action , fringe  = select_a_node( fringe , "ucs" )
            already_expanded.append([node_in_action])

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
        return
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
            temp = temp + " ---> " + node.state
        node = node.parent

    return  node_main , temp
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
def expand( node , matrix ) -> list :
    # i = 0
    # for i in range( 0 , len(matrix[0])):
    #     if node.state = 
    # print("?")

    temp = []
    for i in range( 0 , len(matrix[node.x])):
        if matrix[node.x][i] > 0 :
            # if node.parent != None :
            #     if label[i] != node.parent.state :
                if label[i] not in already_expanded :
                    temp.append(Node( state=label[i] , path_cost=float(node.path_cost+matrix[node.x][i]) , x=i , parent=node ))

    return temp
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
already_expanded = []
label = []
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
    
    matrix = matrix_extractor()

    label_maker(matrix)

    # print( tree_search('A' , 'G' , matrix) )
    node , path = tree_search('A' , 'J' , matrix)
    print(f"Path :\n" ,path)
    print(f"Path cost : {node.path_cost}")

main()
#-------------------------------------------------------------------------------------------------------------------------------

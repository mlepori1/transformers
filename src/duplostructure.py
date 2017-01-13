# -*- coding: utf-8 -*-
"""
duplostructure.py
Class representing a DUPLO block structure

HISTORY
-------
2017-01-05: Created by Jonathan D. Jones
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.image as mpimg
import os

class DuploStructure:
    """
    [TODO]
    """

    def __init__(self, edge_dict=None):
        """
        Parameters
        ----------
        block_strings : list of str
          [TODO]
        """

        self.blocks = ('red square', 'yellow square', 'green square',
                       'blue square', 'red rect', 'yellow rect', 'green rect',
                       'blue rect')
        self.duplos = {}

        # A root is a node with no parents. A leaf is a node with no children.
        # A node is both a root and a leaf if it isn't connected to any other
        # nodes. A Duplo structure starts out with no blocks connected, so
        # the root set and the leaf set both contain all nodes.
        self.roots = list(self.blocks)
        self.leaves = list(self.blocks)
        
        """
        self.printRoots()
        self.printLeaves()
        self.printNodes()
        """

        if edge_dict is not None:
            self.parseEdges(edge_dict)
    
    
    def printRoots(self):
        print('ROOTS: {}'.format(self.roots))
    
    
    def printLeaves(self):
        print('LEAVES: {}'.format(self.leaves))
    
    
    def printNodes(self):
        print('NODES: {}'.format(self.duplos.keys()))


    def parseEdges(self, edge_dict):
        """
        """

        for key, val in edge_dict.items():
            if key[::-1] in edge_dict:
                # this edge is undirected
                self.connectAdjacent(key, val)
            else:
                # this edge is directed
                self.connectAbove(key, val)


    def connectAdjacent(self, block_idxs, stud_info):
        """
        """

        obj_idx, tgt_idx = block_idxs
        obj_studs, tgt_studs = stud_info

        obj_name = self.blocks[obj_idx]
        if not obj_name in self.duplos:
            obj = Duplo(obj_name, obj_studs[0])
            self.duplos[obj_name] = obj 
        else:
            obj = self.duplos[obj_name]

        tgt_name = self.blocks[tgt_idx]
        if not tgt_name in self.duplos:
            tgt = Duplo(tgt_name, tgt_studs[0])
            self.duplos[tgt_name] = tgt 
        else:
            tgt = self.duplos[tgt_name]

        obj.addSibling(tgt, obj_studs, tgt_studs)
        tgt.addSibling(obj, tgt_studs, obj_studs)
        
        #self.printNodes()


    def connectAbove(self, block_idxs, stud_info):
        """
        """

        obj_idx, tgt_idx = block_idxs
        obj_studs, tgt_studs = stud_info

        obj_name = self.blocks[obj_idx]
        if not obj_name in self.duplos:
            obj = Duplo(obj_name, obj_studs[0])
            self.duplos[obj_name] = obj 
        else:
            obj = self.duplos[obj_name]

        tgt_name = self.blocks[tgt_idx]
        if not tgt_name in self.duplos:
            tgt = Duplo(tgt_name, tgt_studs[0])
            self.duplos[tgt_name] = tgt 
        else:
            tgt = self.duplos[tgt_name]

        obj.addChild(tgt, obj_studs, tgt_studs)
        tgt.addParent(obj, tgt_studs, obj_studs)

        # remove object from leaves if it's present
        # remove target from roots if it's present
        if obj_name in self.leaves: self.leaves.remove(obj_name)
        if tgt_name in self.roots: self.roots.remove(tgt_name)
        
        """
        self.printRoots()
        self.printLeaves()
        self.printNodes()
        """
    
    
    def computeCoords(self):
        """
        """
        
        for root_name in self.roots:
            
            if not root_name in self.duplos:
                continue
            
            root = self.duplos[root_name]
            if not root.visited:
                local_origin = (0, 0, 0)
                global_origin = (0, 0, 0)
                root.setGlobalCoords(local_origin, global_origin)
                root.passGlobalCoords()
        
        self.resetVisited()
    
    
    def draw(self, file_path, state_index):
        """
        """
        
        plot_index = 0
        images = []
        for root_name in self.roots:
            
            if not root_name in self.duplos:
                continue
            
            root = self.duplos[root_name]
            if not root.visited:
                plt.figure()
                plt.axis('equal')
                plt.axis('off')
                root.draw()
                
                fn = 'state{}-component{}.png'.format(state_index, plot_index)
                path = os.path.join(file_path, fn)
                #plt.axis((-10, 10, -10, 10))
                plt.gca().relim()
                plt.gca().autoscale_view(True,True,True)
                plt.tight_layout()
                plt.savefig(path, dpi=10)
                plt.close()
                plot_index += 1
                
                images.append(mpimg.imread(path))
        
        #plt.show()
        if images:
            image_fn = 'state{}.png'.format(state_index)
            image_path = os.path.join(file_path, image_fn)
            #import pdb; pdb.set_trace()
            image = np.hstack(tuple(images))
            mpimg.imsave(image_path, image)
        else:
            print('WARNING: no figure for state {}'.format(state_index))
        
        self.resetVisited()
    
    
    def resetVisited(self):
        for duplo in self.duplos.values():
            duplo.visited = False


class Duplo:
    """
    [TODO]
    """

    def __init__(self, block_str, shape):
       
        # Basic block properties
        self.name = block_str
        self.color = self.name.split()[0]
        self.shape = shape
        if self.shape == 'S':
            self.size_x = 2
            self.size_y = 2
        elif self.shape == 'H':
            self.size_x = 4
            self.size_y = 2
        elif self.shape == 'V':
            self.size_x = 2
            self.size_y = 4

        # These dictionaries map parent/child/sibling nodes to the messages
        # that will be passed
        self.parents = {}
        self.children = {}  # vertical adjacency relationships
        self.siblings = {}  # horizontal adjacency relationships

        # self.coords maps local stud coordinates to their coordinates in the
        # global frame. When a node is initialized its position in the global
        # frame is unknown.
        self.coords = {}
        
        self.visited = False
    
    
    def setGlobalCoords(self, local_coord, global_coord):
        """
        Set a block's position in the global frame by specifying the location
        of the origin in its local frame.
        
        Parameters
        ----------
        local_coord : np vector
          a stud's location in the block's local frame
        global_coord : np vector
          Global frame location of the stud at local_coord
        """
        
        
        self.coords[local_coord] = global_coord
        self.completeCoords()
    
    
    def completeCoords(self):
        """
        Fill in the global coordinates dictionary when it has at least one
        entry.
        """
        
        if not self.coords:
            print('coords is empty!')
            return
        
        #print('COORDS: {}'.format(list(self.coords.items())))
        
        # compute the translation vector by comparing local/global coordinates
        # for one stud
        local_coord = list(self.coords.keys())[0]
        global_coord = self.coords[local_coord]
        local_coord = np.array(local_coord, dtype=int)
        global_coord = np.array(global_coord, dtype=int)
        translation = global_coord - local_coord
        
        """
        print('LOCAL: {}'.format(local_coord))
        print('GLOBAL: {}'.format(global_coord))
        print('TRANSLATION: {}'.format(translation))
        """
        
        # Update the rest of the studs
        for x in range(self.size_x):
            for y in range(self.size_y):
                local = (x, y, 0)
                if not local in self.coords:
                    local_coord = np.array(local, dtype=int)
                    global_coord = local_coord + translation
                    self.coords[local] = tuple(global_coord.tolist())
            

    def addSibling(self, sibling, self_studs, sibling_studs):
        if not sibling in self.siblings:
            self.siblings[sibling] = (self_studs, sibling_studs)


    def addChild(self, child, self_studs, child_studs):
        if not child in self.children:
            self.children[child] = (self_studs, child_studs)


    def addParent(self, parent, self_studs, parent_studs):
        if not parent in self.parents:
            self.parents[parent] = (self_studs, parent_studs)
    
    
    def passGlobalCoords(self):
        """
        Pass global coordinate locations to neighboring blocks.
        """
        
        """
        print('{} -- X: {} -- Y: {}'.format(self.name, self.size_x, self.size_y))
        print(list(self.coords.keys()))
        print(list(self.coords.values()))
        """
        
        # update global coords from message
        self.visited = True
        
        for child, (self_studs, child_studs) in self.children.items():
            if not child.visited:
                
                """
                print('SELF STUDS: {}'.format(self_studs))
                print('CHILD STUDS: {}'.format(child_studs))
                """
                
                self_coords = self.parseStudString(self_studs)
                child_coords = self.parseStudString(child_studs)
                
                child_global_coords = list(self.coords[self_coords[0]])
                child_global_coords[2] += 1
                child_global_coords = tuple(child_global_coords)
                
                child.setGlobalCoords(child_coords[0], child_global_coords)
                child.passGlobalCoords()
        
        for parent, (self_studs, parent_studs) in self.parents.items():
            if not parent.visited:
                
                """
                print('SELF STUDS: {}'.format(self_studs))
                print('PARENT STUDS: {}'.format(parent_studs))
                """
                
                self_coords = self.parseStudString(self_studs)
                parent_coords = self.parseStudString(parent_studs)
                
                parent_global_coords = list(self.coords[self_coords[0]])
                parent_global_coords[2] -= 1
                parent_global_coords = tuple(parent_global_coords)
                
                parent.setGlobalCoords(parent_coords[0], parent_global_coords)
                parent.passGlobalCoords()
        
        for sibling, (self_studs, sibling_studs) in self.siblings.items():
            if not sibling.visited:
                
                """
                print('SELF STUDS: {}'.format(self_studs))
                print('SIBLING STUDS: {}'.format(sibling_studs))
                """
                
                self_coords = self.parseStudString(self_studs)
                sibling_coords = self.parseStudString(sibling_studs)
                
                if len(self_coords) == 1 and len(sibling_coords) == 1:
                    print('WARNING: skipping one-stud horizontal adjacency connection')
                    continue
                
                # determine whether blocks are adjacent in x or y axis
                parallel_x = True
                parallel_y = True
                for coord in self_coords:
                    parallel_x = parallel_x and coord[0] == self_coords[0][0]
                    parallel_y = parallel_y and coord[1] == self_coords[0][1]
                assert(not (parallel_x and parallel_y))
                
                # determine whether sibling is adjacent in positive or negative
                # direction
                index = None
                offset = None
                if parallel_x:  # parallel in x --> adjacent in y
                    index = 0
                    # TODO: assert sibling parallel in y
                    x_sibling = sibling_coords[0][0]
                    x_self = self_coords[0][0]
                    if x_sibling > x_self:
                        # sibling is to the left of this block
                        offset = -1
                    elif x_sibling < x_self:
                        # sibling is to the right of this block
                        offset = 1
                    else:
                        print('WARNING: bad connection')
                elif parallel_y:    # parallel in y --> adjacent in x
                    index = 1
                    # TODO: assert sibling parallel in y
                    y_sibling = sibling_coords[0][1]
                    y_self = self_coords[0][1]
                    if y_sibling > y_self:
                        # sibling is above this block
                        offset = -1
                    elif y_sibling < y_self:
                        # sibling is below this block
                        offset = 1
                    else:
                        print('WARNING: bad connection')
                else:
                    print('WARNING: neither direction parallel')
                
                sibling_global_coords = list(self.coords[self_coords[0]])
                sibling_global_coords[index] += offset
                sibling_global_coords = tuple(sibling_global_coords)
                
                """
                print('SELF COORDS: {} --> {}'.format(self_coords[0], self.coords[self_coords[0]]))
                print('SIBLING COORDS: {} --> {}'.format(sibling_coords[0], sibling_global_coords))
                """
                
                sibling.setGlobalCoords(sibling_coords[0], sibling_global_coords)
                sibling.passGlobalCoords()            
    
    
    def parseStudString(self, stud_string):
        """
        [TODO]
        
        Parameters
        ----------
        stud_string : string
          format [H/V/S] XX (:XX)*
        
        Returns
        --------
        coords : [TODO]
          [TODO]
        """
        
        coord_strings = stud_string[1:].split(':')
        coords = [tuple(map(int, s))[::-1] + (0,) for s in coord_strings]
        
        return coords
    
    
    def draw(self):
        """
        """
        
        self.visited = True
        
        coords = np.array(tuple(self.coords.values()))
        #zorder = 8 - coords[0,2]    # FIXME
        
        ll_l = (0, self.size_y - 1, 0)
        x_g, y_g, z_g = self.coords[ll_l]
        ll_g = (x_g - 0.5, -y_g - 0.5)
        z_order = 8 - z_g   # FIXME
        
        """
        lower_left_local = (0, self.size_y - 1, 0)
        #lower_left_local = (0, 0, 0)
        lower_left_global = list(self.coords[lower_left_local])
        lower_left_global[2] += 1
        #print(self.name)
        #print(coords)
        #print('llg: {}  size: ({},{})'.format(lower_left_global[0:2], self.size_x, self.size_y))
        #ll_corner = [float(x) for x in lower_left_global[0:2]]
        #ll_corner[1] = - ll_corner[1]
        ll_corner = tuple(lower_left_global)
        """
        
        rect = patches.Rectangle(ll_g, float(self.size_x), float(self.size_y),
                                 zorder=z_order, fc=self.color, linewidth=1)
        
        
        plt.scatter(coords[:,0], -coords[:,1], zorder=z_order+1, c=self.color) #,
                    #edgecolors='face')
        plt.gca().add_patch(rect)
        
        for child in self.children.keys():
           if not child.visited:
               child.draw()
        
        for parent in self.parents.keys():
            if not parent.visited:
                parent.draw()
        
        for sibling in self.siblings.keys():
            if not sibling.visited:
                sibling.draw()


# run some tests if this file is called as a script
if __name__ == '__main__':
    i = 0   # do nothing for now
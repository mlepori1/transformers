# -*- coding: utf-8 -*-
"""
markov_stats.py
Construct block configuration states and calculate statistics on them.

HISTORY
-------
10-24-2016: Created by Jonathan D. Jones
"""

from duplocorpus import DuploCorpus

import numpy as np


actions = ('place above', 'place adjacent', 'disconnect', 'remove block',
           'rotate 90 clockwise', 'rotate 90 counterclockwise', 'rotate 180')
blocks = ('red square', 'yellow square', 'green square', 'blue square',
          'red rect', 'yellow rect', 'green rect', 'blue rect')


def parseLabels(labels):
    """
    Construct a sequence of graph configuration states by parsing action
    annotations
    
    Parameters
    ----------
    labels : numpy structured array
      [TODO]
    
    Returns
    -------
    states : [TODO]
      [TODO]
    """
    
    graphs = [{}]
    
    for l in labels: #[0:1]:
        
        graph = graphs[-1].copy()
        
        action_idx = l['action']
        object_idx = l['object']
        target_idx = l['target']
        
        obj_stud_str = l['obj_studs']
        tgt_stud_str = l['tgt_studs']
        
        action = actions[action_idx]
        object_block = blocks[object_idx]
        target_block = blocks[target_idx]
        
        print('{} | {} | {}'.format(action, object_block, target_block))
        
        if action == 'place above':
            # Add a directed edge between object and target
            key = (object_idx, target_idx)
            value = (obj_stud_str, tgt_stud_str)
            graph[key] = value
        elif action == 'place adjacent':
            # Add an undirected edge between object and target
            key = (object_idx, target_idx)
            value = (obj_stud_str, tgt_stud_str)
            graph[key] = value
            key = (target_idx, object_idx)
            value = (tgt_stud_str, obj_stud_str)
            graph[key] = value
        elif action == 'disconnect':
            # Delete the specific object-target connection
            key = (object_idx, target_idx)
            del graph[key]
        elif action == 'remove block':
            # Delete all connections involving the object block
            for key in graph.keys():
                obj_idx, tgt_idx = key
                if obj_idx == object_idx or tgt_idx == object_idx:
                    del graph[key]
        elif action == 'rotate 90 clockwise':
            # Rotate structure 90 degrees clockwise about its center
            for key, value in graph.items():
                graph[key] = tuple(rotate90cw(string) for string in value)
        elif action == 'rotate 90 counterclockwise':
            # Rotate structure 90 degrees counterclockwise about its center
            for key, value in graph.items():
                graph[key] = tuple(rotate90ccw(string) for string in value)
        elif action == 'rotate 180':
            # Rotate structure 180 degrees clockwise about its center
            for key, value in graph.items():
                graph[key] = tuple(rotate90cw(string) for string in value)
            for key, value in graph.items():
                graph[key] = tuple(rotate90cw(string) for string in value)
        
        graphs.append(graph)
    
    return graphs


def parseCoordString(coord_str):
    """
    Convert a string representing a list of coordinates to an array whose rows
    represent each coordinate in the list.
    """
        
    coord_strs = coord_str[1:].split(':')
    coords = np.zeros((len(coord_strs), 2), dtype=int)
    for i, coord_str in enumerate(coord_strs):
        coords[i,:] = np.array([int(x) for x in coord_str])
    
    return coords


def equivalent(graph1, graph2):
    """
    Return True if graph 1 is equivalent to graph 2 (modulo 90-degree global
    rotation); False if not.
    """
    
    # Graphs definitely can't match if they aren't the same size
    if not len(graph1) == len(graph2):
        return False
    
    # Enforce rotational invariance by checking all 4 possible 90-degree rotations
    rotated_graph = {key: value for key, value in graph1.items()}
    for i in range(4):
        # Check all keys. If they're all equal, the graphs match.
        for key in rotated_graph.keys():
            if not key in graph2:
                break
            if rotated_graph[key] != graph2[key]:
                break
        else:
            return True
        
        # If keys were not all equal, check the next 90-degree rotation
        rotated_graph = {}
        for key, value in rotated_graph.items():
            rotated_graph[key] = tuple(rotate90cw(string) for string in value)
    
    # If none of the rotations match, the graphs are different
    return False


def rotate90cw(stud_str):
    """
    Return the stud string representing a 90 degree clockwise rotation of the
    input stud string.
    """
    
    if stud_str[0] == 'H':
        height = 2
        prefix = 'V'
    elif stud_str[0] == 'V':
        height = 4
        prefix = 'H'
    elif stud_str[0] == 'S':
        height = 2
        prefix = 'S'
    coords = parseCoordString(stud_str)
    rotated_coords = np.column_stack(((coords[:,1], np.abs(coords[:,0] - (height - 1)))))
    rotated_stud_str = ':'.join(['{}{}'.format(*c.tolist()) for c in rotated_coords])
    
    return prefix + rotated_stud_str


def rotate90ccw(stud_str):
    """
    Return the stud string representing a 90 degree counter-clockwise rotation
    of the input stud string.
    """
    
    if stud_str[0] == 'H':
        width = 4
        prefix = 'V'
    elif stud_str[0] == 'V':
        width = 2
        prefix = 'H'
    elif stud_str[0] == 'S':
        width = 2
        prefix = 'S'
    
    coords = parseCoordString(stud_str)
    rotated_coords = np.column_stack(((np.abs(coords[:,1] - (width - 1)), coords[:,0])))
    rotated_stud_str = ':'.join(['{}{}'.format(*c.tolist()) for c in rotated_coords])
    
    return prefix + rotated_stud_str


if __name__ == '__main__':
    
    TASK = 1
    
    c = DuploCorpus()
    
    state_ids = [{}]
    
    ids = zip(c.meta_data['trial id'], c.meta_data['participant id'], c.meta_data['task id'])
    trials = [t for t, p, task in ids if p.startswith('A') and task == TASK]
    
    end = 5
    for t in trials[0:end]:
        labels = c.readLabels(t)
        print(labels)
        graph_seq = parseLabels(labels)
        
        # Get the state ID for each of the parsed configurations
        graph_ids = []
        for graph in graph_seq:
            if not graph in state_ids:
                state_ids.append(graph)
            graph_ids.append(state_ids.index(graph))
        
        # Print configuration sequence info to console
        for i, graph in enumerate(graph_seq):
            print('=====[ {} : {} ]====='.format(i, graph_ids[i]))
            #for key, value in graph.items():
            #    fmtstr = '{}  {}  {}'
            #    print(fmtstr.format(key, *value))
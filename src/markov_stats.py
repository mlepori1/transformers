# -*- coding: utf-8 -*-
"""
markov_stats.py
Construct block configuration states and calculate statistics from them.

HISTORY
-------
2016-10-24: Created by Jonathan D. Jones
"""

from duplocorpus import DuploCorpus
from duplostructure import DuploStructure
import numpy as np
import graphviz as gv
import os


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
    
    graphs = []
    graph = {}
    
    prev_start = -1
    prev_end = -1
    prev_action = ''
    
    labels.sort(order=['end', 'start'])
    for l in labels:
        
        start_idx = l['start']
        end_idx = l['end']
        
        # FIXME
        if start_idx != prev_start and not prev_action.startswith('rotate'):
            print('-----')
            graphs.append(graph)
            graph = graphs[-1].copy()
        
        action_idx = l['action']
        object_idx = l['object']
        target_idx = l['target']
        
        obj_stud_str = l['obj_studs'].astype('str')
        tgt_stud_str = l['tgt_studs'].astype('str')
        
        action = actions[action_idx]
        object_block = '' if object_idx == -1 else blocks[object_idx]
        target_block = '' if target_idx == -1 else blocks[target_idx]
        
        print('{}:{} | {} | {}, {} | {}, {}'.format(start_idx, end_idx, action,
              object_block, obj_stud_str, target_block, tgt_stud_str))
        
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
            #print('=====')
            #print('{} | {}'.format(object_idx, target_idx))
            key = (object_idx, target_idx)
            #print(graph)
            #print('-----')
            del graph[key]
            reversed_key = key[::-1]
            if reversed_key in graph:
                del graph[reversed_key]
            #print(graph)
            #print('=====')
        elif action == 'remove block':
            # Delete all connections involving the object block
            for key in list(graph.keys()):
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
        
        prev_start = start_idx
        prev_end = end_idx
        prev_action = action
    
    print('-----\n')
    if not prev_action.startswith('rotate'):
        graphs.append(graph)
    
    return graphs


def parseCoordString(coord_str):
    """
    Convert a string representing a list of coordinates to an array whose rows
    represent each coordinate in the list.
    
    Parameters
    ----------
    coord_str : [TODO]
      [TODO]
    
    Returns
    -------
    coords : [TODO]
      [TODO]
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
    
    Parameters
    ----------
    graph1 : [TODO]
      [TODO]
    graph2 : [TODO]
      [TODO]
    
    Returns
    -------
    is_equivalent : bool
      [TODO]
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
            if not keysEquivalent(rotated_graph[key], graph2[key]):
                break
        else:
            return True
        
        # If keys were not all equal, check the next 90-degree rotation
        for key, value in rotated_graph.items():
            rotated_graph[key] = tuple(rotate90cw(string) for string in value)
    
    # If none of the rotations match, the graphs are different
    return False


def keysEquivalent(key1, key2):
    """
    Return True is key 1 is equivalent to key 2 (ignoring order); False if not.
    
    Parameters
    ----------
    key1 : [TODO]
      [TODO]
    key2 : [TODO]
      [TODO]
    
    Returns
    -------
    keys_equivalent : bool
      [TODO]
    """
    
    obj_stud_str_1, tgt_stud_str_1 = key1
    obj_stud_str_2, tgt_stud_str_2 = key2
    
    obj_studs_1 = obj_stud_str_1[1:].split(':')
    tgt_studs_1 = tgt_stud_str_1[1:].split(':')
    obj_studs_2 = obj_stud_str_2[1:].split(':')
    tgt_studs_2 = tgt_stud_str_2[1:].split(':')
    
    for s in obj_studs_1:
        if not s in obj_studs_2:
            return False
    
    for s in tgt_studs_1:
        if not s in tgt_studs_2:
            return False
    
    return True


def rotate90cw(stud_str):
    """
    Return the stud string representing a 90 degree clockwise rotation of the
    input stud string.
    
    Parameters
    ----------
    stud_str : str
      [TODO]
    
    Returns
    -------
    rotated : str
      [TODO]
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
    
    Parameters
    ----------
    stud_str : str
      [TODO]
    
    Returns
    -------
    rotated : str
      [TODO]
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


def printGraph(graph):
    """
    Print a block configuration graph to the console.
    """
    
    for key, value in graph.items():
        names = tuple(blocks[k] for k in key)
        fmtstr = '{}  {}  {}'
        print(fmtstr.format(names, *value))


def rotateGraph90cw(graph):
    """
    Apply a 90-degree global rotation to the block configuration graph.
    
    
    Parameters
    ----------
    graph : [TODO]
      [TODO]
    
    Returns
    -------
    rotated_graph : [TODO]
      [TODO]
    """
    
    rotated_graph = {}
    for key, value in graph.items():
        rotated_graph[key] = tuple(rotate90cw(string) for string in value)
    
    return rotated_graph


def drawTransitions(transition_probs, corpus, task):
    """
    Draw a graph representing all observed state transitions.
    
    Parameters
    ----------
    transition_probs : [TODO]
      [TODO]
    corpus : [TODO]
      [TODO]
    task : int
      [TODO]
    """
    
    task_str = 'task-{}'.format(task)
    fig_path = os.path.join(corpus.paths['figures'], 'results-prelim', task_str)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    # Create a directed graph representing the block construction and add
    # all blocks as nodes
    transitions = gv.Digraph(name='transitions', format='png',
                             directory=fig_path)
    
    node_indices = []
    for (object_index, target_index), prob in transition_probs.items():
        if node_indices.count(object_index) == 0:
            image_fn = 'state{}.png'.format(object_index)
            image_path = os.path.join(fig_path, image_fn)
            transitions.node(str(object_index), image=image_path)
            node_indices.append(object_index)
        if node_indices.count(target_index) == 0:
            image_fn = 'state{}.png'.format(target_index)
            image_path = os.path.join(fig_path, image_fn)
            transitions.node(str(target_index), image=image_path)
            node_indices.append(target_index)
        transitions.edge(str(object_index), str(target_index),
                         label=str(prob))
    
    # Save state image to file
    fn = 'task-{}-transitions'.format(task)
    transitions.render(filename=fn, directory=fig_path)
    
    return transitions
    

def drawStates(state_ids, corpus, task):
    """
    """
    
    task_str = 'task-{}'.format(task)
    fig_path = os.path.join(corpus.paths['figures'], 'results-prelim', task_str)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    for index, state in enumerate(state_ids):
        struct = DuploStructure(state)
        struct.computeCoords()
        struct.draw(fig_path, index)


def drawPaths(paths, state_ids, corpus, task, trials):
    """
    """
    
    task_str = 'task-{}'.format(task)
    
    for path_index, path in enumerate(paths):
        path_str = 'path-{}'.format(trials[path_index])
        fig_path = os.path.join(corpus.paths['figures'], 'results-prelim', task_str, path_str)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        
        for order_index, state_index in enumerate(path):
            struct = DuploStructure(state_ids[state_index])
            struct.computeCoords()
            struct.draw(fig_path, order_index)


if __name__ == '__main__':
    
    import itertools
    selected_idxs = list(itertools.chain(range(150, 197), range(225, 255),
                                         range(259, 264), range(270, 276),
                                         range(280, 286), range(292, 297),
                                         range(304, 310), range(292, 298),
                                         range(316, 322), range(328, 334),
                                         range(340, 346), range(352, 356),
                                         range(357, 358), range(364, 370),
                                         range(371, 377), range(383, 389),
                                         range(395, 401), range(407, 413)))
    
    selected_annotator = 'Cathryn_new'
    c = DuploCorpus()
    for selected_task in range(1,7):

        state_ids = [{}]
        paths = []
        transition_counts = {}
        
        ids = zip(c.meta_data['trial id'], c.meta_data['task id'])
        trials = [trial for trial, task in ids if trial in selected_idxs and task == selected_task]
        #trials = [trial for trial, task in ids if trial == 316 and task == selected_task]
        
        for t in trials:
            print('==[ trial {} ]========'.format(t))
    
            labels = c.readLabels(t, selected_annotator)
            #if 2 in labels['action']:
            #    print('affected by bug: skipping trial')
            #    paths.append([])
            #    continue
            
            graph_seq = parseLabels(labels)
                        
            # Get the state ID for each of the parsed configurations
            graph_ids = []
            for graph in graph_seq:
                for i, state in enumerate(state_ids):
                    if equivalent(graph, state):
                        graph_ids.append(i)
                        break
                else:
                    state_ids.append(graph)
                    graph_ids.append(len(state_ids) - 1)
            
            # Print configuration sequence info to console
            for i, graph in enumerate(graph_seq[1:]):
                key = (graph_ids[i], graph_ids[i+1])
                if not key in transition_counts:
                    transition_counts[key] = 1
                else:
                    transition_counts[key] += 1
                #print('{} : {} -> {}'.format(i, *key))
                #printGraph(graph)
            
            paths.append(graph_ids)
        
        drawStates(state_ids, c, selected_task)
        drawTransitions(transition_counts, c, selected_task)
        drawPaths(paths, state_ids, c, selected_task, trials)
        
        # write states
        states_fn = 'states.txt'
        task_str = 'task-{}'.format(selected_task)
        states_path = os.path.join(c.paths['figures'], 'results-prelim', task_str, states_fn)
        with open(states_path, 'wt') as text_file:
            for i, graph in enumerate(state_ids):
                text_file.write('{}\n'.format(i))
                for key, value in graph.items():
                    names = tuple(blocks[k] for k in key)
                    fmtstr = '{}  {}  {}\n'
                    text_file.write(fmtstr.format(names, *value))

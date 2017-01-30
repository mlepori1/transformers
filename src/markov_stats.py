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
import matplotlib.pyplot as plt
import graphviz as gv
import os
from shutil import copyfile


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
        
        # Group all contiguous events with the same start or end time as a
        # single action
        if start_idx != prev_start and end_idx != prev_end and \
           not prev_action.startswith('rotate'):
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
            key = (object_idx, target_idx)
            del graph[key]
            reversed_key = key[::-1]
            if reversed_key in graph:
                del graph[reversed_key]
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
    
    # FIXME: store attachment type in graph
    
    # Graphs definitely can't match if they aren't the same size
    if not len(graph1) == len(graph2):
        return False
    
    # Enforce rotational invariance by checking all 4 possible 90-degree rotations
    rotated_graph = {key: value for key, value in graph1.items()}
    for i in range(4):
        # If every key in graph 1 is also in graph 2, then 1 is a subgraph of 2
        for key in rotated_graph.keys():
            if not key in graph2:
                break
            if not keysEquivalent(rotated_graph[key], graph2[key]):
                break
        # If every key in graph 2 is also in graph 1, then 2 is a subgraph of 1
        # Since we only get here if 1 is also a subgraph of 2, then graphs 1
        # and 2 must be equivalent.
        for key in graph2.keys():
            if not key in rotated_graph:
                break
            if not keysEquivalent(graph2[key], rotated_graph[key]):
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
    
    # test if 1 is a subset of 2
    for s in obj_studs_1:
        if not s in obj_studs_2:
            return False
    
    # test if 2 is a subset of 1
    for s in obj_studs_2:
        if not s in obj_studs_1:
            return False
    
    # test if 1 is a subset of 2
    for s in tgt_studs_1:
        if not s in tgt_studs_2:
            return False
    
    # test if 2 is a subset of 1
    for s in tgt_studs_2:
        if not s in tgt_studs_1:
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


def drawTransitions(transition_counts, state_counts, fig_path):
    """
    Draw a graph representing all observed state transitions.
    
    Parameters
    ----------
    transition_counts : dict, tuple(int) -> int
      [TODO]
    state_counts : dict, int -> int
      [TODO]
    fig_path : str
      [TODO]
    
    Returns
    -------
    transition_probs : N-by-N np array
      [TODO]
    """
    
    # TODO: draw graph with transition probs labeled
    
    # Create a directed graph representing the block construction and add
    # all blocks as nodes
    counts_graph = gv.Digraph(name='transition-counts', format='png',
                              directory=fig_path)
    probs_graph = gv.Digraph(name='transition-probs', format='png',
                             directory=fig_path)
    
    transition_probs = np.zeros((len(state_counts), len(state_counts)))
    node_indices = []
    for (prev_state_id, state_id), count in transition_counts.items():
        
        prev_state_count = state_counts[prev_state_id]
        prob = float(count) / float(prev_state_count)
        transition_probs[prev_state_id, state_id] = prob
        
        if node_indices.count(prev_state_id) == 0:
            image_fn = 'state{}.png'.format(prev_state_id)
            image_path = os.path.join(fig_path, 'states', 'small', image_fn)
            counts_graph.node(str(prev_state_id), image=image_path)
            probs_graph.node(str(prev_state_id), image=image_path)
            node_indices.append(prev_state_id)
        if node_indices.count(state_id) == 0:
            image_fn = 'state{}.png'.format(state_id)
            image_path = os.path.join(fig_path, 'states', 'small', image_fn)
            counts_graph.node(str(state_id), image=image_path)
            probs_graph.node(str(state_id), image=image_path)
            node_indices.append(state_id)
        
        counts_graph.edge(str(prev_state_id), str(state_id),
                          label=str(count))
        probs_graph.edge(str(prev_state_id), str(state_id),
                         label='{:.2f}'.format(prob))
    
    # Save state image to file
    counts_graph.render(filename='transition-counts', directory=fig_path)
    probs_graph.render(filename='transition-probs', directory=fig_path)
    
    # Remove graphviz config file
    os.remove(os.path.join(fig_path, 'transition-counts'))
    os.remove(os.path.join(fig_path, 'transition-probs'))
    
    return transition_probs
    

def drawStates(state_ids, fig_path):
    """
    """
    
    state_fig_path = os.path.join(fig_path, 'states')
    if not os.path.exists(state_fig_path):
        os.makedirs(state_fig_path)
    
    for index, state in enumerate(state_ids):
        struct = DuploStructure(state)
        struct.computeCoords()
        struct.draw(state_fig_path, index)


def drawPaths(paths, state_types, base_path, trials):
    """
    """
    
    fig_path = os.path.join(base_path, 'paths')
    
    for path_index, path in enumerate(paths):
        path_str = 'path{}'.format(trials[path_index])
        path_graph = gv.Digraph(name=path_str, format='png',
                                directory=fig_path)
        
        prev_order_idx = -1
        for order_index, state_index in enumerate(path):
            image_fn = 'state{}.png'.format(state_index)
            image_path = os.path.join(base_path, 'states', 'small', image_fn)
            
            path_graph.node(str(order_index), image=image_path)
            if prev_order_idx >= 0:
                path_graph.edge(str(prev_order_idx), str(order_index))
            
            prev_order_idx = order_index
        
        path_graph.render(filename=path_str, directory=fig_path)
        os.remove(os.path.join(fig_path, path_str))


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

        state_types = [{}]
        paths = []
        transition_counts = {}
        state_counts = {}
        num_transitions = 0
        num_states = 0
        
        # Create directory for figures
        task_str = 'task-{}'.format(selected_task)
        fig_path = os.path.join(c.paths['figures'], 'results-prelim', task_str)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        
        ids = zip(c.meta_data['trial id'], c.meta_data['task id'])
        trials = [trial for trial, task in ids if trial in selected_idxs and task == selected_task]
        #trials = [trial for trial, task in ids if trial == 254 and task == selected_task]
        
        for t in trials:
            print('==[ trial {} ]========'.format(t))
    
            labels = c.readLabels(t, selected_annotator)
            states = parseLabels(labels)
                        
            # Get the state ID for each of the parsed configurations
            type_ids = []
            for state in states:
                for type_index, state_type in enumerate(state_types):
                    if equivalent(state, state_type):
                        type_ids.append(type_index)
                        break
                else:
                    state_types.append(state)
                    type_ids.append(len(state_types) - 1)
            
            # Count state and edge occurrences
            state_counts[0] = state_counts.get(0, 0) + 1
            prev_state_id = 0
            for i, state in enumerate(states[1:]):
                state_id = type_ids[i+1]
                edge = (prev_state_id, state_id)
                transition_counts[edge] = transition_counts.get(edge, 0) + 1
                state_counts[state_id] = state_counts.get(state_id, 0) + 1
                prev_state_id = state_id
            num_states += len(states)
            num_transitions += len(states) - 1
            
            paths.append(type_ids)
        
        # Draw observed states and transition diagrams
        drawStates(state_types, fig_path)
        drawPaths(paths, state_types, fig_path, trials)
        transition_probs = drawTransitions(transition_counts, state_counts, fig_path)
        
        # Calculate and plot path probabilities
        path_probs = []
        for path in paths:
            prev_state_id = path[0]
            assert(prev_state_id == 0)
            path_prob = 1.0
            for state_id in path[1:]:
                path_prob *= transition_probs[prev_state_id, state_id]
                prev_state_id = state_id
            path_key = tuple(path)
            path_probs.append(path_prob)
        
        num_paths = len(path_probs)
        path_probs_arr = np.array(path_probs)
        path_idxs = np.array(list(range(num_paths)))
        
        plt.figure(figsize=(12,6))
        plt.bar(path_idxs, path_probs_arr, tick_label=np.array(trials),
                align='center', color='skyblue')
        plt.title('Likelihood of assembly paths under model')
        plt.xlabel('Path index (p)')
        plt.ylabel('Pr(p)')
        plt.tight_layout()
        fn = os.path.join(fig_path, 'path-probs.png')
        plt.savefig(fn)
        plt.close()
        
        # Sort states by probability and plot the result
        # Reverse sort_idxs to sort in descending order instead of ascending
        path_sort_idxs = np.argsort(path_probs_arr)[::-1]
        sorted_trials = np.array(trials)[path_sort_idxs]
        plt.figure(figsize=(12,6))
        plt.bar(path_idxs, path_probs_arr[path_sort_idxs],
                tick_label=sorted_trials, align='center', color='skyblue')
        plt.title('Assembly paths ordered by likelihood')
        plt.xlabel('Path index (p)')
        plt.ylabel('Pr(p)')
        plt.tight_layout()
        fn = os.path.join(fig_path, 'path-probs-sorted.png')
        plt.savefig(fn)
        plt.close()
        
        ranked_paths_dir = os.path.join(fig_path, 'ranked-paths')
        if not os.path.exists(ranked_paths_dir):
            os.makedirs(ranked_paths_dir)
        for rank, trial_id in enumerate(sorted_trials):
            src_fn = 'path{}.png'.format(trial_id)
            dst_fn = 'rank{}-path{}.png'.format(rank, trial_id)
            source = os.path.join(fig_path, 'paths', src_fn)
            destination = os.path.join(ranked_paths_dir, dst_fn)
            copyfile(source, destination)
        
        # Calculate and plot state (unigram) probabilities
        num_types = len(state_types)
        state_probs = np.zeros(num_types)
        for state, count in state_counts.items():
            state_prob = float(count) / float(num_states)
            state_probs[state] = state_prob
        idxs = np.array(list(range(num_types)))
        plt.figure(figsize=(12,6))
        plt.bar(idxs, state_probs, tick_label=idxs, align='center', color='skyblue')
        plt.title('Relative frequencies of assembly states')
        plt.xlabel('state index')
        plt.ylabel('frequency')
        plt.tight_layout()
        fn = os.path.join(fig_path, 'state-probs.png')
        plt.savefig(fn)
        plt.close()
        
        # Sort states by probability and plot the result
        # Reverse sort_idxs to sort in descending order instead of ascending
        sort_idxs = np.argsort(state_probs)[::-1]
        plt.figure(figsize=(12,6))
        plt.bar(idxs, state_probs[sort_idxs], tick_label=sort_idxs,
                align='center', color='skyblue')
        plt.title('Assembly states ordered by relative frequency')
        plt.xlabel('state index')
        plt.ylabel('frequency')
        plt.tight_layout()
        fn = os.path.join(fig_path, 'state-probs-sorted.png')
        plt.savefig(fn)
        plt.close()
        
        ranked_states_dir = os.path.join(fig_path, 'ranked-states')
        if not os.path.exists(ranked_states_dir):
            os.makedirs(ranked_states_dir)
        for rank, state_id in enumerate(sort_idxs):
            src_fn = 'state{}.png'.format(state_id)
            dst_fn = 'rank{}-state{}.png'.format(rank, state_id)
            source = os.path.join(fig_path, 'states', src_fn)
            destination = os.path.join(ranked_states_dir, dst_fn)
            copyfile(source, destination)
                
        """
        # write states
        states_fn = 'states.txt'
        task_str = 'task-{}'.format(selected_task)
        states_path = os.path.join(c.paths['figures'], 'results-prelim', task_str, states_fn)
        with open(states_path, 'wt') as text_file:
            for i, graph in enumerate(state_types):
                text_file.write('{}\n'.format(i))
                for key, value in graph.items():
                    names = tuple(blocks[k] for k in key)
                    fmtstr = '{}  {}  {}\n'
                    text_file.write(fmtstr.format(names, *value))
        """

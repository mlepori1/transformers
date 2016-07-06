# -*- coding: utf-8 -*-
"""
annotationgui.py
  User interface for annotating trials from video frames.

AUTHOR
  Jonathan D. Jones
"""

from __future__ import print_function
import Tkinter as tk
from PIL import Image, ImageTk
import graphviz as gv
import os
import numpy

from duplocorpus import DuploCorpus

class Application:
    
    def __init__(self, parent):
        
        self.parent = parent
        self.popup = None
        
        # Constants
        self.actions = ('place above', 'place adjacent', 'remove', 'rotate')
        self.blocks = ('red square', 'yellow square', 'green square',
                       'blue square', 'red rect', 'yellow rect', 'green rect',
                       'blue rect')
        self.block2index = {block: str(i) for i, block in enumerate(self.blocks)}
        
        # Corpus object manages file I/O -- filenames and paths are filled in
        # after reading input from trial selection interface
        self.corpus = DuploCorpus()
        self.trial_id = None
        self.rgb_frame_fns = None
        self.state_fig_path = None
        
        # Initial values
        self.action_start_index = -1
        self.action_end_index = -1
        self.cur_frame = 0
        self.states = []
        self.labels = []
        
        
        # User input (to be read from interface later)
        self.action_field = None
        self.object_field = None
        self.target_fields = {}
        self.trial_field = None
        
        # Start drawing annotation interface
        self.content_frame = tk.Frame(self.parent)
        self.navigation_frame = tk.Frame(self.parent)
        self.defineTrialSelectionInterface()
        self.drawInterface()
        
        #define matrix for object studs
        #may be deprecated - consider removing
        self.obj_studs = numpy.zeros((2,2), dtype=bool)
        
        #create object arrays for square blocks
        self.arrays = {}
        names = ('red square','green square','yellow square','blue square')
        for n in names:
            a = numpy.zeros((2,2), dtype=bool)
            self.arrays[n] = a
        
        #relate names of square blocks and colors
        names = ('red square','green square','yellow square','blue square')
        colors = ('red','green','yellow','blue')
        self.name2color_square = {n:c for n,c in zip(names, colors)}
        
        #create object arrays for rectangular blocks
        self.arrays_rect = {}
        names_rect = ('red rect','green rect','yellow rect','blue rect')
        for n in names_rect:
            a = numpy.zeros((2,4), dtype=bool)
            self.arrays_rect[n] = a
            
        #relate names of rectangular blocks and colors
        names_rect = ('red rect','green rect','yellow rect','blue rect')
        colors = ('red','green','yellow','blue')
        self.name2color_rect = {n:c for n,c in zip(names_rect, colors)}
        
        #create target arrays for square blocks
        self.arrays_target = {}
        names = ('red square','green square','yellow square','blue square')
        for n in names:
            a = numpy.zeros((2,2), dtype=bool)
            self.arrays_target[n] = a       
            
        #create object arrays for rectangular blocks
        self.arrays_rect_target = {}
        names_rect = ('red rect','green rect','yellow rect','blue rect')
        for n in names_rect:
            a = numpy.zeros((2,4), dtype=bool)
            self.arrays_rect_target[n] = a
            
        #define event queue
        self.event_queue = None
            
    def initState(self):
        """
        Initialize the world state as an unconnected graph with one node for
        each block.
        
        Returns:
        --------
        [gv digraph] state: graphviz directed graph representing the initial
          world state.
        """
        
        # Block height when drawn as graph node, in inches
        height = 0.25
        
        # Create a directed graph representing the block construction and add
        # all blocks as nodes
        state = gv.Digraph(name=str(len(self.states)), format='png',
                           directory=self.state_fig_path)
        for i, block in enumerate(self.blocks):
            color, shape = block.split(' ')
            width = height if shape == 'square' else 2 * height
            gv_shape = 'Msquare' if shape == 'square' else 'box'
            state.node(str(i), height=str(height), width=str(width),
                       color=color, shape=gv_shape, style='filled',
                       body='size=4,4')
        
        # Save state image to file
        state.render(filename=str(len(self.states)),
                     directory=self.state_fig_path)
        
        return state


    def drawInterface(self):
        """
        Draw main content and navigation frames onto the parent
        """
        
        self.content_frame.place(relx=0.5, rely=0.5, anchor='center')
        self.navigation_frame.place(relx=0.5, rely=0.9, anchor='center')
    
    
    def clearInterface(self):
        """
        Destroy and re-initialize main content and navigation frames.
        """
        
        self.content_frame.destroy()
        self.navigation_frame.destroy()
        
        self.content_frame = tk.Frame(self.parent)
        self.navigation_frame = tk.Frame(self.parent)
    
    
    def defineTrialSelectionInterface(self):
        """
        Draw an interface that allows the user to choose which trial to
        annotate.
        """
        
        master = self.content_frame
        
        user_text = 'Select a trial to annotate:'
        instructions = tk.Label(master, text=user_text)
        instructions.pack()
        
        # Draw a listbox with a vertical scrollbar
        scrollbar = tk.Scrollbar(master, orient=tk.VERTICAL)
        self.trial_field = tk.Listbox(master, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.trial_field.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.trial_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        # Fill the listbox with trial data
        for entry in self.corpus.meta_data:
            fmtstr = '{}, task {}'
            trial_info = fmtstr.format(entry['participant id'], entry['task id'])
            self.trial_field.insert(tk.END, trial_info)
        
        # Draw a select button
        master = self.navigation_frame
        submit = tk.Button(master, text='Select', command=self.submit)
        submit.pack()
    
    
    def submit(self):
        """
        Determine which trial the user selected for annotation.
        """
        
        # Prompt the user for input if no field is currently selected
        if not self.trial_field.curselection():
            error_string = 'Please make a selection.'
            self.badInputDialog(error_string)
            return
        
        # Read trial ID
        self.trial_id = self.trial_field.curselection()[0]
        
        # Define I/O paths
        self.rgb_frame_fns = self.corpus.getRgbFrameFns(self.trial_id)
        self.state_fig_path = os.path.join(self.corpus.paths['figures'],
                                           'states', str(self.trial_id))
        if not os.path.exists(self.state_fig_path):
            os.makedirs(self.state_fig_path)
        
        # Draw the initial empty state
        self.states.append(self.initState())
        
        # Move on to the annotation interface
        self.clearInterface()
        self.defineAnnotationInterface()
        self.drawInterface()
  
    def assignVar_square(self, x,y,z):
        a = self.arrays[z]
        a[x][y] = not a[x][y]
        b = self.button_dict_square[z]
        
        if a[x][y]:
            b[x][y].config(bg='black')
            b[x][y].config(activebackground='black')
            b[x][y].config(relief='sunken')
        else:
            b[x][y].config(bg=self.name2color_square[z])
            b[x][y].config(activebackground=self.name2color_square[z])
            b[x][y].config(relief='raised')
            
    def assignVar_rect(self, x,y,z):
        a = self.arrays_rect[z]
        a[x][y] = not a[x][y]
        b = self.button_dict_rect[z]
        
        if a[x][y]:
            b[x][y].config(bg='black')
            b[x][y].config(activebackground='black')
            b[x][y].config(relief='sunken')
        else:
            b[x][y].config(bg=self.name2color_rect[z])
            b[x][y].config(activebackground=self.name2color_rect[z])
            b[x][y].config(relief='raised')
    
    def assignVar_square_target(self, x,y,z):
        a = self.arrays_target[z]
        a[x][y] = not a[x][y]
        b = self.button_dict_square_target[z]
        
        if a[x][y]:
            b[x][y].config(bg='black')
            b[x][y].config(activebackground='black')
            b[x][y].config(relief='sunken')
        else:
            b[x][y].config(bg=self.name2color_square[z])
            b[x][y].config(activebackground=self.name2color_square[z])
            b[x][y].config(relief='raised')
            
    def assignVar_rect_target(self, x,y,z):
        a = self.arrays_rect_target[z]
        a[x][y] = not a[x][y]
        b = self.button_dict_rect_target[z]
        
        if a[x][y]:
            b[x][y].config(bg='black')
            b[x][y].config(activebackground='black')
            b[x][y].config(relief='sunken')
        else:
            b[x][y].config(bg=self.name2color_rect[z])
            b[x][y].config(activebackground=self.name2color_rect[z])
            b[x][y].config(relief='raised')
            
    def defineAnnotationInterface(self):
        """
        Draw the annotation interface with RGB frame.
        """
        
        master = self.content_frame
        
        # Navigate forwards and backwards in RGB frames by using j and k keys
        self.parent.bind('k', self.forward)
        self.parent.bind('j', self.back)
        
        # Draw RGB frame
        frame1 = tk.Frame(master)
        
        # RGB video
        rgb_frame_fn = self.rgb_frame_fns[self.cur_frame]
        rgb_image = ImageTk.PhotoImage(Image.open(rgb_frame_fn))
        self.rgb_display = tk.Label(frame1, image=rgb_image)
        self.rgb_display.image = rgb_image
        self.rgb_display.pack(side=tk.LEFT)
        
        # Draw annotation frame
        ann_frame = tk.Frame(frame1)
        
        # Action label (radio buttons)
        action_label = tk.Label(ann_frame, text='Action')
        action_label.grid(row=0, column=0)
        self.action_field = tk.IntVar()
        for i, label in enumerate(self.actions):
            action_button = tk.Radiobutton(ann_frame, text=label,
                                    variable=self.action_field, value=i+1)
            action_button.grid(sticky=tk.W, row=i+1, column=0)
        """
        # Object label (radio buttons)
        obj_label = tk.Label(ann_frame, text='Object')
        obj_label.grid(row=0, column=1)
        self.object_field = tk.IntVar()
        for i, block in enumerate(self.blocks):
            button = tk.Radiobutton(ann_frame, text=block,
                                    variable=self.object_field, value=i+1)
            button.grid(sticky=tk.W, row=i+1, column=1)
        
        # Target label(s) (check boxes)
        target_label = tk.Label(ann_frame, text='Target(s)')
        target_label.grid(row=0, column=2)
        for i, block in enumerate(self.blocks):
            self.target_fields[block] = tk.IntVar()
            target_box = tk.Checkbutton(ann_frame, text=block,
                                        variable=self.target_fields[block])
            target_box.grid(sticky=tk.W, row=i+1, column=2)
        """
        
        #define object buttons for square shapes
        self.button_dict_square = {}
        names = ('red square','green square','yellow square','blue square')        
        for i,n in enumerate(names):
            self.annotate_buttons = []
            f = tk.Frame(ann_frame)
            g = tk.Frame(ann_frame)
            square_colors = ('red','green','yellow','blue')
            for r in range(2):
                self.annotate_buttons.append([])
                for c in range(2):
                    func=lambda x=r, y=c, z=n: self.assignVar_square(x,y,z)
                    b = tk.Button(f, command=func, bg=square_colors[i], 
                                  activebackground=square_colors[i])
                    b.grid(row=r, column=c)
                    a = tk.Label(g, text='\t', font=("TkDefaultFont",1))
                    a.grid(row=r-r+1, column=c-c+1)
                    self.annotate_buttons[-1].append(b)
            self.button_dict_square[n] = self.annotate_buttons
            f.grid(row=2*i+1, column=1)
            g.grid(row=2*i+2, column=1)
        
        #define object buttons for rectangular shapes
        self.button_dict_rect = {}
        names_rect = ('red rect','green rect','yellow rect','blue rect')        
        for i,n in enumerate(names_rect):
            self.annotate_buttons_rect = []
            h = tk.Frame(ann_frame)
            k = tk.Frame(ann_frame)
            l = tk.Frame(ann_frame)
            rect_colors = ('red','green','yellow','blue')
            for r in range(2):
                self.annotate_buttons_rect.append([])
                for c in range(4):
                    func=lambda x=r, y=c, z=n: self.assignVar_rect(x,y,z)
                    br = tk.Button(h, command=func, bg=rect_colors[i], 
                                   activebackground=rect_colors[i])
                    br.grid(row=r, column=c)
                    ar = tk.Label(k, text='\t', font=("TkDefaultFont",1))
                    ar.grid(row=r-r+1, column=c-c+3)
                    cr = tk.Label(l, text='    ', font=("TkDefaultFont",14))
                    cr.grid(row=r-r+1, column=c-c+2)
                    self.annotate_buttons_rect[-1].append(br)
            self.button_dict_rect[n] = self.annotate_buttons_rect
            h.grid(row=2*i+1, column=3)
            k.grid(row=2*i+2, column=3)
            l.grid(row=2*i, column=2)
            
        #define target buttons for square shapes
        self.button_dict_square_target = {}
        names = ('red square','green square','yellow square','blue square')        
        for i,n in enumerate(names):
            self.annotate_buttons_target = []
            ft = tk.Frame(ann_frame)
            gt = tk.Frame(ann_frame)
            square_colors = ('red','green','yellow','blue')
            for r in range(2):
                self.annotate_buttons_target.append([])
                for c in range(2):
                    func=lambda x=r, y=c, z=n: self.assignVar_square_target(x,y,z)
                    bt = tk.Button(ft, command=func, bg=square_colors[i], 
                                  activebackground=square_colors[i])
                    bt.grid(row=r, column=c)
                    at = tk.Label(gt, text='\t', font=("TkDefaultFont",1))
                    at.grid(row=r-r+1, column=c-c+1)
                    self.annotate_buttons_target[-1].append(bt)
            self.button_dict_square_target[n] = self.annotate_buttons_target
            ft.grid(row=2*i+1, column=5)
            gt.grid(row=2*i+2, column=5)
        
        #define target buttons for rectangular shapes
        self.button_dict_rect_target = {}
        names_rect = ('red rect','green rect','yellow rect','blue rect')        
        for i,n in enumerate(names_rect):
            self.annotate_buttons_rect_target = []
            ht = tk.Frame(ann_frame)
            kt = tk.Frame(ann_frame)
            lt = tk.Frame(ann_frame)
            xt = tk.Frame(ann_frame)
            rect_colors = ('red','green','yellow','blue')
            for r in range(2):
                self.annotate_buttons_rect_target.append([])
                for c in range(4):
                    func=lambda x=r, y=c, z=n: self.assignVar_rect_target(x,y,z)
                    brt = tk.Button(ht, command=func, bg=rect_colors[i], 
                                   activebackground=rect_colors[i])
                    brt.grid(row=r, column=c)
                    art = tk.Label(kt, text='\t', font=("TkDefaultFont",1))
                    art.grid(row=r-r+1, column=c-c+3)
                    crt = tk.Label(lt, text='    ', font=("TkDefaultFont",14))
                    crt.grid(row=r-r+1, column=c-c+2)
                    xrt = tk.Label(xt, text='              ', font=("TkDefaultFont",14))
                    xrt.grid(row=r-r+1, column=c-c+4)
                    self.annotate_buttons_rect_target[-1].append(brt)
            self.button_dict_rect_target[n] = self.annotate_buttons_rect_target
            ht.grid(row=2*i+1, column=7)
            kt.grid(row=2*i+2, column=7)
            lt.grid(row=2*i, column=6)
            xt.grid(row=2*1, column=4)
        
        ann_frame.pack(side=tk.RIGHT)
        frame1.pack(side=tk.TOP)
        
        # Draw visualization of annotation
        frame2 = tk.Frame(master)
        state_fn = '{}.png'.format(len(self.states) - 1)
        state_path = os.path.join(self.state_fig_path, state_fn)
        state_image = ImageTk.PhotoImage(Image.open(state_path))
        self.state_display = tk.Label(frame2, image=state_image)
        self.state_display.image = state_image
        self.state_display.pack()
        frame2.pack(side=tk.BOTTOM)
        
        # Draw start/end, undo, restart, quit, and queue event buttons
        master = self.navigation_frame
        self.start_end = tk.Button(master, text='Start of action',
                                   command=self.startAction, default=tk.ACTIVE)
        self.start_end.grid(sticky=tk.E, row=0, column=1)
        undo = tk.Button(master, text='Undo annotation',
                         command=self.undoAction)
        undo.grid(sticky=tk.W, row=0, column=0)
        restart = tk.Button(master, text='Start new trial', command=self.restart)
        restart.grid(sticky=tk.W, row=0, column=3)
        restart = tk.Button(master, text='Quit', command=self.close)
        restart.grid(sticky=tk.W, row=0, column=4)
        self.queue_button = tk.Button(master, text='Queue event', 
                                      command = self.queue_event, state=tk.DISABLED)
        self.queue_button.grid(stick=tk.W, row=0, column=2)
    
    def queue_event(self):
        """
        Adds array values for obj/target block into event queue
        """
        
        ##validate data
        
        #no self edges check
        names = ('red square','green square','yellow square','blue square')
        for n in names:
            if numpy.any(self.arrays[n]) and numpy.any(self.arrays_target[n]):
                queue_error = 'A block cannot be attached to itself'
                self.badInputDialog(queue_error)
                return
        names_rect = ('red rect','green rect','yellow rect','blue rect')
        for n in names_rect:
            if numpy.any(self.arrays_rect[n]) and numpy.any(self.arrays_rect_target[n]):
                queue_error = 'A block cannot be attached to itself'
                self.badInputDialog(queue_error)
                return
                
        #
        #check to make sure only one block per column(obj/target) is selected
        #as well as record which block was selected
        
            #object blocks        
                #concatenate dictionaries for object blocks first
        self.arrays.update(self.arrays_rect)
        names = ('red square','green square','yellow square','blue square', 
        'red rect','green rect','yellow rect','blue rect')
        selected_object = ""
        count = 0
        for n in names:
            val = self.arrays[n].any()
            count += val
            if count >1:
                queue_error_object = 'Only one object block may be selected at a time'
                self.badInputDialog(queue_error_object)
                return
            elif val:
                selected_object = n
        selected_object_block = self.arrays[selected_object]
        
        self.arrays_target.update(self.arrays_rect_target)
        names = ('red square','green square','yellow square','blue square', 
        'red rect','green rect','yellow rect','blue rect')
        selected_target = ""
        count = 0
        for n in names:
            val = self.arrays_target[n].any()
            count += val
            if count >1:
                queue_error_target = 'Only one target block may be selected at a time'
                self.badInputDialog(queue_error_target)
                return
            elif val:
                selected_target = n
        selected_target_block = self.arrays_target[selected_target]
        
        #get action index
        action_index = self.action_field.get() - 1
        
        # Make sure the user has selected an action
        # (Value of -1 means original value was zero, ie empty)
        if action_index == -1:
            err_str = 'Please select and action and an object block.'
            self.badInputDialog(err_str)
            return 
            
        #define event
        event = (action_index, selected_object, selected_target,
                 selected_object_block, selected_target_block)       
        
        #append the event to the event queue
        self.event_queue.append(event)
        
    def undoAction(self):
        """
        Delete the previous action annotation and re-draw block configuration.
        """
        """
        if not self.states:
            error_string = 'No annotation to undo'
            self.badInputDialog(error_string)
        
        
        # Delete all labels associated with the last annotation
        last_label = self.labels[-1]
        while self.labels[-1][:3] == last_label[:3]:
            self.labels.pop(-1)
        
        self.states = self.states[:-1]
        self.updateWorldState()
        """
        
        #delete last action
        self.labels.pop()        
        print(self.labels)
        return
    
    
    def restart(self):
        """
        Save labels and go back to the trial selection interface.
        """
        
        # Save labels if they exist and update metadata file to reflect changes
        if self.labels:
            self.corpus.writeLabels(self.trial_id, self.labels)
            self.corpus.updateMetaData(self.trial_id)
        self.clearInterface()
        self.__init__(self.parent)
    
    
    def close(self):
        """
        Save labels and exit the annotation interface.
        """
        
        # Save labels if they exist and update metadata file to reflect changes
        if self.labels:
            self.corpus.writeLabels(self.trial_id, self.labels)
            self.corpus.updateMetaData(self.trial_id)
        self.parent.destroy()
    
    
    def back(self, event=None):
        """
        Move backward 10 frames in the video unless the beginning of the video
        is fewer than 10 frames away. Then, display the new rgb frame.
        """
        
        self.cur_frame = max(self.cur_frame - 10, 0)
        
        # Redraw rgb frame
        cur_fn = self.rgb_frame_fns[self.cur_frame]
        rgb_image = ImageTk.PhotoImage(Image.open(cur_fn))
        self.rgb_display.configure(image=rgb_image)
        self.rgb_display.image = rgb_image
    
    
    def forward(self, event=None):
        """
        Move forward 10 frames in the video unless the end of the video is
        fewer than 10 frames away. Then, display the new rgb frame.
        """
        
        self.cur_frame = min(self.cur_frame + 10, len(self.rgb_frame_fns) - 1)
        
        # Redraw rgb frame
        cur_fn = self.rgb_frame_fns[self.cur_frame]
        rgb_image = Image.open(cur_fn)
        rgb_image = ImageTk.PhotoImage(Image.open(cur_fn))
        self.rgb_display.configure(image=rgb_image)
        self.rgb_display.image = rgb_image
    
    
    def startAction(self):
        """
        Update start/end button when user indicates the start of an action.
        """
        
        # Set the action's start to the index of the current RGB frame
        self.action_start_index = self.cur_frame
        
        # Redraw button
        self.start_end.configure(text='End of action', command=self.endAction)
        
        # enable queue event button
        self.queue_button.configure(state=tk.NORMAL)
        
        #enable event queue
        self.event_queue = []
    
    def endAction(self):
        """
        When user indicates the end of an action, read and parse the annotation
        for this action. If the annotation is a valid one, display the new
        world state. If not, display a warning message to the user.
        """
        
        # Set the action's end to the index of the current frame
        self.action_end_index = self.cur_frame
              
        #add start and end indexes to event queue
        self.event_queue = [s + (self.action_start_index, self.action_end_index) 
        for s in self.event_queue]                
           
           
        #write event queue (which represents one action) to action queue (called 'labels')
        self.labels.append(self.event_queue)
        
        #reset event queue
        self.event_queue = None
        
        # Reset start and end indices
        self.action_start_index = -1
        self.action_end_index = -1
        
        #Redraw world state image and stard/end button
        self.updateWorldState()
        self.start_end.configure(text='Start of action',
                                 command=self.startAction)
                                 
        #disable queue event button
        self.queue_button.configure(state=tk.DISABLED)
        
    
    def updateWorldState(self):
        """
        Redraw the block configuration graph which represents the world state.
        """
        
        state_fn = '{}.png'.format(len(self.states) - 1)
        state_path = os.path.join(self.state_fig_path, state_fn)
        state_image = ImageTk.PhotoImage(Image.open(state_path))
        self.state_display.configure(image=state_image)
        self.state_display.image = state_image
    
    
    def parseAction(self, action, object_index, target_indices):
        """
        Update the world state by interpreting an action annotation.
        
        Args:
        -----
        [str] action: Action label (one of self.actions)
        [str] object_block: Block that was placed (index in self.blocks)
        [list(str)] target_blocks: Blocks near the one that was placed
        """
        
        # TODO: validate the provided action by making sure it is a possible
        #   world state configuration
                        
        # Initialize a new state graph with the contents of the previous state.
        # I do it this way because the Digraph class doesn't have a copy
        # constructor.
        cur_state = gv.Digraph(name=str(len(self.states)),
                               directory=self.state_fig_path, format='png')
        cur_state.body = list(self.states[-1].body)
        
        # Add a directed edge for each target block
        if action == 'place above':
            for target_index in target_indices:
                cur_state.edge(str(object_index), str(target_index))
        # Add an undirected edge for each target block
        elif action == 'place adjacent':
            for target_index in target_indices:
                cur_state.edge(str(object_index), str(target_index), dir='none')
        # Remove all edges associated with this node
        elif action == 'remove':
            new_body = []
            for entry in cur_state.body:
                # Matches all edges originating from the removed block
                start_pattern = '\t\t{}'.format(object_index)
                # Matches all undirected edges leading to the removed block
                end_pattern = '-> {} [dir=none]'.format(object_index)
                if not (entry.startswith(start_pattern) or
                   entry.endswith(end_pattern)):
                    new_body.append(entry)
            cur_state.body = new_body
        
        # Save this state both to memory and to file
        cur_state.render(filename=str(len(self.states)),
                         directory=self.state_fig_path)
        self.states.append(cur_state)
        
        return ''
    
    
    def badInputDialog(self, error_string):
        """
        Alert user of malformed input
        
        Args:
        -----
        [str] error_string: Message that will be displayed in popup window
        """
        
        # Destroy the popup window if it exists
        if self.popup:
            self.cancel()
        
        self.popup = tk.Toplevel(self.parent)
        
        l = tk.Label(self.popup, text=error_string)
        l.grid(row=0)
        
        ok = tk.Button(self.popup, text='OK', command=self.cancel)
        ok.grid(row=1)
    
    
    def cancel(self):
        """
        Close the current popup window.
        """
        
        # It shouldn't be possible to close a window that doesn't exist
        assert(not self.popup is None)
        
        self.popup.destroy()
        self.popup = None
        

if __name__ == '__main__':
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry('{0}x{1}+0+0'.format(screen_width, screen_height))
    #root.geometry('800x600')
    
    app = Application(root)
    
    root.mainloop()
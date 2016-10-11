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
import numpy as np

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
        self.cur_frame = 0
        self.states = []
        self.labels = []
        
        # User input (to be read from interface later)
        self.action_field = None
        self.object_field = None
        self.target_field = None
        self.trial_field = None
        
        """
        #create object arrays for square blocks
        self.arrays_square_object = {}
        names = ('red square','green square','yellow square','blue square')
        for n in names:
            a = np.zeros((2,2), dtype=bool)
            self.arrays_square_object[n] = a
        """
        
        # These are placeholders for boolean arrays representing if studs are
        # selected or not
        self.object_studs = None
        self.target_studs = None
        
        # Start drawing annotation interface
        self.content_frame = tk.Frame(self.parent)
        self.navigation_frame = tk.Frame(self.parent)
        self.defineTrialSelectionInterface()
        self.drawInterface()
        
    
    def initState(self):
        """
        Initialize the world state as an unconnected graph with one node for
        each block.
        
        Returns:
        --------
        state: graphviz Digraph
          graphviz directed graph representing the initial world state.
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
            button = tk.Radiobutton(ann_frame, text=label,
                                    variable=self.action_field, value=i+1)
            button.grid(sticky=tk.W, row=i+1, column=0)
        
        # Object label (radio buttons)
        obj_label = tk.Label(ann_frame, text='Object')
        obj_label.grid(row=0, column=1)
        self.object_field = tk.IntVar()
        for i, block in enumerate(self.blocks):
            button = tk.Radiobutton(ann_frame, text=block,
                                    variable=self.object_field, value=i+1)
            button.grid(sticky=tk.W, row=i+1, column=1)
        
        # Target label (radio buttons)
        target_label = tk.Label(ann_frame, text='Target')
        target_label.grid(row=0, column=2)
        self.target_field = tk.IntVar()
        for i, block in enumerate(self.blocks):
            button = tk.Radiobutton(ann_frame, text=block,
                                    variable=self.target_field, value=i+1)
            button.grid(sticky=tk.W, row=i+1, column=2)
        
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
        
        # Draw start/end, undo, restart, and quit buttons
        master = self.navigation_frame
        actions_frame = tk.Frame(master)
        self.start_end = tk.Button(actions_frame, text='Start of action',
                                   command=self.startAction, default=tk.ACTIVE)
        self.start_end.grid(sticky=tk.E, row=0, column=0)
        select = tk.Button(actions_frame, text='Add connection', command=self.studSelectionDialog)
        select.grid(sticky=tk.W, row=0, column=1)
        undo = tk.Button(actions_frame, text='Undo connection',
                         command=self.undoAction)
        undo.grid(sticky=tk.W, row=0, column=2)
        actions_frame.pack(side=tk.TOP)
        
        # Draw restart and quit buttons
        end_frame = tk.Frame(master)
        restart = tk.Button(end_frame, text='Start new trial', command=self.restart)
        restart.grid(sticky=tk.W, row=0, column=0)
        close = tk.Button(end_frame, text='Quit', command=self.close)
        close.grid(sticky=tk.W, row=0, column=1)
        end_frame.pack(side=tk.TOP)
        
        
    def studSelectionDialog(self):
        """
        Create a popup window to code block relations.
        """
        # Destroy the popup window if it exists
        if self.popup:
            self.cancel()
        
        self.popup = tk.Toplevel(self.parent)
        self.popup.geometry('{0}x{1}'.format(screen_width/4, screen_height/4))    
        
        # Create object block
        object_index = self.object_field.get() - 1
        object_name = self.blocks[object_index]
        object_color, object_shape = object_name.split(' ')
        object_color_str = object_color + '2'
        self.object_frame = tk.Frame(self.popup, bg=object_color_str)
        if object_shape == 'square':    # Draw a square
            rows, cols = 2, 2
        elif object_shape == 'rect':    # Draw a rectangle
            rows, cols = 2, 4
        self.object_studs = np.zeros((rows, cols), dtype=bool)
        
        self.object_stud_buttons = [] 
        for r in range(rows):
            self.object_stud_buttons.append([])
            for c in range(cols):
                func = lambda b='object', n=object_name, r=r, c=c: self.toggleStud(b,n,r,c)
                b = tk.Button(self.object_frame, command=func, bg=object_color_str, 
                              activebackground=object_color_str, relief='ridge')
                b.grid(row=r, column=c)
                self.object_stud_buttons[-1].append(b)
        
        self.object_frame.place(relx=0.25, rely=0.33, anchor='center')
        rotate_object = lambda x='object', n=object_name, o='vertical': self.rotate(x, n, o)
        self.object_button = tk.Button(self.popup, command=rotate_object, text='rotate')
        self.object_button.place(relx=0.25, rely=0.75, anchor='center')
        
        # Create target block
        target_index = self.target_field.get() - 1
        target_name = self.blocks[target_index]
        target_color, target_shape = target_name.split(' ')
        target_color_str = target_color + '2'
        self.target_frame = tk.Frame(self.popup, bg=target_color_str)
        if target_shape == 'square':    # Draw a square
            rows, cols = 2, 2
        elif target_shape == 'rect':    # Draw a rectangle
            rows, cols = 2, 4
        self.target_studs = np.zeros((rows, cols), dtype=bool)
        
        self.target_stud_buttons = []
        for r in range(rows):
            self.target_stud_buttons.append([])
            for c in range(cols):
                func = lambda b='target', n=target_name, r=r, c=c: self.toggleStud(b,n,r,c)
                b = tk.Button(self.target_frame, command=func, bg=target_color_str, 
                              activebackground=target_color_str, relief='ridge')
                b.grid(row=r, column=c)
                self.target_stud_buttons[-1].append(b)
       
        self.target_frame.place(relx=0.75, rely=0.33, anchor='center')
        rotate_target = lambda x='target', n=target_name, o='vertical': self.rotate(x, n, o)
        self.target_button = tk.Button(self.popup, command=rotate_target, text='rotate')
        self.target_button.place(relx=0.75, rely=0.75, anchor='center')
        
        swap_button = tk.Button(self.popup, command=self.swap, text='swap')
        swap_button.place(relx=0.5, rely=0.75, anchor='center')
        
        select_ok = tk.Button(self.popup, text='OK', command=self.addConnection)
        select_ok.place(relx=0.5, rely=0.9, anchor='center')
    
    def swap(self):
        """
        """
        
        # TODO
    
    
    def rotate(self, block, name, orientation):
        """
        Rotate a rectangular block by 90 degrees.
        
        Args:
        -----
        block: str
          'object' or 'target'
        """
        
        # Toggle between horizontal and vertical orientations
        rows, cols = None, None
        next_orientation = None
        if orientation == 'horizontal':
            rows, cols = 2, 4
            next_orientation = 'vertical'
        elif orientation == 'vertical':
            rows, cols = 4, 2
            next_orientation = 'horizontal'
        
        color, shape = name.split(' ')
        color_str = color + '2'
        
        if block == 'object':
            if shape == 'rect':
                # Update block buttons
                self.object_frame.destroy()
                self.object_frame = tk.Frame(self.popup, bg=color_str)
                self.object_studs = np.zeros((rows, cols), dtype=bool)
                self.object_stud_buttons = []
                for r in range(rows):
                    self.object_stud_buttons.append([])
                    for c in range(cols):
                        func = lambda b=block, n=name, r=r, c=c: self.toggleStud(b,n,r,c)
                        b = tk.Button(self.object_frame, command=func, bg=color_str, 
                                      activebackground=color_str, relief='ridge')
                        b.grid(row=r, column=c)
                        self.object_stud_buttons[-1].append(b)
                self.object_frame.place(relx=0.25, rely=0.33, anchor='center')
                
                # Update rotate button
                rotate_object = lambda x='object', n=name, o=next_orientation: self.rotate(x, n, o)
                self.object_button.configure(command=rotate_object)
                
        elif block == 'target':
            if shape == 'rect':
                # Update block buttons
                self.target_frame.destroy()
                self.target_frame = tk.Frame(self.popup, bg=color_str)
                self.target_studs = np.zeros((rows, cols), dtype=bool)
                self.target_stud_buttons = []
                for r in range(rows):
                    self.target_stud_buttons.append([])
                    for c in range(cols):
                        func = lambda b=block, n=name, r=r, c=c: self.toggleStud(b,n,r,c)
                        b = tk.Button(self.target_frame, command=func, bg=color_str, 
                                      activebackground=color_str, relief='ridge')
                        b.grid(row=r, column=c)
                        self.target_stud_buttons[-1].append(b)
                self.target_frame.place(relx=0.75, rely=0.33, anchor='center')
                
                # Update rotate button
                rotate_target = lambda x='target', n=name, o=next_orientation: self.rotate(x, n, o)
                self.target_button.configure(command=rotate_target)
    
    
    def toggleStud(self, block, name, row, col):
        """
        """
        
        color, shape = name.split(' ')
        
        # These are standard Tkinter color shades. Greater values are darker.
        color_str = color + '2'
        dark_color_str = color + '4'
        
        if block == 'object':
            self.object_studs[row, col] = not self.object_studs[row, col]
            bg_color = dark_color_str if self.object_studs[row, col] else color_str
            self.object_stud_buttons[row][col].config(bg=bg_color, activebackground=bg_color)
        elif block == 'target':
            self.target_studs[row, col] = not self.target_studs[row, col]
            bg_color = dark_color_str if self.target_studs[row, col] else color_str
            self.target_stud_buttons[row][col].config(bg=bg_color, activebackground=bg_color)
            
    
    def addConnection(self):
        """
        """
        
        # Get action annotation (action and object fields are one-indexed, so
        # convert to zero-indexing)
        action_index = self.action_field.get() - 1
        object_index = self.object_field.get() - 1
        target_index = self.target_field.get() - 1

        rows, cols = self.object_studs.nonzero()
        object_stud_str = ':'.join([''.join([str(r), str(c)]) for r, c in zip(rows, cols)])
        rows, cols = self.target_studs.nonzero()
        target_stud_str = ':'.join([''.join([str(r), str(c)]) for r, c in zip(rows, cols)])
        
        """
        # Make sure the user has selected an action and an object
        # (Value of -1 means original value was zero, ie empty)
        if action_index == -1 or object_index == -1:
            err_str = 'Please select and action and an object block.'
            self.badInputDialog(err_str)
            return
        
        action = self.actions[action_index]
        object_block = self.blocks[object_index]
        target_block = self.blocks[target_index]
        
        # Make sure the user has selected targets for actions that require them
        if not target_block and not action in ('remove', 'rotate'):
            err_str = 'Please select one or more target blocks.'
            self.badInputDialog(err_str)
            return
        """
        
        action = self.actions[action_index]
        
        # Parse and validate annotation
        err_str = self.parseAction(action, object_index, target_index)
        if err_str:
            self.badInputDialog(err_str)
            return
        
        # Store the action annotation
        connection = None
        if action in ('remove', 'rotate'):  # No targets
            connection = (self.action_start_index, self.cur_frame,
                          action_index, object_index, -1, '', '')
        else:
            connection = (self.action_start_index, self.cur_frame,
                          action_index, object_index, target_index,
                          object_stud_str, target_stud_str)
        self.labels.append(connection)
        print(connection)
        
        # Reset object stud and target stud arrays
        self.object_studs = None
        self.target_studs = None
        
        # Redraw world state image and stard/end button
        self.updateWorldState()
        self.cancel()
    
    
    def undoAction(self):
        """
        Delete the previous action annotation and re-draw block configuration.
        """
        
        if not self.states:
            error_string = 'No annotation to undo'
            self.badInputDialog(error_string)
        
        # Delete all labels associated with the last annotation
        last_label = self.labels[-1]
        while self.labels[-1][:3] == last_label[:3]:
            self.labels.pop()
        
        self.states = self.states[:-1]
        self.updateWorldState()
        
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
    
    
    def endAction(self):
        """
        When user indicates the end of an action, read and parse the annotation
        for this action. If the annotation is a valid one, display the new
        world state. If not, display a warning message to the user.
        """
        
        # Reset start and end indices
        self.action_start_index = -1
    
    
    def updateWorldState(self):
        """
        Redraw the block configuration graph which represents the world state.
        """
        
        state_fn = '{}.png'.format(len(self.states) - 1)
        state_path = os.path.join(self.state_fig_path, state_fn)
        state_image = ImageTk.PhotoImage(Image.open(state_path))
        self.state_display.configure(image=state_image)
        self.state_display.image = state_image
    
    
    def parseAction(self, action, object_index, target_index):
        """
        Update the world state by interpreting an action annotation.
        
        Args:
        -----
        action: str
          Action label (one of self.actions)
        object_index: int
          Block that was placed (index in self.blocks)
        target_index: int
          Blocks near the one that was placed (index in self.blocks)
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
            cur_state.edge(str(object_index), str(target_index))
        # Add an undirected edge for each target block
        elif action == 'place adjacent':
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
        error_string: str
          Message that will be displayed in popup window
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
    
    app = Application(root)
    
    root.mainloop()
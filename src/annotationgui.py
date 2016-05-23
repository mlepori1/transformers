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
        
        # Corpus object manages file I/O -- filenames and paths are filled in
        # after reading input from trial selection interface
        self.corpus = DuploCorpus()
        self.trial_id = None
        self.rgb_frame_fns = None
        self.state_fig_path = None
        
        # Initial values
        self.action_started = False
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
        print(state.render(filename=str(len(self.states)),
                           directory=self.state_fig_path))
        
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
        
        # Draw a listbox with a vertical scrollbar
        scrollbar = tk.Scrollbar(master, orient=tk.VERTICAL)
        self.trial_field = tk.Listbox(master, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.trial_field.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.trial_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        # Fill the listbox with trial data
        for entry in self.corpus.meta_data:
            trial_info = '{} {}'.format(entry['participant id'], entry['task id'])
            self.trial_field.insert(tk.END, trial_info)
        
        # Draw a select button
        master = self.navigation_frame
        submit = tk.Button(master, text='Select', command=self.submit)
        submit.pack()
        
        self.content_frame.place(relx=0.5, rely=0.5, anchor='center')
        self.navigation_frame.place(relx=0.5, rely=0.9, anchor='center')
    
    
    def submit(self):
        """
        Determine which trial the user selected for annotation.
        """
        
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
        rgb_frame = tk.Frame(frame1)
        
        rgb_frame_fn = self.rgb_frame_fns[self.cur_frame]
        rgb_image = ImageTk.PhotoImage(Image.open(rgb_frame_fn))
        self.rgb_display = tk.Label(rgb_frame, image=rgb_image)
        self.rgb_display.image = rgb_image
        self.rgb_display.pack()
        
        # Action start/end button
        if not self.action_started:
            button_text = 'Action started'
            button_cmd = self.startAction
        else:
            button_text = 'Action ended'
            button_cmd = self.endAction
        self.start_end = tk.Button(rgb_frame, text=button_text,
                                   command=button_cmd, default=tk.ACTIVE)
        self.start_end.pack()
        
        rgb_frame.pack(side=tk.LEFT)
        
        # Draw annotation frame
        ann_frame = tk.Frame(frame1)
        
        # Action label (radio buttons)
        action_label = tk.Label(ann_frame, text='Action')
        action_label.grid(row=0, column=0)
        self.action_field = tk.IntVar()
        for i, label in enumerate(self.labels):
            button = tk.Radiobutton(ann_frame, text=label,
                                    variable=self.action_field, value=i+1)
            button.grid(sticky=tk.W, row=i+1, column=0)
        
        # Object label (radio buttons)
        obj_label = tk.Label(ann_frame, text='Object')
        obj_label.grid(sticky=tk.W, row=0, column=1)
        self.object_field = tk.IntVar()
        for i, block in enumerate(self.blocks):
            button = tk.Radiobutton(ann_frame, text=block,
                                    variable=self.object_field, value=i+1)
            button.grid(sticky=tk.W, row=i+1, column=1)
        
        # Target label(s) (check boxes)
        target_label = tk.Label(ann_frame, text='Target(s)')
        target_label.grid(sticky=tk.W, row=0, column=2)
        for i, block in enumerate(self.blocks):
            self.target_fields[block] = tk.IntVar()
            target_box = tk.Checkbutton(ann_frame, text=block,
                                        variable=self.target_fields[block])
            target_box.grid(sticky=tk.W, row=i+1, column=2)
        
        ann_frame.pack(side=tk.LEFT)
        
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
        
        self.cur_frame = min(self.cur_frame + 10, len(self.rgb_frame_fns - 1))
        
        # Redraw rgb frame
        cur_fn = self.rgb_frame_fns[self.cur_frame]
        rgb_image = Image.open(cur_fn)
        rgb_image = ImageTk.PhotoImage(Image.open(cur_fn))
        self.rgb_frame.configure(image=rgb_image)
        self.rgb_frame.image = rgb_image
    
    
    def startAction(self):
        """
        Update start/end button when user indicates the start of an action.
        """
        
        print('Action started')
        self.action_started = True
        
        # Redraw button
        self.start_end.configure(text='End action', command=self.endAction)
    
    
    def endAction(self):
        """
        When user indicates the end of an action, read and parse the annotation
        for this action. If the annotation is a valid one, display the new
        world state. If not, display a warning message to the user.
        """
        
        # Get action annotation
        action = self.actions[self.action_field.get()]
        object_block = self.blocks[self.object_field.get()]
        target_blocks = tuple(block for block, field in self.target_fields.items()
                              if field.get())
        
        # Parse and validate annotation
        err_str = self.parseAction(action, object_block, target_blocks)
        if err_str:
            self.badInputDialog(err_str)
        
        print('Action ended')
        self.action_started = False
        
        # Redraw world state
        state_fn = '{}.png'.format(len(self.states) - 1)
        state_path = os.path.join(self.state_fig_path, state_fn)
        state_image = ImageTk.PhotoImage(Image.open(state_path))
        self.state_display.configure(image=state_image)
        self.state_display.image = state_image
        
        # Redraw button
        self.start_end.configure(text='Start action', command=self.startAction)
    
    
    def parseAction(self, action, object_block, target_blocks):
        """
        Update the world state by interpreting an action annotation.
        
        Args:
        -----
        [str] action: Action label -- one of self.actions
        [str] object_block: Block that was placed -- one of self.blocks
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
            for target_block in target_blocks:
                cur_state.edge(object_block, target_block)
        # Add an undirected edge for each target block
        elif action == 'place adjacent':
            for target_block in target_blocks:
                cur_state.edge(object_block, target_block, dir='none')
        # Remove all edges associated with this node
        elif action == 'remove':
            new_body = []
            for entry in cur_state.body:
                # Matches all edges originating from the removed block
                start_pattern = '\t\t{}'.format(object_block)
                # Matches all undirected edges leading to the removed block
                end_pattern = '-> {} [dir=none]'.format(object_block)
                if not (entry.startswith(start_pattern) or
                   entry.endswith(end_pattern)):
                    new_body.append(entry)
            cur_state.body = new_body
        
        # Save this state both to memory and to file
        self.states.append(cur_state)
        cur_state.render(filename=str(len(self.states)),
                         directory=self.state_fig_path)
        
        return 'I am a dummy function!'
    
    
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
    #root.geometry('{0}x{1}+0+0'.format(screen_width, screen_height))
    root.geometry('800x600')
    
    app = Application(root)
    
    root.mainloop()
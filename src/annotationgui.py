# -*- coding: utf-8 -*-
"""
annotationgui.py
  User interface for annotating trials from video frames.

AUTHOR
  Jonathan D. Jones
"""

from __future__ import print_function
import Tkinter as tk
import tkFont
from PIL import Image, ImageTk
import graphviz as gv
import os
import numpy as np

from duplocorpus import DuploCorpus

class Application:
    
    def __init__(self, parent):
        
        # Some text should be bold
        self.bold_font = tkFont.Font(size=10, weight=tkFont.BOLD)
        
        self.parent = parent
        self.popup = None
        
        # Constants
        self.actions = ('place above', 'place adjacent', 'disconnect',
                        'remove block', 'rotate 90 clockwise',
                        'rotate 90 counterclockwise', 'rotate 180')
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
        self.notes = []
        self.global_rotation = 0    # in degrees
        
        # User input (to be read from interface later)
        self.action_field = None
        self.object_field = None
        self.target_field = None
        self.trial_field = None
        
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
        
        user_text = 'Select a video to annotate:\n'
        instructions = tk.Label(master, text=user_text, font=self.bold_font)
        instructions.pack()
        
        # Draw a listbox with a vertical scrollbar
        scrollbar = tk.Scrollbar(master, orient=tk.VERTICAL)
        self.trial_field = tk.Listbox(master, height=20, yscrollcommand=scrollbar.set)
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
        self.parent.bind('u', self.skipBack)
        self.parent.bind('i', self.skipForward)
        self.parent.bind('k', self.forward)
        self.parent.bind('j', self.back)
        
        # Draw RGB frame
        interaction_frame = tk.Frame(master)
        
        # RGB video
        rgb_frame_fn = self.rgb_frame_fns[self.cur_frame]
        rgb_image = ImageTk.PhotoImage(Image.open(rgb_frame_fn))
        self.rgb_display = tk.Label(interaction_frame, image=rgb_image)
        self.rgb_display.image = rgb_image
        self.rgb_display.grid(row=0)
        
        # Draw annotation frame
        ann_frame = tk.Frame(interaction_frame)
        
        # Action label (radio buttons)
        action_label = tk.Label(ann_frame, text='\nAction', font=self.bold_font)
        action_label.grid(row=0, column=0)
        self.action_field = tk.IntVar()
        for i, label in enumerate(self.actions):
            button = tk.Radiobutton(ann_frame, text=label,
                                    variable=self.action_field, value=i+1)
            button.grid(sticky=tk.W, row=i+1, column=0)
        
        # Object label (radio buttons)
        obj_label = tk.Label(ann_frame, text='\nObject', font=self.bold_font)
        obj_label.grid(row=0, column=1)
        self.object_field = tk.IntVar()
        for i, block in enumerate(self.blocks):
            button = tk.Radiobutton(ann_frame, text=block,
                                    variable=self.object_field, value=i+1)
            button.grid(sticky=tk.W, row=i+1, column=1)
        
        # Target label (radio buttons)
        target_label = tk.Label(ann_frame, text='\nTarget', font=self.bold_font)
        target_label.grid(row=0, column=2)
        self.target_field = tk.IntVar()
        for i, block in enumerate(self.blocks):
            button = tk.Radiobutton(ann_frame, text=block,
                                    variable=self.target_field, value=i+1)
            button.grid(sticky=tk.W, row=i+1, column=2)
            
        row = max(len(self.actions), len(self.blocks)) + 1
        
        # Spacer (this is hacky)
        spacer_label = tk.Label(ann_frame, text='  ')
        spacer_label.grid(row=row, column=0, columnspan=3)
        
        # Notes box
        self.notes_field = tk.Entry(ann_frame, width=50)
        self.notes_field.grid(sticky=tk.W, row=row+1, column=0, columnspan=3)
        note_button = tk.Button(ann_frame, text='Save note', command=self.saveNote)
        note_button.grid(sticky=tk.W, row=row+1, column=3)
        
        ann_frame.grid(row=1)
        interaction_frame.grid(row=0, column=0)
        
        # Draw visualization of annotation
        visualization_frame = tk.Frame(master)
        state_fn = '{}.png'.format(len(self.states) - 1)
        state_path = os.path.join(self.state_fig_path, state_fn)
        state_image = ImageTk.PhotoImage(Image.open(state_path))
        self.state_display = tk.Label(visualization_frame, image=state_image)
        self.state_display.image = state_image
        self.state_display.grid(row=0)
        visualization_frame.grid(row=0, column=1)
        
        # Draw start/end, undo, restart, and quit buttons
        master = self.navigation_frame
        actions_frame = tk.Frame(master)
        self.start_end = tk.Button(actions_frame, text='Start of action',
                                   command=self.startAction, default=tk.ACTIVE)
        self.start_end.grid(sticky=tk.E, row=0, column=0)
        undo_action = tk.Button(actions_frame, text='Undo action',
                                command=self.undoAction)
        undo_action.grid(sticky=tk.W, row=0, column=1)
        select = tk.Button(actions_frame, text='Add connection', command=self.studSelectionDialog)
        select.grid(sticky=tk.W, row=0, column=2)
        undo_connection = tk.Button(actions_frame, text='Undo connection',
                                    command=self.undoConnection)
        undo_connection.grid(sticky=tk.W, row=0, column=3)
        actions_frame.pack(side=tk.TOP)
        
        # Draw restart and quit buttons
        end_frame = tk.Frame(master)
        restart = tk.Button(end_frame, text='Annotate a new video', command=self.restart)
        restart.grid(sticky=tk.W, row=0, column=0)
        close = tk.Button(end_frame, text='Quit', command=self.close)
        close.grid(sticky=tk.W, row=0, column=1)
        end_frame.pack(side=tk.TOP)
    
    
    def saveNote(self):
        """
        Save text annotation and current frame, then clear text box.
        """
        
        # TODO
        note = self.notes_field.get()
        self.notes_field.delete(0, tk.END)
        self.notes.append((self.cur_frame, note))
        
        print('{}  |  {}'.format(self.cur_frame, note))
        
        
    def studSelectionDialog(self):
        """
        Create a popup window to code block relations.
        """
        
        # Get coarse-level annotation info
        action_index = self.action_field.get() - 1
        object_index = self.object_field.get() - 1
        target_index = self.target_field.get() - 1
        
        # Make sure annotation is physically possible
        err_str = self.validateAction(action_index, object_index, target_index)
        if err_str:
            self.badInputDialog(err_str)
            return
        
        # Don't draw a window if a connection isn't being annotated -- go
        # straight to addConnection
        action = self.actions[action_index]
        if not action in ('place above', 'place adjacent'):
            self.addConnection()
            return
        
        # Destroy the popup window if it exists
        if self.popup:
            self.cancel()
        
        # Begin drawing popup
        self.popup = tk.Toplevel(self.parent)
        self.popup.geometry('{0}x{1}'.format(screen_width/4, screen_height/4))    
        
        # Draw target and object blocks
        self.drawBlockStuds(True , object_index)
        self.drawBlockStuds(False, target_index)
        
        swap_button = tk.Button(self.popup, command=self.swap, text='swap')
        swap_button.place(relx=0.5, rely=0.75, anchor='center')
        
        select_ok = tk.Button(self.popup, text='OK', command=self.addConnection)
        select_ok.place(relx=0.5, rely=0.9, anchor='center')
    
    
    def validateAction(self, action_index, object_index, target_index):
        """
        """
        
        if self.action_start_index < 0:
            err_str = 'Please click "start of action" before adding a connection.'
            return err_str
        
        # Make sure the user has selected an action and an object
        # (Value of -1 means original value was zero, ie empty)
        if action_index == -1:
            err_str = 'Please choose an action.'
            return err_str
                
        # Make sure the user has selected targets for actions that require them
        action = self.actions[action_index]
        if not action.startswith('rotate'):
            if object_index == -1:
                err_str = 'Please choose an object block.'
                return err_str
            if target_index == -1 and not action == 'remove block':
                err_str = 'Please choose a target block.'
                return err_str
    
    
    def drawBlockStuds(self, is_object, block_index):
        """
        """
        
        # Parse block name into shape and color
        name = self.blocks[block_index]
        color, shape = name.split(' ')
        color_str = color + '2'
        frame = tk.Frame(self.popup, bg=color_str)
        
        # Define layout depending on shape
        if shape == 'square':    # Draw a square
            rows, cols = 2, 2
        elif shape == 'rect':    # Draw a rectangle
            rows, cols = 2, 4
        block_studs = np.zeros((rows, cols), dtype=bool)
        
        # Create buttons
        stud_buttons = [] 
        for r in range(rows):
            stud_buttons.append([])
            for c in range(cols):
                func = lambda o=is_object, n=name, r=r, c=c: self.toggleStud(o,n,r,c)
                b = tk.Button(frame, command=func, bg=color_str, 
                              activebackground=color_str, relief='ridge')
                b.grid(row=r, column=c)
                stud_buttons[-1].append(b)
        
        # Draw frame and update member variables
        rotate = lambda o=is_object, h=False, n=name: self.rotate(o, h, n)
        if is_object:
            frame.place(relx=0.25, rely=0.33, anchor='center')
            self.object_button = tk.Button(self.popup, command=rotate, text='rotate')
            self.object_button.place(relx=0.25, rely=0.75, anchor='center')
            
            self.object_frame = frame
            self.object_studs = block_studs
            self.object_stud_buttons = stud_buttons
        else:
            frame.place(relx=0.75, rely=0.33, anchor='center')
            self.target_button = tk.Button(self.popup, command=rotate, text='rotate')
            self.target_button.place(relx=0.75, rely=0.75, anchor='center')
            
            self.target_frame = frame
            self.target_studs = block_studs
            self.target_stud_buttons = stud_buttons
        
    
    def swap(self):
        """
        """
        
        # TODO
    
    
    def rotate(self, is_object, to_horizontal, name):
        """
        Rotate a rectangular block by 90 degrees.
        
        Args:
        -----
        """
        
        # Toggle between horizontal and vertical orientations
        rows, cols = None, None
        if to_horizontal:
            rows, cols = 2, 4
        else:   # to vertical orientation
            rows, cols = 4, 2
        
        color, shape = name.split(' ')
        color_str = color + '2'
        
        if is_object:
            if shape == 'rect':
                # Update block buttons
                self.object_frame.destroy()
                self.object_frame = tk.Frame(self.popup, bg=color_str)
                self.object_studs = np.zeros((rows, cols), dtype=bool)
                self.object_stud_buttons = []
                for r in range(rows):
                    self.object_stud_buttons.append([])
                    for c in range(cols):
                        func = lambda o=is_object, n=name, r=r, c=c: self.toggleStud(o,n,r,c)
                        b = tk.Button(self.object_frame, command=func, bg=color_str, 
                                      activebackground=color_str, relief='ridge')
                        b.grid(row=r, column=c)
                        self.object_stud_buttons[-1].append(b)
                self.object_frame.place(relx=0.25, rely=0.33, anchor='center')
                
                # Update rotate button
                rotate = lambda o=is_object, h=not to_horizontal, n=name: self.rotate(o, h, n)
                self.object_button.configure(command=rotate)
                
        else:
            if shape == 'rect':
                # Update block buttons
                self.target_frame.destroy()
                self.target_frame = tk.Frame(self.popup, bg=color_str)
                self.target_studs = np.zeros((rows, cols), dtype=bool)
                self.target_stud_buttons = []
                for r in range(rows):
                    self.target_stud_buttons.append([])
                    for c in range(cols):
                        func = lambda o=is_object, n=name, r=r, c=c: self.toggleStud(o,n,r,c)
                        b = tk.Button(self.target_frame, command=func, bg=color_str, 
                                      activebackground=color_str, relief='ridge')
                        b.grid(row=r, column=c)
                        self.target_stud_buttons[-1].append(b)
                self.target_frame.place(relx=0.75, rely=0.33, anchor='center')
                
                # Update rotate button
                rotate = lambda o=is_object, h=not to_horizontal, n=name: self.rotate(o, h, n)
                self.target_button.configure(command=rotate)
    
    
    def toggleStud(self, is_object, name, row, col):
        """
        """
        
        color, shape = name.split(' ')
        
        # These are standard Tkinter color shades. Greater values are darker.
        color_str = color + '2'
        dark_color_str = color + '4'
        
        if is_object:
            self.object_studs[row, col] = not self.object_studs[row, col]
            bg_color = dark_color_str if self.object_studs[row, col] else color_str
            self.object_stud_buttons[row][col].config(bg=bg_color, activebackground=bg_color)
        else:
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

        action = self.actions[action_index]
        
        if not action in ('place above', 'place adjacent'):
            err_str = 'Only add connections for "place above" and "place adjacent."'
            self.badInputDialog(err_str)
            return
        
        # Parse and validate annotation
        err_str = self.parseAction(action, object_index, target_index)
        if err_str:
            self.badInputDialog(err_str)
            return
        
        # Build a string representing the connected studs on the object block
        rows, cols = self.object_studs.nonzero()
        prefix = ''
        num_rows, num_cols = self.object_studs.shape
        if num_rows > num_cols:
            prefix = 'V'
        elif num_rows < num_cols:
            prefix = 'H'
        else:   # num_rows == num_cols
            prefix = 'S'
        coord_strings = [''.join([str(r), str(c)]) for r, c in zip(rows, cols)]
        object_stud_str = prefix + ':'.join(coord_strings)
        
        # Build a string representing the connected studs on the target block
        rows, cols = self.target_studs.nonzero()
        prefix = ''
        num_rows, num_cols = self.target_studs.shape
        if num_rows > num_cols:
            prefix = 'V'
        elif num_rows < num_cols:
            prefix = 'H'
        else:   # num_rows == num_cols
            prefix = 'S'
        coord_strings = [''.join([str(r), str(c)]) for r, c in zip(rows, cols)]
        target_stud_str = prefix + ':'.join(coord_strings)
        
        connection = (self.action_start_index, self.cur_frame, action_index,
                      object_index, target_index, object_stud_str,
                      target_stud_str)
        self.labels.append(connection)
        print(connection)
        
        # Reset object stud and target stud arrays
        self.object_studs = None
        self.target_studs = None
        
        # Redraw world state image
        self.updateWorldState()
        self.cancel()
    
    
    def undoConnection(self):
        """
        Delete the previous connection annotation and re-draw block
        configuration.
        """
        
        if not self.labels:
            error_string = 'No connection to undo!'
            self.badInputDialog(error_string)
            return
        
        # Delete the last label if it represents a graph edge
        last_label = self.labels[-1]
        if self.actions[last_label[2]] in ('place above', 'place adjacent'):
            self.labels.pop()
            self.states.pop()
            self.updateWorldState()
    
    
    def undoAction(self):
        """
        Delete the previous action annotation and re-draw the block
        configuration.
        """
        
        if not self.labels:
            error_string = 'No action to undo!'
            self.badInputDialog(error_string)
            return
        
        last_label = self.labels[-1]
        last_action = self.actions[last_label[2]]
        # Delete all edges annotated during the last action
        if last_action in ('place above', 'place adjacent'):
            start_index = last_label[0]
            while self.labels and self.labels[-1][0] == start_index:
                self.labels.pop()
                self.states.pop()
            self.updateWorldState()
        # Delete disconnect-type annotations
        elif last_action in ('disconnect', 'remove block'):
            self.labels.pop()
            self.states.pop()
            self.updateWorldState()
        else:   # rotate 90, rotate -90, rotate 180 don't alter state graph
            self.labels.pop()
            message_str = 'Removed action: {}'.format(last_action)
            self.badInputDialog(message_str)
    
    
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
        Move backward num_frames in the video unless the beginning of the video
        is fewer than num_frames away. Then, display the new rgb frame.
        """
        
        self.cur_frame = max(self.cur_frame - 1, 0)
        
        # Redraw rgb frame
        cur_fn = self.rgb_frame_fns[self.cur_frame]
        rgb_image = ImageTk.PhotoImage(Image.open(cur_fn))
        self.rgb_display.configure(image=rgb_image)
        self.rgb_display.image = rgb_image
    
    
    def forward(self, num_frames, event=None):
        """
        Move forward num_frames in the video unless the end of the video is
        fewer than num_frames away. Then, display the new rgb frame.
        """
        
        last_frame = len(self.rgb_frame_fns) - 1
        self.cur_frame = min(self.cur_frame + 1, last_frame)
        
        # Redraw rgb frame
        cur_fn = self.rgb_frame_fns[self.cur_frame]
        rgb_image = Image.open(cur_fn)
        rgb_image = ImageTk.PhotoImage(Image.open(cur_fn))
        self.rgb_display.configure(image=rgb_image)
        self.rgb_display.image = rgb_image
    
    
    def skipBack(self, event=None):
        """
        """
        
        # FIXME
        
        self.cur_frame = max(self.cur_frame - 10, 0)
        
        # Redraw rgb frame
        cur_fn = self.rgb_frame_fns[self.cur_frame]
        rgb_image = ImageTk.PhotoImage(Image.open(cur_fn))
        self.rgb_display.configure(image=rgb_image)
        self.rgb_display.image = rgb_image
        
    
    def skipForward(self, event=None):
        """
        """
        
        # FIXME
        
        last_frame = len(self.rgb_frame_fns) - 1
        self.cur_frame = min(self.cur_frame + 10, last_frame)
        
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
        
        # Get action annotation (action and object fields are one-indexed, so
        # convert to zero-indexing)
        action_index = self.action_field.get() - 1
        object_index = self.object_field.get() - 1
        target_index = self.target_field.get() - 1

        action = self.actions[action_index]
        
        # Store the action annotation
        # Place above and place adjacent actions were already processed when
        # connections were added.
        if not action in ('place above', 'place adjacent'):
            connection = None
            if action.startswith('rotate'):
                # World state is unchanged
                connection = (self.cur_frame, self.cur_frame,
                              action_index, -1, -1, '', '')
            elif action in ('disconnect', 'remove block'):
                # Parse action and draw new world state
                err_str = self.parseAction(action, object_index, target_index)
                if err_str:
                    self.badInputDialog(err_str)
                    return
                self.updateWorldState()
                connection = (self.action_start_index, self.cur_frame,
                              action_index, object_index, -1, '', '')                
            self.labels.append(connection)
            print(connection)
        
        # Reset start and end indices
        self.action_start_index = -1
        
        # Redraw button
        self.start_end.configure(text='Start of action', command=self.startAction)
    
    
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
        elif action == 'remove block':
            new_body = []
            for entry in cur_state.body:
                # Matches all edges originating from the removed block
                start_pattern = '\t\t{}'.format(object_index)
                # Matches all undirected edges leading to the removed block
                end_pattern_undirected = '-> {} [dir=none]'.format(object_index)
                end_pattern_directed = '-> {}'.format(object_index)
                if not (entry.startswith(start_pattern) or
                   entry.endswith(end_pattern_undirected) or
                   entry.endswith(end_pattern_directed)):
                    new_body.append(entry)
            cur_state.body = new_body
        # Remove the edge between object and target
        elif action == 'disconnect':
            new_body = []
            pattern_str = '\t\t{} -> {}'
            for entry in cur_state.body:
                if not (entry.startswith(pattern_str.format(object_index, target_index)) or
                    entry.startswith(pattern_str.format(target_index, object_index))):
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
        
        if self.popup:
            self.popup.destroy()
        self.popup = None
        

if __name__ == '__main__':
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry('{0}x{1}+0+0'.format(screen_width, screen_height))
    
    app = Application(root)
    
    root.mainloop()
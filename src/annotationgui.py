# -*- coding: utf-8 -*-
"""
annotationgui.py
  User interface for annotating trials from video frames.

AUTHOR
  Jonathan D. Jones
"""

from __future__ import print_function
from Tkinter import *
from PIL import Image, ImageTk

from duplocorpus import DuploCorpus

class Application:
    
    def __init__(self, parent):
        
        self.parent = parent
        
        self.parent.bind('k', self.forward)
        self.parent.bind('j', self.back)
        
        self.blocks = ('red square', 'yellow square', 'green square',
                       'blue square', 'red rect', 'yellow rect', 'green rect',
                       'blue rect')
        
        self.action_started = False
        
        self.corpus = DuploCorpus()
        self.trial_id = 1
        self.rgb_frame_fns = self.corpus.getRgbFrameFns(self.trial_id)
        self.cur_frame = 0
        
        self.obj_vars = {}
        self.target_vars = {}
        
        """
        # Handle for a popup window (we only want one at a time)
        self.popup = None
        
        self.cur_position = 0
        self.controlflow = (self.drawInfoContent, self.drawTaskContent,
                            self.drawImuContent)
        self.getters = (self.getMetaData, self.getTaskData, self.getImuData)
        """
        
        self.window_w = 768
        self.window_h = 768
        self.parent.geometry('{}x{}'.format(self.window_w, self.window_h))
        
        """
        # This is updated after prompting user for input
        self.imu_ids = ('08F1', '0949', '090F', '095D')
        self.block2imu = {x: 'UNUSED' for x in self.imu_ids}
        self.frame_vars = ()
        self.active_blocks = ()
        """
        
        self.content = self.defaultFrame()
        self.drawAnnotator()
        
        """
        cur_frame = self.controlflow[self.cur_position]
        cur_frame()
        """
    
    def defaultFrame(self):
        return Frame(self.parent)
    
    
    def drawAnnotator(self):
        
        master = self.content
        
        frame1 = Frame(master)
        
        # Draw rgb frame
        rgb_frame = Frame(frame1)
        
        cur_fn = self.rgb_frame_fns[self.cur_frame]
        image = Image.open(cur_fn)
        photo = ImageTk.PhotoImage(image)
        label = Label(rgb_frame, image=photo)
        label.image = photo
        label.pack()
        
        if not self.action_started:
            button_text = 'Action started'
            button_cmd = self.startAction
        else:
            button_text = 'Action ended'
            button_cmd = self.endAction
        start_end = Button(rgb_frame, text=button_text, command=button_cmd,
                           default=ACTIVE)
        start_end.pack()
        
        rgb_frame.pack(side=LEFT)
        
        # Draw annotation frame
        ann_frame = Frame(frame1)
        
        # Action label
        action_label = Label(ann_frame, text='Action')
        
        # Set up radio button widgets
        self.task = IntVar()
        b1 = Radiobutton(ann_frame, text="place above", variable=self.task, value=1)
        b2 = Radiobutton(ann_frame, text="place adjacent", variable=self.task, value=2)
        b3 = Radiobutton(ann_frame, text="rotate", variable=self.task, value=3)
        b4 = Radiobutton(ann_frame, text="translate", variable=self.task, value=4)
        b5 = Radiobutton(ann_frame, text="remove", variable=self.task, value=5)
        b6 = Radiobutton(ann_frame, text="pick up", variable=self.task, value=6)
        
        action_label.grid(row=0, column=0)
        b1.grid(sticky=W, row=1, column=0)
        b2.grid(sticky=W, row=2, column=0)
        b3.grid(sticky=W, row=3, column=0)
        b4.grid(sticky=W, row=4, column=0)
        b5.grid(sticky=W, row=5, column=0)
        b6.grid(sticky=W, row=6, column=0)
        
        # Object label(s)
        obj_label = Label(ann_frame, text='Object(s)')
        obj_label.grid(sticky=W, row=0, column=1)
        obj_boxes = []
        for i, block in enumerate(self.blocks):
            self.obj_vars[block] = IntVar()
            obj_boxes.append(Checkbutton(ann_frame, text=block,
                                         variable=self.obj_vars[block]))
            obj_boxes[-1].grid(sticky=W, row=i+1, column=1)
        
        # Target label(s)
        target_label = Label(ann_frame, text='Target(s)')
        target_label.grid(sticky=W, row=0, column=2)
        target_boxes = []
        for i, block in enumerate(self.blocks):
            self.target_vars[block] = IntVar()
            target_boxes.append(Checkbutton(ann_frame, text=block,
                                            variable=self.target_vars[block]))
            target_boxes[-1].grid(sticky=W, row=i+1, column=2)
        
        ann_frame.pack(side=LEFT)
        
        frame1.pack(side=TOP)
        
        frame2 = Frame(master)
        state_fn = '/home/jdjones/state.png'
        state_image = Image.open(state_fn)
        state_photo = ImageTk.PhotoImage(state_image)
        state_label = Label(frame2, image=state_photo)
        state_label.image = state_photo
        state_label.pack()
        frame2.pack(side=BOTTOM)
        
        
        master.place(relx=0.5, rely=0.5, anchor='center')
    
    
    def back(self, event=None):
        if self.cur_frame  - 20 > 0:
            self.cur_frame -= 20
        else:
            self.cur_frame = 0
        
        self.content.destroy()
        self.content = self.defaultFrame()
        self.drawAnnotator()
    
    
    def forward(self, event=None):
        if self.cur_frame  + 20 < len(self.rgb_frame_fns) - 1:
            self.cur_frame += 20
        else:
            self.cur_frame = len(self.rgb_frame_fns) - 1
        
        self.content.destroy()
        self.content = self.defaultFrame()
        self.drawAnnotator()
    
    def startAction(self):
        print('Action started')
        self.action_started = True
        
        # Redraw frame
        self.content.destroy()
        self.content = self.defaultFrame()
        self.drawAnnotator()
    
    
    def endAction(self):
        print('Action ended')
        self.action_started = False
        
        # Redraw frame
        self.content.destroy()
        self.content = self.defaultFrame()
        self.drawAnnotator()
        

if __name__ == '__main__':
    root = Tk()
    
    app = Application(root)
    
    root.mainloop()
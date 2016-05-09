# -*- coding: utf-8 -*-
"""
guitest.py

AUTHOR
  Jonathan D. Jones
"""

from __future__ import print_function
from Tkinter import *
from PIL import Image, ImageTk


class Application:
    
    def __init__(self, parent):
        
        self.parent = parent
        
        # Handle for a popup window (we only want one at a time)
        self.popup = None
        
        self.cur_position = 0
        self.controlflow = (self.drawInfoContent, self.drawTaskContent,
                            self.drawImuContent)
        self.getters = (self.getMetaData, self.getTaskData, self.getImuData)
        
        self.window_w = 768
        self.window_h = 576
        self.parent.geometry('{}x{}'.format(self.window_w, self.window_h))
        
        # This is updated after prompting user for input
        self.imu_ids = ('08F1', '0949', '090F', '095D')
        self.block2imu = {x: 'UNUSED' for x in self.imu_ids}
        self.frame_vars = ()
        self.active_blocks = ()
        
        self.content = self.defaultFrame()
        
        cur_frame = self.controlflow[self.cur_position]
        cur_frame()
    
    
    def defaultFrame(self):
        return Frame(self.parent)
    
    
    def drawInfoContent(self):
        
        master = self.content
        
        # Metadata category labels
        user_text = "Please fill in the following fields for this child."
        instructions = Label(master, text=user_text)
        pnum_label = Label(master, text="Participant number")
        dob_label = Label(master, text="Date of birth")
        gender_label = Label(master, text="Gender")
        
        # Set up widgets for metadata collection
        self.participant_num = Entry(master)
        
        months = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                  'Sep', 'Oct', 'Nov', 'Dec')
        self.dob_month = StringVar(master)
        self.dob_month.set(months[0])
        month_menu = apply(OptionMenu, (master, self.dob_month) + months)
        
        years = (2010, 2011, 2013, 2014)
        self.dob_year = StringVar(master)
        self.dob_year.set(years[0])
        year_menu = apply(OptionMenu, (master, self.dob_year) + years)
        
        genders = ('Male', 'Female', 'Not disclosed')
        self.gender = StringVar(master)
        self.gender.set(genders[0])
        gender_menu = apply(OptionMenu, (master, self.gender) + genders)
        
        # Define widget layout
        instructions.grid(row=0, columnspan=3)
        pnum_label.grid(sticky=E, row=1)
        dob_label.grid(sticky=E, row=2)
        gender_label.grid(sticky=E, row=3)
        self.participant_num.grid(sticky=W, row=1, column=1, columnspan=2)
        month_menu.grid(sticky=W, row=2, column=1)
        year_menu.grid(sticky=W, row=2, column=2)
        gender_menu.grid(sticky=W, row=3, column=1)
            
        # Add 'submit' box
        submit = Button(master, text="Next", command=self.forward, default=ACTIVE)
        submit.grid(sticky=E, row=4, column=2)
        
        master.place(relx=0.5, rely=0.5, anchor='center')
    
    
    def drawTaskContent(self):
        
        master = self.content
        
        user_text = "Select the current block construction."
        instructions = Label(master, text=user_text)
        
        # Set up radio button widgets
        self.task = IntVar()
        b1 = Radiobutton(master, text="1", variable=self.task, value=1)
        b2 = Radiobutton(master, text="2", variable=self.task, value=2)
        b3 = Radiobutton(master, text="3", variable=self.task, value=3)
        b4 = Radiobutton(master, text="4", variable=self.task, value=4)
        b5 = Radiobutton(master, text="5", variable=self.task, value=5)
        b6 = Radiobutton(master, text="6", variable=self.task, value=6)
        
        instructions.grid(row=0, columnspan=3)
        b1.grid(row=1, column=0)
        b2.grid(row=1, column=1)
        b3.grid(row=1, column=2)
        b4.grid(row=2, column=0)
        b5.grid(row=2, column=1)
        b6.grid(row=2, column=2)
        
        # Add 'submit' box
        submit = Button(master, text="Next", command=self.forward, default=ACTIVE)
        submit.grid(sticky=E, row=3, column=2)
        
        back = Button(master, text="Back", command=self.back)
        back.grid(sticky=W, row=3, column=0)
        
        master.place(relx=0.5, rely=0.5, anchor='center')


    def drawImuContent(self):
        
        master = self.content
        
        # Metadata category labels
        user_text = "Place IMUs in the following blocks and select their IDs."
        instructions = Label(master, text=user_text)
        instructions.grid(sticky=W, row=0, columnspan=3)
    
        # Set up widgets and define layout
        self.dev_ids = {}
        menus = []
        buttons = []
        commands = []
        for i, block in enumerate(self.active_blocks):
            
            id_label = Label(master, text='{} rectangle'.format(block))
            id_label.grid(sticky=E, row=i+1, column=0)
            
            self.dev_ids[block] = StringVar(master)
            self.dev_ids[block].set(self.imu_ids[i])
            
            menus.append(apply(OptionMenu, (master, self.dev_ids[block]) + self.imu_ids))
            menus[-1].grid(sticky=W, row=i+1, column=1)
            
            commands.append(lambda b=str(block): self.connect(b))
            buttons.append(Button(master, text='Connect', command=commands[-1]))
            buttons[-1].grid(sticky=W, row=i+1, column=2)
            
        submit = Button(master, text='Next', command=self.forward, default=ACTIVE)
        submit.grid(sticky=E, row=len(self.dev_ids)+1, column=2)
        
        back = Button(master, text='Back', command=self.back)
        back.grid(sticky=W, row=len(self.dev_ids)+1, column=0)
        
        master.place(relx=0.5, rely=0.5, anchor='center')
    
    
    def connect(self, block):
        
        if not self.popup is None:
            return
        
        dev_id = self.dev_ids[block]
        imu_id = dev_id.get()
        
        self.popup = Toplevel(self.parent)
        fmtstr = 'Pretending to connect to {}...'
        l = Label(self.popup, text=fmtstr.format(imu_id))
        l.pack()
        
        c = Button(self.popup, text='cancel', command=self.cancel)
        c.pack()
    
    
    def cancel(self):
        self.popup.destroy()
        self.popup = None
    
    
    def back(self):
        
        if self.cur_position > 0:
            self.cur_position -= 1
        
        self.content.destroy()
        self.content = self.defaultFrame()
        
        cur_frame = self.controlflow[self.cur_position]
        cur_frame()

    
    def forward(self):
        getter = self.getters[self.cur_position]
        getter()
        
        if self.cur_position < len(self.controlflow) - 1:
            self.cur_position += 1
        
        self.content.destroy()
        self.content = self.defaultFrame()
        
        cur_frame = self.controlflow[self.cur_position]
        cur_frame()
    
    
    def getMetaData(self):
        pnum = self.participant_num.get()
        dob_month = self.dob_month.get()
        dob_year = self.dob_year.get()
        gender = self.gender.get()
        
        print('Participant ID: {}'.format(pnum))
        print('Birth date: {} / {}'.format(dob_month, dob_year))
        print('Gender: {}'.format(gender))
    
    
    def getTaskData(self):
        task = self.task.get()
        print('Task: {}'.format(task))
        
        if task == 1:
            self.active_blocks = ('red', 'yellow')
        elif task == 2:
            self.active_blocks = ('green', 'blue')
        elif task in (3, 4, 5, 6):
            self.active_blocks = ('red', 'yellow', 'green', 'blue')
        
    
    def getImuData(self):
        
        for block, dev_id in self.dev_ids.items():
            self.block2imu[block] = dev_id.get()
            print('{}: {}'.format(block, self.block2imu[block]))


    def dummy(self):
        print('submitted')


if __name__ == '__main__':
    
    root = Tk()
    #app = Application(root)
    
    image = Image.open('/Users/jonathan/0.png')
    photo = ImageTk.PhotoImage(image)
    label = Label(image=photo)
    label.image = photo
    label.pack()
    
    root.mainloop()
    
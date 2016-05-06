# -*- coding: utf-8 -*-
"""
guitest.py

AUTHOR
  Jonathan D. Jones
"""

from Tkinter import *
from PIL import Image, ImageTk
from tksimpledialog import Dialog


""" 
def applyinfo(self):
    
    # Read data
    p_num = self.participant_num.get()
    month = self.dob_month.get()
    year = self.dob_year.get()
    gen = self.gender.get()
    
    print(' | '.join((p_num, month, year, gen)))


def imubody(self, master):
    
    # Metadata category labels
    user_text = "Put an IMU in the [XXX] block."
    instructions = Label(master, text=user_text)
    id_label = Label(master, text="Device ID")

    # Set up widgets for metadata collection
    dev_ids = ('08F1', '0949', '090F', '095D')
    self.dev_id = StringVar(master)
    self.dev_id.set(dev_ids[0])
    id_menu = apply(OptionMenu, (master, self.dev_id) + dev_ids)
    
    # Define widget layout
    instructions.grid(row=0, columnspan=2)
    id_label.grid(sticky=E, row=1)
    id_menu.grid(sticky=W, row=1, column=1)
    
    return id_menu # initial focus

def applyblocks(self):
    
    # Read data
    dev_id = self.dev_id.get()
    
    print(dev_id)
"""

def infoContent(master):
    """ Prompt user for metadata """
    # Metadata category labels
    user_text = "Please fill in the following fields for this child."
    instructions = Label(master, text=user_text)
    pnum_label = Label(master, text="Participant number")
    dob_label = Label(master, text="Date of birth")
    gender_label = Label(master, text="Gender")
    
    # Set up widgets for metadata collection
    participant_num = Entry(master)
    
    months = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
              'Sep', 'Oct', 'Nov', 'Dec')
    dob_month = StringVar(master)
    dob_month.set(months[0])
    month_menu = apply(OptionMenu, (master, dob_month) + months)
    
    years = (2010, 2011, 2013, 2014)
    dob_year = StringVar(master)
    dob_year.set(years[0])
    year_menu = apply(OptionMenu, (master, dob_year) + years)
    
    genders = ('Male', 'Female', 'Not disclosed')
    gender = StringVar(master)
    gender.set(genders[0])
    gender_menu = apply(OptionMenu, (master, gender) + genders)
    
    # Define widget layout
    instructions.grid(row=0, columnspan=3)
    pnum_label.grid(sticky=E, row=1)
    dob_label.grid(sticky=E, row=2)
    gender_label.grid(sticky=E, row=3)
    participant_num.grid(sticky=W, row=1, column=1, columnspan=2)
    month_menu.grid(sticky=W, row=2, column=1)
    year_menu.grid(sticky=W, row=2, column=2)
    gender_menu.grid(sticky=W, row=3, column=1)
        
    # Add 'submit' box
    #proceed = lambda x: submit(master, x)
    def submit():
        master.destroy()
        master = Frame(root)
        imuContent(master)
        master.pack()
    
    w = Button(master, text="Submit", command=submit, default=ACTIVE)
    w.grid(sticky=E, row=4, column=2)


def imuContent(master):
    # Metadata category labels
    user_text = "Put an IMU in the [XXX] block."
    instructions = Label(master, text=user_text)
    id_label = Label(master, text="Device ID")

    # Set up widgets for metadata collection
    dev_ids = ('08F1', '0949', '090F', '095D')
    dev_id = StringVar(master)
    dev_id.set(dev_ids[0])
    id_menu = apply(OptionMenu, (master, dev_id) + dev_ids)
    
    # Define widget layout
    instructions.grid(row=0, columnspan=2)
    id_label.grid(sticky=E, row=1)
    id_menu.grid(sticky=W, row=1, column=1)
    
    w = Button(master, text="Submit", command=dummy, default=ACTIVE)
    w.grid(sticky=E, row=2, column=1)
    




def dummy():
    print('submitted')


if __name__ == '__main__':
    
    root = Tk()
    
    master = Frame(root)
    infoContent(master)
    master.pack()
    
    root.mainloop()
    
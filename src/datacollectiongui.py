# -*- coding: utf-8 -*-
"""
datacollectiongui.py

AUTHOR
  Jonathan D. Jones
"""

from __future__ import print_function
import Tkinter as tk
from PIL import Image, ImageTk

import multiprocessing as mp

import os

import numpy as np

import libwax9 as wax9
import libprimesense as ps
from duplocorpus import DuploCorpus


class Application:
    
    def __init__(self, parent):
        """
        Args:
        -----
        [tk window] parent: Parent tkinter object
        """
        
        self.parent = parent
        
        # Handle for a popup window (we only want one at a time)
        self.popup = None
        
        self.cur_position = 0
        self.controlflow = (self.drawInfoContent, self.drawTaskContent,
                            self.drawImuContent, self.drawStreamContent)
        self.getters = (self.getMetaData, self.getTaskData, self.getImuData,
                        self.getStreamData)
        
        self.window_w = 768
        self.window_h = 576
        self.parent.geometry('{}x{}'.format(self.window_w, self.window_h))
        
        # This is updated after prompting user for input
        self.connected_devices = {}
        self.camera_ids = ('rgb',) #'depth')
        self.imu2block = {x: 'UNUSED' for x in self.corpus.imu_ids}
        self.frame_vars = ()
        self.active_blocks = ()
        self.trial_metadata = ()
        self.imu_settings = []
        
        # For streaming in parallel
        self.die = mp.Event()
        self.corpus = DuploCorpus()
        self.trial_id = self.corpus.meta_data.shape[0]
        
        fn = '{}-imu.csv'.format(self.trial_id)
        path = os.path.join(self.corpus.paths['raw'], fn)
        frame_path = os.path.join(self.corpus.paths['video-frames'],
                                  '{}-rgb'.format(self.trial_id))
        timestamp_fn = '{}-timestamps.csv'.format(self.trial_id)
        timestamp_path = os.path.join(self.corpus.paths['frame-timestamps'],
                                      timestamp_fn)
        self.processes = (mp.Process(target=wax9.stream,
                                     args=(self.connected_devices, path, self.die)),
                          mp.Process(target=ps.stream,
                                     args=(frame_path, timestamp_path, self.die)))
        
        self.content = self.defaultFrame()
        
        cur_frame = self.controlflow[self.cur_position]
        cur_frame()
    
    
    def defaultFrame(self):
        return tk.Frame(self.parent)
    
    
    def drawInfoContent(self):
        """
        Draw the metadata collection window.
        """
        
        master = self.content
        
        # Metadata category labels
        user_text = "Please fill in the following fields for this child."
        instructions = tk.Label(master, text=user_text)
        pnum_label = tk.Label(master, text="Participant number")
        dob_label = tk.Label(master, text="Date of birth")
        gender_label = tk.Label(master, text="Gender")
        
        # Set up widgets for metadata collection
        self.participant_num = tk.Entry(master)
        
        months = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                  'Sep', 'Oct', 'Nov', 'Dec')
        self.dob_month = tk.StringVar(master)
        self.dob_month.set(months[0])
        month_menu = apply(tk.OptionMenu, (master, self.dob_month) + months)
        
        years = (2010, 2011, 2013, 2014)
        self.dob_year = tk.StringVar(master)
        self.dob_year.set(years[0])
        year_menu = apply(tk.OptionMenu, (master, self.dob_year) + years)
        
        genders = ('Male', 'Female', 'Not disclosed')
        self.gender = tk.StringVar(master)
        self.gender.set(genders[0])
        gender_menu = apply(tk.OptionMenu, (master, self.gender) + genders)
        
        # Define widget layout
        instructions.grid(row=0, columnspan=3)
        pnum_label.grid(sticky=tk.E, row=1)
        dob_label.grid(sticky=tk.E, row=2)
        gender_label.grid(sticky=tk.E, row=3)
        self.participant_num.grid(sticky=tk.W, row=1, column=1, columnspan=2)
        month_menu.grid(sticky=tk.W, row=2, column=1)
        year_menu.grid(sticky=tk.W, row=2, column=2)
        gender_menu.grid(sticky=tk.W, row=3, column=1)
            
        # Add 'submit' box
        submit = tk.Button(master, text="Next", command=self.forward,
                           default=tk.ACTIVE)
        submit.grid(sticky=tk.E, row=4, column=2)
        
        master.place(relx=0.5, rely=0.5, anchor='center')
    
    
    def drawTaskContent(self):
        """
        Draw the block task selection window.
        """
        
        master = self.content
        
        user_text = "Select the current block construction."
        instructions = tk.Label(master, text=user_text)
        
        # Set up radio button widgets
        # TODO: Add images to radio buttons
        self.task = tk.IntVar()
        b1 = tk.Radiobutton(master, text='1', variable=self.task, value=1)
        b2 = tk.Radiobutton(master, text='2', variable=self.task, value=2)
        b3 = tk.Radiobutton(master, text='3', variable=self.task, value=3)
        b4 = tk.Radiobutton(master, text='4', variable=self.task, value=4)
        b5 = tk.Radiobutton(master, text='5', variable=self.task, value=5)
        b6 = tk.Radiobutton(master, text='6', variable=self.task, value=6)
        
        instructions.grid(row=0, columnspan=3)
        b1.grid(row=1, column=0)
        b2.grid(row=1, column=1)
        b3.grid(row=1, column=2)
        b4.grid(row=2, column=0)
        b5.grid(row=2, column=1)
        b6.grid(row=2, column=2)
        
        # Add 'submit' box
        submit = tk.Button(master, text="Next", command=self.forward,
                           default=tk.ACTIVE)
        submit.grid(sticky=tk.E, row=3, column=2)
        
        back = tk.Button(master, text="Back", command=self.back)
        back.grid(sticky=tk.W, row=3, column=0)
        
        master.place(relx=0.5, rely=0.5, anchor='center')


    def drawImuContent(self):
        """
        Draw the IMU connection window.
        """
        
        master = self.content
        
        # Metadata category labels
        user_text = "Place IMUs in the following blocks and select their IDs."
        instructions = tk.Label(master, text=user_text)
        instructions.grid(sticky=tk.W, row=0, columnspan=3)
    
        # Set up widgets and define layoutdev_id.get()
        self.dev_ids = {}
        menus = []
        buttons = []
        commands = []
        for i, block in enumerate(self.active_blocks):
            
            id_label = tk.Label(master, text='{} rectangle'.format(block))
            id_label.grid(sticky=tk.E, row=i+1, column=0)
            
            self.dev_ids[block] = tk.StringVar(master)
            self.dev_ids[block].set(self.corpus.imu_ids[i])
            
            menus.append(apply(tk.OptionMenu,
                               (master, self.dev_ids[block]) + self.corpus.imu_ids))
            menus[-1].grid(sticky=tk.W, row=i+1, column=1)
            
            commands.append(lambda b=str(block): self.connectionAttemptDialog(b))
            buttons.append(tk.Button(master, text='Connect', command=commands[-1]))
            buttons[-1].grid(sticky=tk.W, row=i+1, column=2)
            
        submit = tk.Button(master, text='Next', command=self.forward,
                           default=tk.ACTIVE)
        submit.grid(sticky=tk.E, row=len(self.dev_ids)+1, column=2)
        
        back = tk.Button(master, text='Back', command=self.back)
        back.grid(sticky=tk.W, row=len(self.dev_ids)+1, column=0)
        
        master.place(relx=0.5, rely=0.5, anchor='center')
    
    
    def drawStreamContent(self):
        """
        Draw the data streaming window.
        """
        
        # TODO: 
        
        master = self.content
        
        l = tk.Label(master, text='Streaming data...')
        l.grid(row=0, column=0, columnspan=2)
        
        q = tk.Button(master, text='Quit', command=self.closeStream)
        q.grid(row=1, column=1)
        
        p = tk.Button(master, text='Pause', command=self.stopStream)
        p.grid(row=1, column=0)
        
        master.place(relx=0.5, rely=0.5, anchor='center')
        
        for p in self.streamProcesses:
            p.start()
    
    
    def connectionAttemptDialog(self, block):
        """
        For a given block, check whether the specified block is already in use.
        If it is, draw a popup window prompting the user to select a different
        device. If it isn't, try to connect to the device.
        
        Args:
        -----
        [str] block: Color of the rectangular block housing the IMU
        """
        
        if not self.popup is None:
            return
        
        imu_id = self.dev_ids[block].get()
        
        self.popup = tk.Toplevel(self.parent)
        
        if imu_id in self.connected_devices.keys():
            fmtstr = 'Device {} is already in use! Choose a different device.'
            l = tk.Label(self.popup, text=fmtstr.format(imu_id))
            l.pack()
            
            ok = tk.Button(self.popup, text='OK', command=self.cancel)
            ok.pack()
        else:
            fmtstr = 'Connecting to {}...'
            l = tk.Label(self.popup, text=fmtstr.format(imu_id))
            l.pack()
            
            # Schedule connection attempt
            func = lambda: self.attemptConnection(imu_id)
            self.parent.after(0, func)
    
    
    def attemptConnection(self, imu_id):
        """
        Try to connect to the specified IMU.
        
        Args:
        -----
        [str] imu_id: 4-digit hex ID of the IMU, as a string
        """
        
        mac_prefix = ['00', '17', 'E9', 'D7']
        imu_address = ':'.join(mac_prefix + [imu_id[0:2], imu_id[2:4]])
        socket, name = wax9.connect(imu_address)
        if name is None:
            block = self.imu2block[imu_id]
            self.connectionFailureDialog(block)
        else:
            self.connected_devices[imu_id] = socket
            
            # Read off IMU settings for each device. If I skip this before
            # sending the stream command, the devices stall. I think it's
            # because of some kind of firmware or hardware instability, but I'm
            # still not sure.
            settings = wax9.getSettings(socket)
            parsed_settings = wax9.parseSettings(settings)
            # TODO: Correct settings if they aren't what we expect
            self.imu_settings.append(parsed_settings)
            self.connectionSuccessDialog(name)
        
    
    def connectionFailureDialog(self, block):
        """
        Draw a popup window informing the user that the connection attempt has
        failed.
        
        Args:
        -----
        [str] block: Color of the block housing the IMU which failed to connect
        """
        
        self.popup.destroy()
        self.popup = tk.Toplevel(self.parent)
        
        fmtstr = 'Connection attempt failed! Try again?'
        l = tk.Label(self.popup, text=fmtstr)
        l.grid(row=0, columnspan=2)
        
        func = lambda b=str(block): self.connectionAttemptDialog(b)
        y = tk.Button(self.popup, text='Yes', command=func)
        n = tk.Button(self.popup, text='No', command=self.cancel)
        
        y.grid(row=1, column=0)
        n.grid(row=1, column=1)
    
    
    def connectionSuccessDialog(self, imu_id):
        """
        Draw a popup window informing the user that the connection attempt was
        successful.
        
        Args:
        -----
        [imu_id]: 4-digit hex ID of the IMU, as a string
        """
        
        self.popup.destroy()
        self.popup = tk.Toplevel(self.parent)
        
        fmtstr = 'Successfully connected to {}!'
        l = tk.Label(self.popup, text=fmtstr.format(imu_id))
        l.pack()
        
        ok = tk.Button(self.popup, text='OK', command=self.cancel)
        ok.pack()
    
    
    def cancel(self):
        """
        Close the current popup window.
        """
        
        self.popup.destroy()
        self.popup = None
    
    
    def back(self):
        """
        Go back to the previous stage of data collection.
        """
        
        if self.cur_position > 0:
            self.cur_position -= 1
        
        self.content.destroy()
        self.content = self.defaultFrame()
        
        cur_frame = self.controlflow[self.cur_position]
        cur_frame()

    
    def forward(self):
        """
        Go to the next stage of data collection.
        """
        
        getter = self.getters[self.cur_position]
        getter()
        
        if self.cur_position < len(self.controlflow) - 1:
            self.cur_position += 1
        
        self.content.destroy()
        self.content = self.defaultFrame()
        
        cur_frame = self.controlflow[self.cur_position]
        cur_frame()
    
    
    def closeStream(self):
        """
        Stop data collection, write metadata to file, and quit GUI.
        """
        
        if not self.die.is_set():
            self.stopStream()
        
        for name, socket in self.connected_devices.items():
            print('Disconnecting from {}...'.format(name))
            socket.close()
        
        self.corpus.postprocess(self.trial_id, self.trial_metadata,
                                self.imu_settings, self.imu2block)
        self.corpus.makeImuFigs(self.trial_id)
        
        self.parent.destroy()
    
    
    def stopStream(self):
        """
        Stop streaming from IMUs and camera.
        """
        
        self.die.set()
        for p in self.streamProcesses:
            p.join()
    
    
    def getMetaData(self):
        """
        Read metadata from tkinter widgets.
        """
        
        self.trial_metadata = (self.participant_num.get(), self.dob_month.get(),
                               self.dob_year.get(), self.gender.get())
        
        print('Participant ID: {}'.format(self.trial_metadata[0]))
        print('Birth date: {} / {}'.format(self.trial_metadata[1],
                                           self.trial_metadata[2]))
        print('Gender: {}'.format(self.trial_metadata[3]))
    
    
    def getTaskData(self):
        """
        Read the block construction task from tkinter widgets.
        """
        
        task = self.task.get()
        print('Task: {}'.format(task))
        
        if task == 1:
            self.active_blocks = ('red', 'yellow')
        elif task == 2:
            self.active_blocks = ('green', 'blue')
        elif task in (3, 4, 5, 6):
            self.active_blocks = ('red', 'yellow', 'green', 'blue')
        
    
    def getImuData(self):
        """
        Read block to IMU mappings from tkinter widgets.
        """
        
        # FIXME: this converts imu_settings from a list to a numpy array (bad)
        self.imu_settings = np.hstack(tuple(self.imu_settings))
        
        for block, dev_id in self.dev_ids.items():
            imu = dev_id.get()
            self.imu2block[imu] = block
            print('{}: {}'.format(block, self.imu2block[imu]))
    

if __name__ == '__main__':
    
    root = tk.Tk()
    app = Application(root)
    
    root.mainloop()
    
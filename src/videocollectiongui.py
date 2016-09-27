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
import glob

import numpy as np

import libwax9 as wax9
import libprimesense as ps
from duplocorpus import DuploCorpus


class Application:
    
    def __init__(self, parent):
        """
        Args:
        -----
        parent:  tk window
          Parent tkinter object
        """
        
        # Define parent window and resize
        self.parent = parent
        
        # Window content master = self.navigation_frame
        self.content_frame = tk.Frame(self.parent)
        self.navigation_frame = tk.Frame(self.parent)
        self.popup = None
        
        # These structures define how the data collection screens progress
        self.interface_index = 0
        self.interfaces = (self.defineInfoInterface, self.defineTaskInterface,
                           self.defineStreamInterface)
        self.getters = (self.getMetaData, self.getTaskData, self.getStreamData)
        
        # Define some constants
        # NOTE: The trial id will increase after a call to corpus.postprocess!!
        #   (which happens in chooseNewTask and closeStream)
        self.corpus = DuploCorpus()
        self.trial_id = self.corpus.meta_data.shape[0]
        self.blocks = ('red', 'yellow', 'green', 'blue')
        # 1: ./img/4-1.png -- 2: ./img/4-2.png
        # 3: ./img/6-1.png -- 4: ./img/6-2.png
        # 5: ./img/8-1.png -- 6: ./img/8-2.png
        self.task2block = {1: self.blocks[:2], 2: self.blocks[0:1] + self.blocks[2:],
                           3: self.blocks, 4: self.blocks[0:2] + self.blocks[3:4],
                           5: self.blocks, 6: self.blocks}
        self.imu2block = {imu: 'UNUSED' for imu in self.corpus.imu_ids}
        
        #===[DATA FIELDS]======================================================
        self.participant_id_field = None
        self.birth_month_field = None
        self.birth_year_field = None
        self.gender_field = None
        
        self.task_field = None
        
        self.block2imu_id_field = {}
        
        #===[DATA GATHERED FROM FIELDS]========================================
        self.participant_id = None
        self.birth_month = None
        self.birth_year = None
        self.gender = None
        
        self.task = None
        self.active_blocks = None
        
        self.block2button = {}
        self.block2imu_nickname = {}
        self.imu_id2socket = {}
        self.imu_settings = []
        
        # For streaming in parallel: die tells all streaming processes to quit,
        # q is used to communicate between rgb stream and main process
        self.die = mp.Event()
        self.die_set_by_user = False
        self.video_q = mp.Queue()
        self.imu_q = mp.Queue()
                 
        # Start drawing interfaces
        interface = self.interfaces[self.interface_index]
        interface()
        self.drawInterface()
    
    
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
    
    
    def defineInfoInterface(self):
        """
        Set up the metadata collection interface.
        """
        
        master = self.content_frame
        
        # Draw instructions
        user_text = 'Please enter the following information for this participant.'
        instructions = tk.Label(master, text=user_text)
        instructions.grid(row=0, columnspan=3)
        
        # Draw participant ID field
        participant_id_label = tk.Label(master, text='Participant ID: ')
        participant_id_label.grid(sticky=tk.E, row=1, column=0)
        self.participant_id_field = tk.Entry(master)
        self.participant_id_field.grid(sticky=tk.W, row=1, column=1, columnspan=2)
        
        # Draw birth month field
        birth_date_label = tk.Label(master, text='Date of birth: ')
        birth_date_label.grid(sticky=tk.E, row=2, column=0)
        months = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                  'Sep', 'Oct', 'Nov', 'Dec', '---')
        self.birth_month_field = tk.StringVar(master)
        if not self.birth_month:
            self.birth_month_field.set(months[-1])
        else:
            self.birth_month_field.set(self.birth_month)
        month_menu = apply(tk.OptionMenu, (master, self.birth_month_field) + months)
        month_menu.grid(sticky=tk.W, row=2, column=1)
        
        # Draw birth year field
        years = tuple(range(1980, 2000))
        self.birth_year_field = tk.StringVar(master)
        if not self.birth_year:
            self.birth_year_field.set(years[0])
        else:
            self.birth_year_field.set(self.birth_year)
        year_menu = apply(tk.OptionMenu, (master, self.birth_year_field) + years)
        year_menu.grid(sticky=tk.W, row=2, column=2)
        
        # Draw gender field
        gender_label = tk.Label(master, text='Gender: ')
        gender_label.grid(sticky=tk.E, row=3, column=0)
        genders = ('Male', 'Female', '---')
        self.gender_field = tk.StringVar(master)
        if not self.gender:
            self.gender_field.set(genders[-1])
        else:
            self.gender_field.set(self.gender)
        gender_menu = apply(tk.OptionMenu, (master, self.gender_field) + genders)
        gender_menu.grid(sticky=tk.W, row=3, column=1)
            
        # Draw navigation buttons
        master = self.navigation_frame
        forward = tk.Button(master, text='Choose task >>', command=self.forward,
                            default=tk.ACTIVE)
        forward.grid(sticky=tk.E, row=0, column=1)
    
    
    def defineTaskInterface(self):
        """
        Set up the block task selection interface.
        """
        
        master = self.content_frame
        
        # Draw instructions
        user_text = "Select the current block construction."
        instructions = tk.Label(master, text=user_text)
        instructions.grid(row=0, columnspan=3)
        
        # Draw block construction choices
        self.task_field = tk.IntVar()
        block_image_fns = sorted(glob.glob(os.path.join('img', '*.png')))
        for i, filename in enumerate(block_image_fns):
            # filename format is [4,6,8]block-[1,2].png
            name = os.path.splitext(os.path.basename(filename))[0]
            button_row = int(name[-1])
            button_column = int(name[0]) / 2 - 2
            block_image = ImageTk.PhotoImage(Image.open(filename))
            b = tk.Radiobutton(master, image=block_image,
                               variable=self.task_field, value=i+1)
            b.image = block_image
            b.grid(row=button_row, column=button_column)
        
        # Draw navigaton buttons
        master = self.navigation_frame
        forward = tk.Button(master, text="Collect data >>", command=self.forward,
                           default=tk.ACTIVE)
        forward.grid(sticky=tk.E, row=0, column=1)
        back = tk.Button(master, text="<< Enter data", command=self.back)
        back.grid(sticky=tk.W, row=0, column=0)
                    

    def defineStreamInterface(self):
        """
        Set up the data streaming interface.
        """
        
        # Start streaming data
        self.startStreamProcesses()
        
        master = self.content_frame
        
        # Draw placeholder for video monitor
        self.rgb_video = tk.Label(master, text='Waiting for video...')
        self.rgb_video.grid(row=1, column=0)        

        self.video_label = tk.Label(master, text='Video stream')        
        self.video_label.grid(row=0, column=0, sticky='N')

        # Draw navigation buttons
        master = self.navigation_frame
        q = tk.Button(master, text='Quit', command=self.closeStream)
        q.grid(sticky=tk.E, row=0, column=1)
        p = tk.Button(master, text='New task', command=self.chooseNewTask)
        p.grid(sticky=tk.W, row=0, column=0)
                
        self.parent.after(75, self.refreshStreamInterface)
    
       
    def startStreamProcesses(self):
        """
        Define and start running process that streams from cameras.
        """
        
        # Define paths used for file I/O when streaming data
        frame_base_path = os.path.join(self.corpus.paths['video-frames'],
                                       str(self.trial_id))
        timestamp_fn = '{}-timestamps.csv'.format(self.trial_id)
        timestamp_path = os.path.join(self.corpus.paths['raw'], timestamp_fn)
        
        # Define process that streamd from camera
        videostream_args = (frame_base_path, timestamp_path,
                            self.corpus.image_types, self.die, self.video_q)
        self.processes = (mp.Process(target=ps.stream, args=videostream_args),)
        
        for p in self.processes:
            p.start()
    
 
    def refreshStreamInterface(self):
        """
        Update frame on the RGB video monitor.
        """
        
        # Draw a new frame if one has been sent by the video stream
        if not self.video_q.empty():
            newest_frame_path = self.video_q.get()
            newest_frame = ImageTk.PhotoImage(Image.open(newest_frame_path))
            self.rgb_video.configure(image=newest_frame)
            self.rgb_video.image = newest_frame
        
        self.parent.after(75, self.refreshStreamInterface)
    
    
    def chooseNewTask(self):
        """
        Stop streaming, write data, and go back to the task selection
        interface.
        """
        
        if not self.die.is_set():
            self.stopStream()
        
        # No data were collected from blocks, so map all blocks that to the
        # string, 'UNUSED'.
        block_mapping = {imu_id: 'UNUSED' for imu_id in self.corpus.imu_ids}

        
        # Update the metadata array and increment the trial index
        metadata = (self.participant_id, self.birth_month, self.birth_year,
                    self.gender, self.task)
        
        frame_timestamps = self.corpus.parseVideoData(self.trial_id)
        self.corpus.writeFrameTimestamps(self.trial_id, frame_timestamps)
        self.corpus.updateMetaData(self.trial_id, metadata, block_mapping)
        
        self.trial_id = self.corpus.meta_data.shape[0]
        
        # Reset die so we don't immediately quit streaming data in the next
        # round
        self.die.clear()
        
        # task selection interface is at position 1
        self.clearInterface()
        self.interface_index = 1
        interface = self.interfaces[self.interface_index]
        interface()
        self.drawInterface()
    
    
    def cancel(self):
        """
        Close the current popup window.
        """
        
        self.popup.destroy()
        self.popup = None
    
    
    def back(self):
        """
        Go back to the previous interface.
        """
        
        # Do nothing if there's an active popup window
        if not self.popup is None:
            return
        
        self.interface_index = max(self.interface_index - 1, 0)
        
        self.clearInterface()
        interface = self.interfaces[self.interface_index]
        interface()
        self.drawInterface()

    
    def forward(self):
        """
        Get data, check if it's valid, and go to the next interface if it is.
        """
        
        # Do nothing if there's an active popup window
        if not self.popup is None:
            return
        
        # Get data from the current interface and validate
        getter = self.getters[self.interface_index]
        error_string = getter()
        
        # Alert user if data didn't pass validation, otherwise go to next
        # interface
        if error_string:
            self.badInputDialog(error_string)
        else:
            self.interface_index = min(self.interface_index + 1,
                                       len(self.interfaces) - 1)
            
            self.clearInterface()
            interface = self.interfaces[self.interface_index]
            interface()
            self.drawInterface()
    
    
    def badInputDialog(self, error_string):
        """
        Alert user of malformed input
        
        Args:
        -----
        error_string:  str
          Message that will be displayed in popup window
        """
        
        # It shouldn't be possible to get here if a popup window is active
        assert(self.popup is None)
        
        self.popup = tk.Toplevel(self.parent)
        
        l = tk.Label(self.popup, text=error_string)
        l.grid(row=0)
        
        ok = tk.Button(self.popup, text='OK', command=self.cancel)
        ok.grid(row=1)
    
    
    def closeStream(self):
        """
        Stop data collection, write metadata to file, and quit GUI.
        """
        
        if not self.die.is_set():
            self.stopStream()
        
        # No data were collected from blocks, so map all blocks that to the
        # string, 'UNUSED'.
        block_mapping = {imu_id: 'UNUSED' for imu_id in self.corpus.imu_ids}

        
        # Update the metadata array and increment the trial index
        metadata = (self.participant_id, self.birth_month, self.birth_year,
                    self.gender, self.task)
        
        frame_timestamps = self.corpus.parseVideoData(self.trial_id)
        self.corpus.writeFrameTimestamps(self.trial_id, frame_timestamps)
        self.corpus.updateMetaData(self.trial_id, metadata, block_mapping)
        
        self.parent.destroy()
    
    
    def stopStream(self):
        """
        Stop streaming from IMUs and camera.
        """
        
        self.die_set_by_user = True
        self.die.set()
        for p in self.processes:
            p.join()
    
    
    def getMetaData(self):
        """
        Read metadata from tkinter widgets.
        
        Returns:
        --------
        error_string:  str
          Data validation error message. Empty string if user input passes
          validation.
        """
        
        self.participant_id = self.participant_id_field.get()
        self.birth_month = self.birth_month_field.get()
        self.birth_year = self.birth_year_field.get()
        self.gender = self.gender_field.get()
        
        if not self.participant_id:
            return 'Please enter a participant ID.'
        
        print('Participant ID: {}'.format(self.participant_id))
        print('Birth date: {} / {}'.format(self.birth_month, self.birth_year))
        print('Gender: {}'.format(self.gender))
        
        return ''
    
    
    def getTaskData(self):
        """
        Read the block construction task from tkinter widgets.
        
        Returns:
        --------
        error_string:  str
          Data validation error message. Empty string if user input passes
          validation.
        """
        
        self.task = self.task_field.get()
        if not self.task:
            return 'Please make a selection.'
        
        self.active_blocks = self.task2block[self.task]
        
        print('Task: {}'.format(self.task))
        print('Active blocks: {}'.format(self.active_blocks))
        
        return ''
        
    
    def getStreamData(self):
        # TODO:
        print("I am a dummy function!")
    

if __name__ == '__main__':
    
    # Start the application in quasi-fullscreen mode
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry('{0}x{1}+0+0'.format(screen_width, screen_height))
    #root.geometry('800x600')
    
    app = Application(root)
    
    root.mainloop()
    
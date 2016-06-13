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
        [tk window] parent: Parent tkinter object
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
                           self.defineImuInterface, self.defineStreamInterface)
        self.getters = (self.getMetaData, self.getTaskData, self.getImuData,
                        self.getStreamData)
        
        # Define some constants        
        self.corpus = DuploCorpus()
        blocks = ('red', 'yellow', 'green', 'blue')
        self.task2block = {1: blocks[:2], 2: blocks[2:], 3: blocks,
                           4:blocks, 5: blocks, 6: blocks}
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
        self.block2imu_id = {}
        self.imu_id2socket = {}
        self.imu_settings = []
        
        # For streaming in parallel: die tells all streaming processes to quit,
        # q is used to communicate between rgb stream and main process
        self.die = mp.Event()
        self.die_set_by_user = False
        self.video_q = mp.Queue()
        self.imu_q = mp.Queue()
        
        self.defineStreamProcesses()
         
        # Start drawing interfaces
        interface = self.interfaces[self.interface_index]
        interface()
        self.drawInterface()
    
    
    def defineStreamProcesses(self):
        """
        """
        
        # Define paths used for file I/O when streaming data
        # NOTE: The trial id will increase after a call to corpus.postprocess!!!
        #   (which happens in chooseNewTask and closeStream)
        self.trial_id = self.corpus.meta_data.shape[0]
        raw_imu_fn = '{}-imu.csv'.format(self.trial_id)
        raw_imu_path = os.path.join(self.corpus.paths['raw'], raw_imu_fn)
        frame_base_path = os.path.join(self.corpus.paths['video-frames'],
                                       str(self.trial_id))
        timestamp_fn = '{}-timestamps.csv'.format(self.trial_id)
        timestamp_path = os.path.join(self.corpus.paths['raw'], timestamp_fn)
        
        # Define processes that stream from IMUs and camera
        videostream_args = (frame_base_path, timestamp_path,
                            self.corpus.image_types, self.die, self.video_q)
        imustream_args = (self.imu_id2socket, raw_imu_path, self.die, self.imu_q)
        self.processes = (mp.Process(target=ps.stream, args=videostream_args),
                          mp.Process(target=wax9.stream, args=imustream_args),)
    
    
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
        Draw the metadata collection window.
        """
        
        master = self.content_frame
        
        # Draw instructions
        user_text = 'Please enter the following information for this child.'
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
                  'Sep', 'Oct', 'Nov', 'Dec')
        self.birth_month_field = tk.StringVar(master)
        if not self.birth_month:
            self.birth_month_field.set(months[0])
        else:
            self.birth_month_field.set(self.birth_month)
        month_menu = apply(tk.OptionMenu, (master, self.birth_month_field) + months)
        month_menu.grid(sticky=tk.W, row=2, column=1)
        
        # Draw birth year field
        years = (2010, 2011, 2012, 2013)
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
        genders = ('Male', 'Female', 'Not disclosed')
        self.gender_field = tk.StringVar(master)
        if not self.gender:
            self.gender_field.set(genders[0])
        else:
            self.gender_field.set(self.gender)
        gender_menu = apply(tk.OptionMenu, (master, self.gender_field) + genders)
        gender_menu.grid(sticky=tk.W, row=3, column=1)
            
        # Draw navigation buttons
        master = self.navigation_frame
        forward = tk.Button(master, text='Next', command=self.forward,
                            default=tk.ACTIVE)
        forward.grid(sticky=tk.E, row=0, column=1)
    
    
    def defineTaskInterface(self):
        """
        Draw the block task selection window.
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
        forward = tk.Button(master, text="Next", command=self.forward,
                           default=tk.ACTIVE)
        forward.grid(sticky=tk.E, row=0, column=1)
        back = tk.Button(master, text="Back", command=self.back)
        back.grid(sticky=tk.W, row=0, column=0)
                

    def defineImuInterface(self):
        """
        Draw the IMU connection window.
        """
        
        master = self.content_frame
        
        # Draw instructions
        user_text = "Place IMUs in the following blocks and select their IDs."
        instructions = tk.Label(master, text=user_text)
        instructions.grid(sticky=tk.W, row=0, columnspan=3)
    
        # Draw IMU-block connection interface
        for i, block in enumerate(self.active_blocks):
            
            id_label = tk.Label(master, text=' \t ', background=block)
            id_label.grid(row=i+1, column=0)
            
            # Draw different buttons depending on whether the block is
            # associated with a connected IMU or not
            self.block2imu_id_field[block] = tk.StringVar(master)
            if block in self.block2imu_id:
                imu = self.block2imu_id[block]
                self.block2imu_id_field[block].set(imu)
                button_text = '[Connected]'
                func = lambda b=str(block): self.connectionAttemptDialog(b)
            else:
                self.block2imu_id_field[block].set(self.corpus.imu_ids[i])
                button_text = 'Connect'
                func = lambda b=str(block): self.connectionAttemptDialog(b)
            
            menu_args = (master, self.block2imu_id_field[block]) \
                      + self.corpus.imu_ids
            imu_menu = apply(tk.OptionMenu, menu_args)
            imu_menu.grid(sticky=tk.W, row=i+1, column=1)
            
            func = lambda b=str(block): self.connectionAttemptDialog(b)
            button = tk.Button(master, text=button_text, command=func)
            button.grid(sticky=tk.W, row=i+1, column=2)
            self.block2button[block] = button
        
        # Draw navigation buttons
        master = self.navigation_frame
        forward = tk.Button(master, text='Collect data', command=self.forward,
                           default=tk.ACTIVE)
        forward.grid(sticky=tk.E, row=0, column=1)
        back = tk.Button(master, text='Back', command=self.back)
        back.grid(sticky=tk.W, row=0, column=0)
    
    
    def defineStreamInterface(self):
        """
        Draw the data streaming window.
        """
        
        # Start streaming data
        for p in self.processes:
            p.start()
        
        master = self.content_frame
        
        # Draw placeholder for video monitor
        self.rgb_video = tk.Label(master, text='Waiting for video...')
        self.rgb_video.grid(row=0, column=0)
        
        l = tk.Label(master, text='Streaming data...')
        l.grid(row=1, column=0)
        
        # Draw placeholders for IMU monitors
        self.imu_id2activity_color = {}
        imu_monitor_frame = tk.Frame(master)
        for i, imu_id in enumerate(self.imu_id2socket.keys()):
            # Label text
            block_color = self.imu2block[imu_id]
            color_label = tk.Label(imu_monitor_frame, text=' \t ',
                                   background=block_color)
            color_label.grid(row=i, column=0)
            id_label = tk.Label(imu_monitor_frame, text=' active: ')
            id_label.grid(row=i, column=1)
            # Activity indicator
            self.imu_id2activity_color[imu_id] = tk.Label(imu_monitor_frame,
                                                          text='    ',
                                                          background='yellow')
            self.imu_id2activity_color[imu_id].grid(row=i, column=2)
        imu_monitor_frame.grid(row=2, column=0)
            

        # Draw navigation buttons
        master = self.navigation_frame
        q = tk.Button(master, text='Quit', command=self.closeStream)
        q.grid(sticky=tk.E, row=0, column=1)
        p = tk.Button(master, text='New task', command=self.chooseNewTask)
        p.grid(sticky=tk.W, row=0, column=0)
                
        self.parent.after(75, self.refreshStreamInterface)
    
    
    def refreshStreamInterface(self):
        """
        Update frame on the RGB video monitor and check to make sure none of
        the IMUs has died.
        """
        
        # Check if an IMU has died and notify the user if it has
        if self.die.is_set() and not self.die_set_by_user:
            self.imuFailureDialog()
            return
        
        if not self.imu_q.empty():
            samples = self.imu_q.get()
            for sample in samples:
                imu_id = sample[-1]
                # Calculate (unitless) l1 norm of acceleration
                # (4096 bits in 1 g --> why we use 5000 as the threshold)
                accel_mag = abs(sample[4]) + abs(sample[5]) + abs(sample[6])
                bg_color = 'green' if accel_mag > 4950 else 'yellow'
                self.imu_id2activity_color[imu_id].configure(background=bg_color)
        
        # Draw a new frame if one has been sent by the video stream
        if not self.video_q.empty():
            newest_frame_path = self.video_q.get()
            newest_frame = ImageTk.PhotoImage(Image.open(newest_frame_path))
            self.rgb_video.configure(image=newest_frame)
            self.rgb_video.image = newest_frame
        
        self.parent.after(75, self.refreshStreamInterface)
    
    
    def chooseNewTask(self):
        """
        Stop streaming, write data,  and go back to the task selection
        interface.
        """
        
        if not self.die.is_set():
            self.stopStream()
        
        metadata = (self.participant_id, self.birth_month, self.birth_year,
                    self.gender, self.task)
        imu_settings_array = np.hstack(tuple(self.imu_settings))
        self.corpus.postprocess(self.trial_id, metadata, self.imu2block,
                                imu_settings_array)
        
        # Reset die so we don't immediately quit streaming data in the next
        # round
        self.die.clear()
        
        # Re-define streaming processes
        self.defineStreamProcesses()
        
        # task selection interface is at position 1
        self.clearInterface()
        self.interface_index = 1
        interface = self.interfaces[self.interface_index]
        interface()
        self.drawInterface()
    
    
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
        
        imu_id = self.block2imu_id_field[block].get()
        
        self.popup = tk.Toplevel(self.parent)
        
        if imu_id in self.imu_id2socket:
            fmtstr = 'Device {} is already in use! Choose a different device.'
            l = tk.Label(self.popup, text=fmtstr.format(imu_id))
            l.pack()
            
            ok = tk.Button(self.popup, text='OK', command=self.cancel)
            ok.pack()
        elif block in self.block2imu_id:
            block_dev = self.block2imu_id[block]
            fmtstr = 'This block is already associated with device {}!'
            l = tk.Label(self.popup, text=fmtstr.format(block_dev))
            l.pack()
            
            ok = tk.Button(self.popup, text='OK', command=self.cancel)
            ok.pack()
        else:
            fmtstr = 'Connecting to {}...'
            l = tk.Label(self.popup, text=fmtstr.format(imu_id))
            l.pack()
            
            self.attemptConnection(imu_id, block)
    
    
    def attemptConnection(self, imu_id, block):
        """
        Try to connect to the specified IMU.
        
        Args:
        -----
        [str] imu_id: 4-digit hex ID of the IMU
        [str] block: Color of the current block
        """
        
        mac_prefix = ['00', '17', 'E9', 'D7']
        imu_address = ':'.join(mac_prefix + [imu_id[0:2], imu_id[2:4]])
        socket, name = wax9.connect(imu_address)
        if name is None:
            self.connectionFailureDialog()
        else:
            self.imu_id2socket[imu_id] = socket
            self.block2imu_id[block] = imu_id
            self.imu2block[imu_id] = block
            print('{}: {}'.format(block, imu_id))
            
            # Read off IMU settings for each device. If I skip this before
            # sending the stream command, the devices stall. I think it's
            # because of some kind of firmware or hardware instability, but I'm
            # still not sure.
            settings = wax9.getSettings(socket)
            parsed_settings = wax9.parseSettings(settings)
            # TODO: Correct settings if they aren't what we expect
            self.imu_settings.append(parsed_settings)
            
            # Update 'connect' button
            self.block2button[block].configure(text='[Connected]')
            
            sample_str = wax9.sample(socket)
            data_str = sample_str.strip().split('\r\n')[1]
            battery = int(data_str.split(',')[-4])  # Battery voltage in mV
            self.connectionSuccessDialog(name, battery)
        
    
    def connectionFailureDialog(self):
        """
        Draw a popup window informing the user that the connection attempt has
        failed.
        """
        
        self.popup.destroy()
        self.popup = tk.Toplevel(self.parent)
        
        fmtstr = 'Connection attempt failed! Cycle the device and try again.'
        l = tk.Label(self.popup, text=fmtstr)
        l.grid(row=0)
        
        ok = tk.Button(self.popup, text='OK', command=self.cancel)
        ok.grid(row=1)
    
    
    def connectionSuccessDialog(self, imu_id, battery):
        """
        Draw a popup window informing the user that the connection attempt was
        successful.
        
        Args:
        -----
        [str] imu_id: 4-digit hex ID of the IMU, as a string
        [int] battery: IMU battery voltage in mV
        """
        
        self.popup.destroy()
        self.popup = tk.Toplevel(self.parent)
        
        # Battery discharge curve has a sharp knee around 3300 mV, so take that
        # as zero. Max charge is about 4200 mV.
        # (see WAX9 developer's guide, p. 17)
        min_charge = 3300
        max_charge = 4200        
        charge_percent = float(battery - min_charge) / (max_charge - min_charge)
        # Battery can't be more than 100% charged, but it could look that way
        # because the max capacity is only approximate
        charge_percent = int(min(charge_percent, 1) * 100)
        
        fmtstr = '\nSuccessfully connected to {}!\n\nBattery: {}%\n'
        l = tk.Label(self.popup, text=fmtstr.format(imu_id, charge_percent))
        l.pack()
        
        ok = tk.Button(self.popup, text='OK', command=self.cancel)
        ok.pack()
    
    
    def imuFailureDialog(self):
        """
        Warn the user that one or more IMUs has failed.
        """
        
        # die was set, so the data streaming processes have stopped
        for p in self.processes:
            p.join()
        
        self.popup = tk.Toplevel(self.parent)
        
        error_string = """A device is unresponsive. Data collection halted.
                          Quit this program, cycle the devices, and restart
                          (the data from this trial will be saved)."""
        l = tk.Label(self.popup, text=error_string)
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
        [str] error_string: Message that will be displayed in popup window
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
        
        for name, socket in self.imu_id2socket.items():
            print('Disconnecting from {}...'.format(name))
            socket.close()
        
        metadata = (self.participant_id, self.birth_month, self.birth_year,
                    self.gender, self.task)
        imu_settings_array = np.hstack(tuple(self.imu_settings))
        self.corpus.postprocess(self.trial_id, metadata, self.imu2block,
                                imu_settings_array)
        #self.corpus.makeImuFigs(self.trial_id)
        
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
        [str] error_string: Data validation error message. Empty string if
          user input passes validation.
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
        [str] error_string: Data validation error message. Empty string if
          user input passes validation.
        """
        
        self.task = self.task_field.get()
        if not self.task:
            return 'Please make a selection.'
        
        self.active_blocks = self.task2block[self.task]
        
        print('Task: {}'.format(self.task))
        print('Active blocks: {}'.format(self.active_blocks))
        
        return ''
        
    
    def getImuData(self):
        """
        Read block to IMU mappings from tkinter widgets.
        """
                
        # Make sure there is a connected IMU associated with every block
        for block in self.active_blocks:
            if not block in self.block2imu_id:
                fmtstr = 'Please put an IMU in the {} block and click connect.'
                return fmtstr.format(block)
        
        return ''
            
    
    def getStreamData(self):
        # TODO:
        print("I am a dummy function!")
    

if __name__ == '__main__':
    
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry('{0}x{1}+0+0'.format(screen_width, screen_height))
    #root.geometry('800x600')
    
    app = Application(root)
    
    root.mainloop()
    
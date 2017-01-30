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
import time

import numpy as np

#import libwax9 as wax9
import libprimesense as ps
from duplocorpus import DuploCorpus

from pymetawear.exceptions import PyMetaWearConnectionTimeout
from pymetawear_logging import MetawearDevice


class Application:
    
    def __init__(self, parent):
        """
        Parameters
        ----------
        parent:  tk window
          Parent tkinter object
        """
        
        # Define parent window and resize
        self.parent = parent
        
        # Window content
        self.content_frame = tk.Frame(self.parent)
        self.navigation_frame = tk.Frame(self.parent)
        self.popup = None
        
        # These structures define how the data collection screens progress
        self.interface_index = 0
        self.interfaces = (self.defineInfoInterface, self.defineImuInterface,
                           self.defineTaskInterface, self.defineStreamInterface)
        self.getters = (self.getMetaData, self.getImuData, self.getTaskData,
                        self.getStreamData)
                    
        self.bt_interfaces = ('hci0', 'hci1', 'hci2', 'hci3')
        self.bt_interface_index = 0
        
        # Define some constants
        # NOTE: The trial id will increase after a call to corpus.postprocess!!
        #   (which happens in chooseNewTask and closeStream)
        self.corpus = DuploCorpus()
        self.trial_id = self.corpus.meta_data.shape[0]
        self.blocks = ('red square', 'yellow square', 'green square', 'blue square',
                       'red rect', 'yellow rect', 'green rect', 'blue rect')
        # 1: ./img/4-1.png -- 2: ./img/4-2.png
        # 3: ./img/6-1.png -- 4: ./img/6-2.png
        # 5: ./img/8-1.png -- 6: ./img/8-2.png
        self.task2block = {1: self.blocks[2:4] + self.blocks[4:6],
                           2: self.blocks[1:2] + self.blocks[4:5] + self.blocks[6:],
                           3: self.blocks[2:4] + self.blocks[4:],
                           4: self.blocks[1:4] + self.blocks[4:6] + self.blocks[7:],
                           5: self.blocks,
                           6: self.blocks}
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
        self.imu_id2dev = {}
        
        # For streaming in parallel: die tells all streaming processes to quit,
        # q is used to communicate between rgb stream and main process
        self.die = mp.Event()
        self.die_set_by_user = False
        self.video_q = mp.Queue()
                 
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
        forward = tk.Button(master, text='Connect IMUs >>', command=self.forward,
                            default=tk.ACTIVE)
        forward.grid(sticky=tk.E, row=0, column=1)
    
    
    def defineImuInterface(self):
        """
        Set up the IMU connection interface.
        """
        
        master = self.content_frame
        
        # Draw instructions
        user_text = "Place IMUs in the following blocks and select their IDs."
        instructions = tk.Label(master, text=user_text)
        instructions.grid(sticky=tk.W, row=0, columnspan=3)
    
        # Draw IMU-block connection interface
        imu_nicknames = tuple(sorted(self.corpus.nickname2id.keys()))
        for i, block in enumerate(self.blocks):
            
            color, shape = block.split()
            
            shape_text = '            ' if shape == 'rect' else '    '
            id_label = tk.Label(master, text=shape_text, background=color)
            id_label.grid(row=i+1, column=0)
            
            # Draw different buttons depending on whether the block is
            # associated with a connected IMU or not
            self.block2imu_id_field[block] = tk.StringVar(master)
            if block in self.block2imu_nickname:
                nickname = self.block2imu_nickname[block]
                self.block2imu_id_field[block].set(nickname)
                button_text = 'Disconnect'
                func = lambda b=str(block): self.resetConnection(b)
            else:
                self.block2imu_id_field[block].set(imu_nicknames[i])
                button_text = 'Connect'
                func = lambda b=str(block): self.connectionAttemptDialog(b)
            
            menu_args = (master, self.block2imu_id_field[block]) \
                      + imu_nicknames
            imu_menu = apply(tk.OptionMenu, menu_args)
            imu_menu.grid(sticky=tk.W, row=i+1, column=1)
            
            button = tk.Button(master, text=button_text, command=func)
            button.grid(sticky=tk.W, row=i+1, column=2)
            self.block2button[block] = button
        
        # Draw navigation buttons
        master = self.navigation_frame
        forward = tk.Button(master, text='Select task >>', command=self.forward,
                           default=tk.ACTIVE)
        forward.grid(sticky=tk.E, row=0, column=1)
        back = tk.Button(master, text='<< Enter data', command=self.back)
        back.grid(sticky=tk.W, row=0, column=0)
    
    
    def defineTaskInterface(self):
        """
        Set up the block task selection interface.
        """
        
        master = self.content_frame
        
        # Draw instructions
        user_text = "Select the current block construction task."
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
        back = tk.Button(master, text="<< Connect IMUs", command=self.back)
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
        
        self.imu_label = tk.Label(master, text='IMU activity')        
        self.imu_label.grid(row=1, column=2, sticky='N') 
        
        # Draw placeholders for IMU monitors
        self.imu_id2activity_color = {}
        imu_monitor_frame = tk.Frame(master)
        for i, imu_id in enumerate(self.imu_id2dev.keys()):
            # Label text
            color, shape = self.imu2block[imu_id].split()
            shape_text = '            ' if shape == 'rect' else '    '
            color_label = tk.Label(imu_monitor_frame, text=shape_text,
                                   background=color)
            color_label.grid(row=i, column=0)
            id_label = tk.Label(imu_monitor_frame, text=' active: ')
            id_label.grid(row=i, column=1)
            # Activity indicator
            self.imu_id2activity_color[imu_id] = tk.Label(imu_monitor_frame,
                                                          text='    ',
                                                          background='black')
            self.imu_id2activity_color[imu_id].grid(row=i, column=2)
        imu_monitor_frame.grid(row=1, column=2)
            

        # Draw navigation buttons
        master = self.navigation_frame
        q = tk.Button(master, text='Quit', command=self.closeStream)
        q.grid(sticky=tk.E, row=0, column=1)
        p = tk.Button(master, text='New task', command=self.chooseNewTask)
        p.grid(sticky=tk.W, row=0, column=0)
                
        self.parent.after(75, self.refreshStreamInterface)
    
       
    def startStreamProcesses(self):
        """
        Define and start running processes that stream from IMUs and cameras
        """
        
        # Define paths used for file I/O when streaming data
        self.raw_imu_path = os.path.join(self.corpus.paths['raw'], str(self.trial_id))
        frame_base_path = os.path.join(self.corpus.paths['video-frames'],
                                       str(self.trial_id))
        timestamp_fn = '{}-timestamps.csv'.format(self.trial_id)
        timestamp_path = os.path.join(self.corpus.paths['raw'], timestamp_fn)
        
        # Active block --> IMU nickname --> IMU id --> IMU device
        # I know this is stupid but it's the best I can do for now
        active_nicknames = [self.block2imu_nickname[block] for block in self.active_blocks]
        active_ids = [self.corpus.nickname2id[nn] for nn in active_nicknames]
        self.active_devices = {ID: self.imu_id2dev[ID] for ID in active_ids}
        
        # Define processes that stream from IMUs and camera
        videostream_args = (frame_base_path, timestamp_path,
                            self.corpus.image_types, self.die, self.video_q)
        self.processes = (mp.Process(target=ps.stream, args=videostream_args),)
        
        # Turn on imu sampling
        for dev in self.active_devices.values():
            dev.start_sampling()
        
        # Turn on RGBD sampling
        for p in self.processes:
            p.start()
    
    
    def float2HexColor(self, f, upper_thresh, lower_thresh):
        """
        Convert a double-precision floating point number to a TK color string.

        Parameters
        ----------
        f : double
          Double-precision floating-point number that will be converted
        upper_thresh : double
          Upper limit on range of quantity
        lower_thresh : double
          Lower limit on range of quantity

        Returns
        -------
        color_str : str
          f encoded as a 24-byte hex string, format '#RRGGBB'
        """
        
        thresh_centered = upper_thresh - lower_thresh
        f_centered = max(f, lower_thresh) - lower_thresh
        f_clipped = min(f_centered, thresh_centered)
        f_int = int(f_clipped / thresh_centered * 255)
        
        r = 255 - f_int
        g = f_int
        b = 0
        
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)
    
 
    def refreshStreamInterface(self):
        """
        Update frame on the RGB video monitor and check to make sure none of
        the IMUs has died.
        """
        
        # Check if an IMU has died and notify the user if it has
        if self.die.is_set() and not self.die_set_by_user:
            self.imuFailureDialog()
            return
        
        for imu_id, dev in self.imu_id2dev.items():
            a = dev.get_accel_sample()
            if a is not None:
                a_norm = (a[1] ** 2 + a[2] ** 2 + a[3] ** 2) ** 0.5 - 1.0
                a_color = self.float2HexColor(a_norm, 3.5, 0.0)
                self.imu_id2activity_color[imu_id].configure(background=a_color)
        
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
        
        # Map the blocks used in this trial to the IDs of the IMUs inside them.
        # Map the blocks that weren't used to the string, 'UNUSED'.
        block_mapping = {imu_id: 'UNUSED' for imu_id in self.corpus.imu_ids}
        for block in self.active_blocks:
            imu_id = self.corpus.nickname2id[self.block2imu_nickname[block]]
            block_mapping[imu_id] = block
        
        # Update the metadata array and increment the trial index
        # FIXME
        metadata = (self.participant_id, self.birth_month, self.birth_year,
                    self.gender, self.task)
        #imu_settings_array = np.hstack(tuple(self.imu_settings))
        #self.corpus.postprocess(self.trial_id, metadata, block_mapping,
        #                        imu_settings_array)
        self.corpus.postprocess(self.trial_id, metadata, block_mapping)
        self.trial_id = self.corpus.meta_data.shape[0]
        
        # Reset die so we don't immediately quit streaming data in the next
        # round
        self.die.clear()
        
        # task selection interface is at position 2
        self.clearInterface()
        self.interface_index = 2
        interface = self.interfaces[self.interface_index]
        interface()
        self.drawInterface()
    
    
    def connectionAttemptDialog(self, block):
        """
        For a given block, check whether the specified block is already in use.
        If it is, draw a popup window prompting the user to select a different
        device. If it isn't, try to connect to the device.
        
        Parameters
        ----------
        block : str
          Color of the block housing the IMU
        """
        
        if not self.popup is None:
            return
        
        nickname = self.block2imu_id_field[block].get()
        imu_id = self.corpus.nickname2id[nickname]
        
        if imu_id in self.imu_id2dev:
            self.popup = tk.Toplevel(self.parent)
            fmtstr = 'Device {} is already in use! Choose a different device.'
            l = tk.Label(self.popup, text=fmtstr.format(nickname))
            l.pack()
            ok = tk.Button(self.popup, text='OK', command=self.cancel)
            ok.pack()
        elif block in self.block2imu_nickname:
            self.popup = tk.Toplevel(self.parent)
            nickname = self.block2imu_nickname[block]
            fmtstr = 'This block is already associated with device {}!'
            l = tk.Label(self.popup, text=fmtstr.format(nickname))
            l.pack()
            ok = tk.Button(self.popup, text='OK', command=self.cancel)
            ok.pack()
        else:
            self.attemptConnection(nickname, block)
    
    
    def resetConnection(self, block):
        """
        Close the socket for the IMU currently connected to the specified block
        
        Parameters
        ----------
        block : str
          Color of the current block
        """
        
        # block name --> imu nickname --> imu 4-digit hex id --> imu socket
        nickname = self.block2imu_nickname[block]
        imu_id = self.corpus.nickname2id[nickname]
        dev = self.imu_id2dev[imu_id]
        
        # Disconnect from socket
        print('Disconnecting from {}...'.format(imu_id))
        dev.disconnect()
        
        # Update dictionaries
        self.imu_id2dev.pop(imu_id, None)
        self.block2imu_nickname.pop(block, None)
        self.imu2block.pop(imu_id, None)
        
        # Re-configure button
        func = lambda b=str(block): self.connectionAttemptDialog(b)
        self.block2button[block].configure(text='Connect', command=func)
        
    
    def attemptConnection(self, nickname, block):
        """
        Try to connect to the specified IMU.
        
        Parameters
        ----------
        nickname : str
          Simple label given to IMU (one character, e.g. '1')
        block : str
          Color of the current block
        """
        
        imu_address = self.corpus.nickname2address[nickname]
        try:
            bt_interface = self.bt_interfaces[self.bt_interface_index]
            print('\nConnecting to device {} on {}'.format(imu_address, bt_interface))
            dev = MetawearDevice(imu_address, interface=bt_interface)
        except (PyMetaWearConnectionTimeout, RuntimeError) as e:
            self.badInputDialog(str(e))
            return
        
        # cycle to next bluetooth interface
        self.bt_interface_index = (self.bt_interface_index + 1) % len(self.bt_interfaces)
        dev.init_streaming()
        dev.set_accel_params(sample_rate=50.0)
        dev.set_gyro_params(sample_rate=50.0)
        dev.set_ble_params(7.5, 7.5, 0, 10)
        
        imu_id = dev.name
        self.imu_id2dev[imu_id] = dev
        self.block2imu_nickname[block] = nickname
        self.imu2block[imu_id] = block
        print('{}: {}'.format(block, imu_id))
                    
        # Update 'connect' button
        func = lambda b=str(block): self.resetConnection(b)
        self.block2button[block].configure(text='Disconnect', command=func)
        
        battery = dev.battery[1]  # Battery voltage in % charge remaining
        self.connectionSuccessDialog(nickname, battery)
        
    
    def connectionFailureDialog(self):
        """
        Draw a popup window informing the user that the connection attempt has
        failed.
        """
        
        self.popup = tk.Toplevel(self.parent)
        
        fmtstr = 'Connection attempt failed! Cycle the device and try again.'
        l = tk.Label(self.popup, text=fmtstr)
        l.grid(row=0)
        
        ok = tk.Button(self.popup, text='OK', command=self.cancel)
        ok.grid(row=1)
    
    
    def connectionSuccessDialog(self, nickname, battery):
        """
        Draw a popup window informing the user that the connection attempt was
        successful.
        
        Parameters
        ----------
        nickname : str
          Simple label given to IMU (one character, e.g. '1')
        battery : int
          IMU charge relative to full, in percent
        """
        
        self.popup = tk.Toplevel(self.parent)
        
        fmtstr = '\nSuccessfully connected to device {}!\n\nBattery: {}%\n'
        l = tk.Label(self.popup, text=fmtstr.format(nickname, battery))
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
        
        Parameters 
        ----------
        error_string :  str
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
        
        for name, dev in self.imu_id2dev.items():
            print('Disconnecting from {}...'.format(name))
            dev.disconnect()
        
        # Map the blocks used in this trial to the IDs of the IMUs inside them.
        # Map the blocks that weren't used to the string, 'UNUSED'.
        block_mapping = {imu_id: 'UNUSED' for imu_id in self.corpus.imu_ids}
        for block in self.active_blocks:
            imu_id = self.corpus.nickname2id[self.block2imu_nickname[block]]
            block_mapping[imu_id] = block
        
        # FIXME
        metadata = (self.participant_id, self.birth_month, self.birth_year,
                    self.gender, self.task)
        #imu_settings_array = np.hstack(tuple(self.imu_settings))
        #self.corpus.postprocess(self.trial_id, metadata, block_mapping,
        #                        imu_settings_array)
        self.corpus.postprocess(self.trial_id, metadata, block_mapping)
        #self.corpus.makeImuFigs(self.trial_id)
        
        self.parent.destroy()
    
    
    def stopStream(self):
        """
        Stop streaming from IMUs and camera.
        """
        
        self.die_set_by_user = True
        self.die.set()
        
        # turn off inertial sampling
        for dev in self.active_devices.values():
            dev.stop_sampling()
        
        # turn off camera
        for p in self.processes:
            p.join()
    
        for dev in self.active_devices.values():
            base_path = os.path.join(self.corpus.paths['figures'], str(self.trial_id))
            dev.plot_data(base_path)
        
        for dev in self.active_devices.values():
            dev.write_data(self.raw_imu_path)
    
    
    def getMetaData(self):
        """
        Read metadata from tkinter widgets.
        
        Returns
        -------
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
    
    
    def getImuData(self):
        """
        Read block to IMU mappings from tkinter widgets.
        """
                
        # Make sure there is a connected IMU associated with every block
        for block in self.blocks:
            if not block in self.block2imu_nickname:
                fmtstr = 'Please put an IMU in the {} block and click connect.'
                return fmtstr.format(block)
        
        return ''
    
    
    def getTaskData(self):
        """
        Read the block construction task from tkinter widgets.
        
        Returns
        -------
        error_string : str
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
    

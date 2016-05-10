# -*- coding: utf-8 -*-
"""
datacollectiongui.py

AUTHOR
  Jonathan D. Jones
"""

from __future__ import print_function
from Tkinter import *
from PIL import Image, ImageTk

import multiprocessing as mp
import select

import csv
import os
import sys
import time
import struct

from primesense import openni2
import cv2
import numpy as np

#import streamdata as sd
import libwax9 as wax9
from duplocorpus import DuploCorpus


class Application:
    
    def __init__(self, parent):
        
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
        self.imu_ids = ('08F1', '0949', '090F', '095D')
        self.imu2block = {x: 'UNUSED' for x in self.imu_ids}
        self.frame_vars = ()
        self.active_blocks = ()
        self.metadata = {}
        self.imu_settings = {}
        
        # For streaming in parallel
        self.die = mp.Event()
        self.corpus = DuploCorpus()
        self.trial_id = self.corpus.meta_data.shape[0]
        raw_file_path = os.path.join(self.corpus.paths['imu-raw'], '{}.csv'.format(self.trial_id))
        rgb_trial_path = os.path.join(self.corpus.paths['rgb'], str(self.trial_id))
        if not os.path.exists(rgb_trial_path):
            os.makedirs(rgb_trial_path)
        self.streamProcesses = (mp.Process(target=self.streamImu,
                                           args=(raw_file_path,)),
                                mp.Process(target=self.streamVideo,
                                           args=('IMG-RGB', rgb_trial_path)))
        
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
    
        # Set up widgets and define layoutdev_id.get()
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
            
            commands.append(lambda b=str(block): self.connectionAttemptDialog(b))
            buttons.append(Button(master, text='Connect', command=commands[-1]))
            buttons[-1].grid(sticky=W, row=i+1, column=2)
            
        submit = Button(master, text='Next', command=self.forward, default=ACTIVE)
        submit.grid(sticky=E, row=len(self.dev_ids)+1, column=2)
        
        back = Button(master, text='Back', command=self.back)
        back.grid(sticky=W, row=len(self.dev_ids)+1, column=0)
        
        master.place(relx=0.5, rely=0.5, anchor='center')
    
    
    def drawStreamContent(self):
        
        # TODO: 
        
        master = self.content
        
        l = Label(master, text='Streaming data...')
        l.grid(row=0, column=0, columnspan=2)
        
        q = Button(master, text='Quit', command=self.closeStream)
        q.grid(row=1, column=1)
        
        p = Button(master, text='Pause', command=self.stopStream)
        p.grid(row=1, column=0)
        
        master.place(relx=0.5, rely=0.5, anchor='center')
        
        for dev_id, imu_socket in self.connected_devices.items():
            print(dev_id)
            settings = wax9.getSettings(imu_socket)
            self.imu_settings[dev_id] = settings
            print(settings)
        
        for p in self.streamProcesses:
            p.start()
    
    
    def connectionAttemptDialog(self, block):
        
        if not self.popup is None:
            return
        
        dev_id = self.dev_ids[block]
        imu_id = dev_id.get()
        
        self.popup = Toplevel(self.parent)
        
        if imu_id in self.connected_devices.keys():
            fmtstr = 'Device {} is already in use! Choose a different device.'
            l = Label(self.popup, text=fmtstr.format(imu_id))
            l.pack()
            
            ok = Button(self.popup, text='OK', command=self.cancel)
            ok.pack()
        else:
            fmtstr = 'Connecting to {}...'
            l = Label(self.popup, text=fmtstr.format(imu_id))
            l.pack()
            
            c = Button(self.popup, text='cancel', command=self.cancel)
            c.pack()
            
            # Actually try to connect
            mac_prefix = ['00', '17', 'E9', 'D7']
            imu_address = ':'.join(mac_prefix + [imu_id[0:2], imu_id[2:4]])
            socket, name = wax9.connect(imu_address)
            if name is None:
                self.connectionFailureDialog(block)
            else:
                self.connected_devices[imu_id] = socket
                self.connectionSuccessDialog(name)
    
    
    def connectionFailureDialog(self, block):
        self.popup.destroy()
        self.popup = Toplevel(self.parent)
        
        fmtstr = 'Connection attempt failed! Try again?'
        l = Label(self.popup, text=fmtstr)
        l.grid(row=0, columnspan=2)
        
        func = lambda b=str(block): self.connectionAttemptDialog(b)
        y = Button(self.popup, text='Yes', command=func)
        n = Button(self.popup, text='No', command=self.cancel)
        
        y.grid(row=1, column=0)
        n.grid(row=1, column=1)
    
    
    def connectionSuccessDialog(self, imu_id):
        self.popup.destroy()
        self.popup = Toplevel(self.parent)
        
        fmtstr = 'Successfully connected to {}!'
        l = Label(self.popup, text=fmtstr.format(imu_id))
        l.pack()
        
        ok = Button(self.popup, text='OK', command=self.cancel)
        ok.pack()
    
    
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
    
    
    def closeStream(self):
        
        if not self.die.is_set():
            self.stopStream()
        
        for name, socket in self.connected_devices.items():
            print('Disconnecting from {}...'.format(name))
            socket.close()
        
        meta = (self.metadata['pnum'], self.metadata['dob_month'],
                self.metadata['dob_year'], self.metadata['gender'])
        child_id = '_'.join(meta)
        self.corpus.postprocess(child_id, self.trial_id, self.connected_devices,
                                self.imu_settings, 'IMG-RGB', self.imu2block)
        
        ids = [x[-4:] for x in self.connected_devices.keys()]  # Grab hex ID from WAX9 ID
        self.corpus.makeImuFigs(self.trial_id, ids)
        
        self.parent.destroy()
    
    
    def stopStream(self):
        
        self.die.set()
        for p in self.streamProcesses:
            p.join()
    
    
    def getMetaData(self):
        self.metadata['pnum'] = self.participant_num.get()
        self.metadata['dob_month'] = self.dob_month.get()
        self.metadata['dob_year'] = self.dob_year.get()
        self.metadata['gender'] = self.gender.get()
        
        print('Participant ID: {}'.format(self.metadata['pnum']))
        print('Birth date: {} / {}'.format(self.metadata['dob_month'],
                                           self.metadata['dob_year']))
        print('Gender: {}'.format(self.metadata['gender']))
    
    
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
            imu = dev_id.get()
            self.imu2block[imu] = block
            print('{}: {}'.format(block, self.imu2block[imu]))
    
    
    def getStreamData(self):
        # TODO
        print('I am a fake method!')
    
    
    def streamImu(self, path):
        """
        Stream data from WAX9 devices until die is set
    
        Args:
        -----
        [str] path: Path to raw IMU output file
        """
        
        SAMPLE_RATE = 15
        TIMEOUT = 1.5 * (1.0 / SAMPLE_RATE)
        
        # If we see more than this many timeouts, it means we've dropped samples
        # for three seconds consecutively. The IMU is probably stalled in this
        # case, so stop data collection and prompt user to cycle the device.
        MAX_NUM_TIMEOUTS = 3 * SAMPLE_RATE
        
        # Set up output files
        f_imu = open(path, 'w')
        imu_writer = csv.writer(f_imu)
        
        # Special SLIP bytes
        SLIP_END = '\xc0'
        SLIP_ESC = '\xdb'
        SLIP_ESC_END = '\xdc'
        SLIP_ESC_ESC = '\xdd'
        
        # This dictionary is used to track the number of consecutive dropped
        # samples for each IMU device
        num_consec_timeouts = {dev_name: 0 for dev_name in self.connected_devices.keys()}
    
        # Tell the imu devices to start streaming
        for imu_socket in self.connected_devices.values():
            imu_socket.sendall('stream\r\n')
        
        while True:
            
            # Read data sequentially from sensors until told to terminate
            for dev_id, imu_socket in self.connected_devices.items():
                
                # Read bytes from the current sensor until we have a complete
                # packet or hang on a timeout
                prev_escaped = False
                packet_ready = False
                timeout = False
                frame = ''
                while not packet_ready:                
                    # Time out if we spend longer than the sample period waiting to
                    # read from the socket
                    ready = select.select([imu_socket], [], [], TIMEOUT)
                    if ready[0]:
                        # Reset the number of consecutive dropped samples once we
                        # successfully read one
                        num_consec_timeouts[dev_id] = 0
                        
                        # Read and process one byte from the IMU
                        byte = imu_socket.recv(1)
                        if prev_escaped:
                            prev_escaped = False
                            if byte == SLIP_ESC_END:
                                frame += SLIP_END
                            elif byte == SLIP_ESC_ESC:
                                frame += SLIP_ESC
                            else:
                                # Anything else means we received an unexpected escaped byte
                                print('ERR | {} | unexpected byte'.format(byte.encode('hex')))
                        elif byte == SLIP_END and len(frame) > 0:
                            frame += byte
                            packet = frame
                            frame = ''
                            packet_ready = True
                        elif byte == SLIP_ESC:
                            prev_escaped = True
                        else:
                            frame += byte
                    # If select times out, we've waited longer than a sample
                    # period. This sample must be bad, so throw it out by resetting
                    # the data frame.
                    else:
                        packet = frame
                        
                        frame = ''
                        packet_ready = True
                        prev_escaped = False
                        timeout = True
                        
                        num_consec_timeouts[dev_id] += 1
                        if num_consec_timeouts[dev_id] > MAX_NUM_TIMEOUTS:
                            self.die.set()
                            fmtstr = '\nDevice {} is unresponsive. Data collection' \
                                     ' halted. Cycle devices and restart software.'
                            print(fmtstr.format(dev_id))
                    
                # Once we read a complete packet, unpack the bytes and write the
                # sample to file
                cur_time = time.time()
                
                if timeout:
                    timeouts = num_consec_timeouts[dev_id]
                    error = 1
                    fmtstr = 'ERR | timeout  ({} consecutive) | {} | {}'
                    print(fmtstr.format(timeouts, dev_id, packet.encode('hex')))
                    data = [0] * 14
                    data[1] = error
                    data[-1] = dev_id
                    timeout = False
                elif len(packet) == 28:   # Standard packet
                    error = 0
                    # Convert data from hex representation (see p. 7, 'WAX9
                    # application developer's guide')
                    fmtstr = '<' + 3 * 'B' + 'h' + 'I' + 9 * 'h' + 'B'
                    unpacked = list(struct.unpack(fmtstr, packet))
                    data = [cur_time, error] + unpacked[3:14] + [dev_id]
                elif len(packet) == 36: # Long packet
                    error = 0
                    # Convert data from hex representation (see p. 7, 'WAX9
                    # application developer's guide')
                    fmtstr = '<' + 3 * 'B' + 'h' + 'I' + 11 * 'h' + 'I' + 'B'
                    unpacked = list(struct.unpack(fmtstr, packet))
                    data = [cur_time, error] + unpacked[3:14] + [dev_id]
                else:
                    error = 1
                    # Record an error
                    fmtstr = 'ERR | Bad packet | {} | {}'
                    print(fmtstr.format(dev_id, packet.encode('hex')))
                    data = [0] * 14
                    data[1] = error
                    data[-1] = dev_id
                    
                # Queue the packet to be written
                imu_writer.writerow(data)
            
            # Terminate when main tells us to -- but only after we finish
            # reading the current set of packets
            if self.die.is_set():
                # Tell the devices to stop streaming
                for imu_socket in self.connected_devices.values():
                    imu_socket.sendall('\r\n')
                break
        
        f_imu.close()


    def streamVideo(self, dev_name, path):
        """
        Stream data from camera until die is set
    
        Args:
        -----
        [dict(str->cv stream)] dev_name:
        [str] path: Path (full or relative) to image output directory
        """
        
        f_rgb = open(os.path.join(path, 'frame_timestamps.csv'), 'w')
        rgb_writer = csv.writer(f_rgb)
        
        # Open video streams
        print("Opening RGB camera...")
        openni2.initialize()
        dev = openni2.Device.open_any()
        stream = dev.create_color_stream()
        print("RGB camera opened")
        
        frame_index = 0
        
        print("Starting RGB stream...")
        stream.start()
        print("RGB stream started")
        
        while not self.die.is_set():
            """
            # Read depth frame data, convert to image matrix, write to file,
            # record frame timestamp
            frametime = time.time()
            depth_frame = depth_stream.read_frame()
            depth_data = depth_frame.get_buffer_as_uint16()
            depth_array = np.ndarray((depth_frame.height, depth_frame.width),
                                     dtype=np.uint16, buffer=depth_data)
            filename = "depth_" + str(i) + ".png"
            cv2.imwrite(os.path.join(dirname, filename), depth_array)
            q.put((i, frametime, "IMG_DEPTH"))
            """
            
            # Read frame data, record frame timestamp
            frame = stream.read_frame()
            frametime = time.time()
            
            # Convert to image array
            data = frame.get_buffer_as_uint8()
            img_array = np.ndarray((frame.height, 3*frame.width),
                                   dtype=np.uint8, buffer=data)
            img_array = np.dstack((img_array[:,2::3], img_array[:,1::3], img_array[:,0::3]))
            
            # Write to file
            filename = '{:06d}.png'.format(frame_index)
            cv2.imwrite(os.path.join(path, filename), img_array)
            rgb_writer.writerow((frametime, frame_index, dev_name))
    
            frame_index += 1
    
        # Stop streaming
        print("Closing RGB camera")    
        stream.stop()
        openni2.unload()
        
        f_rgb.close()


if __name__ == '__main__':
    
    root = Tk()
    app = Application(root)
    
    root.mainloop()
    
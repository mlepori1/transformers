"""
processdata.py
  Align recorded data according to universal timestamp

AUTHOR
  Jonathan D. Jones
"""

import sys
import csv
from collections import deque

def allNotEmpty(queues):
    """
    Return True if no elements of input are empty, else False
    """

    allNotEmpty = True
    for q in queues:
        allNotEmpty = allNotEmpty and len(q) > 0

    return allNotEmpty


# Map device name to position in file
name2pos = {'WAX9-08F1':0, 'WAX9-090F':1, 'WAX9-095D':2, 'WAX9-0949':3}
queues = [deque() for i in name2pos.items()]

filename = "imu-data_1453909220.csv"
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        # Push row to the appropriate queue
        if len(row) == 13:
            q_idx = name2pos[row[-1]]
            queues[q_idx].appendleft(row)

        # When none of the queues are empty, we're ready to write a new line
        if allNotEmpty(queues):
            minval = int(1e9) # SURE HOPE WE NEVER HAVE MORE THAN A BILLION SAMPLES
            maxval = 0
            argmax = []
            rows = []
            for i, q in enumerate(queues):
                cur_row = q.pop()
                sample_idx = int(cur_row[0])
                rows.append(cur_row)

                #import pdb; pdb.set_trace()
                if sample_idx < minval:
                    minval = sample_idx
                if sample_idx > maxval:
                    maxval = sample_idx
                    argmax = [i]
                elif sample_idx == maxval:
                    argmax.append(i)

            # All sample indices are the same, so we didn't miss any data.
            # Write a line to the output file.
            if minval == maxval:
                line = []
                for cur_row in rows:
                    line += cur_row[:1] + cur_row[-2:]
                print(line)
            # We missed a sample somewhere. Skip this line and push the
            # max-index samples back to the front of their queues
            else:
                print('')
                line = []
                for cur_row in rows:
                    line += cur_row[:1] + cur_row[-2:]
                print(line)
                for i in argmax:
                    queues[i].append(rows[i])


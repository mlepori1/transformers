
import numpy as np
import ezodf
import os
from pylab import *
import matplotlib.gridspec as gridspec

uri = os.path.expanduser("~/Desktop/Blocks/annotations/annotations/")
filename_spatial = "/SpatialLabels.ods"

# Get folders
folders = os.listdir(uri)
folders = filter(lambda x:"model" in x, folders)
folders = filter(lambda x:"annotated" not in x, folders)

# for folder in folders:
if 1:
    folder = folders[5]


    doc = ezodf.opendoc(uri+folder+filename_spatial)
    table = doc.sheets[0]

    list(table.rows())

    class BlockConfig:
        layer = None
        color = None
        start = None
        end = None

        def __init__(self, layer, color, start, end):
            # Start/end = (y,x)
            self.layer = layer
            self.color = color
            self.start = start
            self.end = end

    # Extract block configurations
    configs_all = []
    img_names_all = []
    config = []
    for i, row in enumerate(table.rows()):
        cols = [c.plaintext() for c in row]
        # print cols
        if "filename" in cols[0]:
            continue    
        # If new image/config
        elif "png" in cols[0]:
            if len(config) == 0:
                config = []
                try:
                    img_names_all.pop()
                except:
                    print "Can't pop"
            else:
                configs_all += [config]
                config = []
            img_names_all += [cols[0]]                

        # If continuation of block config
        else:
            try:
                _,layer,color,s_y,s_x,e_y,e_x = cols
                config += [BlockConfig(layer, color, (int(s_y)-1,int(s_x)-1), 
                                                     (int(e_y)-1,int(e_x)-1))]
            except:
                print cols
    # Add last config
    if len(config) == 0:
        img_names_all.pop() 
    else:
        configs_all += [config]
    img_names_all += [cols[0]]                


    n_configs = len(configs_all)
    grid = np.zeros([n_configs, 6,6,3])
    for i, blocks in enumerate(configs_all):
        for b in blocks:
            color = matplotlib.colors.hex2color(matplotlib.colors.cnames[b.color])
            grid[i, b.start[0]:b.end[0]+1, b.start[1]:b.end[1]+1] = color


    figure(figsize=(18,10))
    n_cols = int(round(np.sqrt(n_configs)))
    n_rows = int(round(np.sqrt(n_configs)))
    while n_cols*n_rows < n_configs:
        n_rows += 1

    # Setup grid with two panels
    gs = gridspec.GridSpec(n_rows, n_cols*2)

    for i in range(n_configs):
        r,c = np.unravel_index(i, (n_rows, n_cols))
        ax1 = plt.subplot(gs[r, c])
        ax1.imshow(grid[i], interpolation='nearest')
        axis("off")

        r,c = np.unravel_index(i, (n_rows, n_cols))
        ax1 = plt.subplot(gs[r, c+n_cols])
        img_name = img_names_all[i].split("/")[-1]
        img = imread(uri+folder+"/"+img_name)
        ax1.imshow(img)
        axis("off")

    suptitle(folder)
    savefig(uri+"annotated_"+folder+".jpg")






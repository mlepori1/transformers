# -*- coding: utf-8 -*-
"""
renderworldstate.py
  Load block positions and orientations and draw them in blender.

AUTHOR
  Jonathan D. Jones
"""

import bpy
import csv

fn = '/Users/jonathan/block_coords/0.csv'
with open(fn) as csvfile:
    csvreader = csv.reader(csvfile)
    
    #"""
    i = 0
    #i = 10.0
    #bpy.ops.mesh.primitive_cube_add(radius=15.9 / 10)
    #ob = bpy.context.object
    #ob.color = (255, 0, 0, 0)
    #me = ob.data
    #ob.keyframe_insert(data_path='location', frame=i)
    #ob.keyframe_insert(data_path='rotation_euler', frame=i)
    objects = []
    for row in csvreader:
        
        position = list(float(x) / 10 for x in row[0:3])
        position[-1] = - position[-1]
        angle = (0, 0, float(row[3]) + 3.14 / 2)
        r_width, r_length, r_height = tuple(float(x) / 2 / 10 for x in row[4:7])
        color = tuple(int(x) for x in row[7:10])
        
        #c = bpy.ops.mesh.primitive_cube_add(radius=15.9, location=position,
        #                                    rotation=angle)
        # Create mesh and object
        meshName = "block_mesh_{}".format(i)
        obName = "block_{}".format(i)
        me = bpy.data.meshes.new(meshName)
        ob = bpy.data.objects.new(obName, me)
        ob.show_name = True
        
        # Link object to scene and make active
        scn = bpy.context.scene
        scn.objects.link(ob)
        scn.objects.active = ob
        ob.select = True
        
        #r_height = 19.2 / 2 / 10
        #r_width = 31.8 / 2 / 10
        #r_length = 2 * r_width
        verts = [( r_width,  r_length,  r_height), (-r_width,  r_length,  r_height),
                 (-r_width, -r_length,  r_height), ( r_width, -r_length,  r_height),
                 ( r_width,  r_length, -r_height), (-r_width,  r_length, -r_height),
                 (-r_width, -r_length, -r_height), ( r_width, -r_length, -r_height),]
        faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 4, 7, 3), (1, 5, 6, 2),
                 (0, 4, 5, 1), (3, 7, 6, 2)]
        me.from_pydata(verts, [], faces)
        me.update()
        
        ob.rotation_mode = 'XYZ'
        ob.location = position
        ob.rotation_euler = angle
        #ob.keyframe_insert(data_path='location', frame=i)
        #ob.keyframe_insert(data_path='rotation_euler', frame=i)
        
        objects.append(ob)
        
        i += 1
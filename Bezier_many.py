
#%%
import os
from pathlib import Path

import numpy as np
import random

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import *

import pygestalt as gs
from pygestalt import sampler

import skimage as ski

import math
from matplotlib.backends.backend_agg import FigureCanvasAgg

import HandyFuncs as HF
from IPython.display import clear_output

# This just allows editing of HandyFuncs.py without having to restart VS Code
from importlib import reload
reload(HF)
reload(gs.utils)
reload(sampler)
########################################################################
#%%
### This cell for adding hagelslag ###

start = 1
n_each = 1

TL, BL, TR, BR = [.25,.75],[.25,.25],[.75,.75],[.75,.25]
shapes = ['BC','C']
Qs = [TL, BL, TR, BR]
Pos = [-0.15, -0.05, 0.05, 0.15]

n_total = n_each * len(Qs) * len(shapes) * len(Pos)
counter = 0

SeedJs, SeedJo, SeedPs = None, None, None
Js, Jo = 0.125,10 #0.125, 10 # Spatial and Orientation Jitter

breakOut = False
# plt.ioff()
for c in range(start,start+n_each):
    for a in shapes:
        for b in Qs:
            for p in Pos:
                # Generate the 7 points
                pMod = np.array([p,0])
                b = np.array(b)
                Ps,Ts = HF.regularPolygon(20,a, center=b+pMod)
                if a == 'BC': # It duplicates the center line, this is easiest fix
                    Ps = Ps[0:7]
                Ps = Ps[Ps[:, 1].argsort()[::-1]]
                bx,by = Ps[:,0], Ps[:,1]
                # # These 5 things override the plot and save just the bezier, not any hagelslag
                # plt.plot(bx,by,'.')
                # ax = plt.gca()
                # ax.set_xlim((0,1))
                # ax.set_ylim((0,1))
                # ax.set_aspect('equal')
                
                if Qs.index(b.tolist()) == 0:
                    bLab = 'TL'
                elif Qs.index(b.tolist()) == 1:
                    bLab = 'BL'
                elif Qs.index(b.tolist()) == 2:
                    bLab = 'TR'
                else:
                    bLab = 'BR'
                pLab = chr(97 + Pos.index(p))# 'a' = 97 in ASCII

                # Radius and Thresh here are important
                #Generate Random Stimuli
                radius = 0.03
                thresh = 0.5e-2 # e-3
                # Ps = np.asarray((bx,by)).T

                ### Here we mess around w/ the default Han code some more 
                ### and use our own points for tangents
                C, H = Ps, Ps # Ps, Ts, but Ts is wrong for `bc` and pointless
                # C, H = sampler.draw_positions(radius, sampler.bezier_curve(Ps.T,SeedPs), thresh=thresh)
                # C, H = sampler.draw_positions(radius, sampler.point_set(Ps), thresh=thresh)
                C = HF.spatialJitterUni(Ps,radius,Js,SeedJs)
                D, _ = sampler.draw_positions(radius, sampler.box(), exclusions=C, thresh=thresh)

                # Add Jitters
                H1 = gs.utils.add_jitter3(bx, by, phi = 20,shape=a, jitter = Jo, seed=SeedJo).T #The final number is the jitter value

                theta = np.rad2deg(np.linalg.norm(np.stack([np.arccos(H1[:,0]),np.arcsin(H1[:,1])],1),axis=1))
                gamma = np.random.uniform(0,360,D.shape[0])

                GT = np.hstack([gamma,theta])
                DP = np.vstack([D,Ps])
                GT = GT.reshape([GT.size,1])
                DPGT = np.hstack([DP,GT]) # columns are [x,y,angle], contains all points

                adjusted = HF.adjust_angles(DPGT,
                                            len(Ps),
                                            angle_threshold=30,
                                            max_attempts=1000,
                                            method='r',
                                            K=8,
                                            R=radius*6)
                all_angles = adjusted[:,2]
                # `all_angles[-len(Ps):] == theta` should be true
                D_angles = all_angles[:-len(Ps)]
                D1 = np.asarray([np.cos(D_angles), np.sin(D_angles)]).T

                # Define the patch function
                l=0.025
                w=0.005
                pfunc = lambda z,h: gs.patch.segment(z, h, l, w)


                #Perfectly aligned
                N = 750
                Ig = gs.patch.generate_image(D, D1, N=N, pfunc=pfunc)

                

                If = gs.patch.generate_image(C, H1, N=N, pfunc=pfunc)
                #If = gs.patch.generate_image(C, N=N, pfunc=pfunc)

                I = If + Ig

                Im = I.copy()
                # Im = np.fliplr(np.rot90(Im,-1))
                pix = 1069
                plt.figure(figsize=(pix/72,pix/72))
                # plt.figure(figsize=(N,N), dpi=1)
                plt.imshow(Im, aspect='equal', origin='lower', cmap='gray')
                # plt.set_cmap('binary')

                # This part from Mandoh
                plt.tight_layout()
                plt.axis('Off')

                if breakOut:
                    break
                output_folder = 'outputs'

                file_path = os.path.join(output_folder, f"{a}_{bLab}_{pLab}_{c}.png")

                plt.savefig(file_path, bbox_inches = 'tight', dpi = 72, pad_inches = 0, pil_kwargs = {'size':(1024,1024)})
                plt.clf()
                counter += 1
                print(f"Image {counter} saved.    {n_total - counter} remaining.")
                clear_output(wait=True)
            if breakOut:
                break
        if breakOut:
            break
    if breakOut:
        break
 # %%

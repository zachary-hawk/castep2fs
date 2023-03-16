#!/usr/bin/env python3
import numpy as np
import sys,os
import pyvista as pv
import warnings
import time
import argparse
from colorsys import rgb_to_hsv, hsv_to_rgb
import ase.io as io
#from Source import BZ
#from Source import bands
import BZ
import bands_old as bands
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from ase.spacegroup import Spacegroup
from itertools import cycle
from matplotlib import colors
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.collections as mcoll
import matplotlib.path as mpath





def main():    

    warnings.filterwarnings("ignore")
    
    
    # Start with the parser
    parser = argparse.ArgumentParser(description= "Visualisation of Brillouin zone and fermi surfaces for DFT calculations performed in the CASTEP code.")
    parser.add_argument("seed",help="The seed from the CASTEP calculation.")
    
    parser.add_argument("--save",help="Save image of BZ.",action="store_true")
    #parser.add_argument("-l","--labels",help="Turn on special labels",action="store_true")
    #parser.add_argument("--paths",help="Turn on special paths",action="store_true")
    parser.add_argument("-fs","--fermi",help="Suppress plot the of Fermi surface from a DOS .bands",action="store_false")
    parser.add_argument("-c","--colour",help="Matplotlib colourmap for surface colouring",default="default")
    parser.add_argument("--show",help="Choose which spin channels to plot.",choices=['both','up','down'],default="both")
    parser.add_argument("--nsurf",help="Choose which surfaces to plot, 0 indexed.",nargs="+")
    parser.add_argument("-p","--primitive",help="Display the primitive cell",action="store_true")
    parser.add_argument("-s","--smooth",help="Smoothing factor for Fermi surfaces",default=300,type=int)
    parser.add_argument("-v","--velocity",help="Colour Fermi Surfaces by Fermi Velocity",action="store_true")
    parser.add_argument("-o","--opacity",help="Opacity of Fermi Surfaces",default=1,type=float)
    parser.add_argument("--verbose",help="Set print verbosity",action="store_true")
    parser.add_argument("-z","--zoom",help="Zoom multiplier",default=1)
    parser.add_argument("-P","--position",help="Camera position vector, 6 arguments required in order given by 'verbose' output",nargs=6,default=np.array([0.,0.,0.,0.,0.,0.]),type=float)
    parser.add_argument("-f","--faces",help="Show faces surounding the Brillouin zone.", action="store_true")
    parser.add_argument("-B","--background",help="Background colour of plotting environment",default="Document",choices=["Document","ParaView","night","default"])
    parser.add_argument("-O","--offset",help="Fermi surface isovalue offset in eV",default=0.0,type=float)
    parser.add_argument("-a","--axes",help="Toggle axes visability",action="store_false")
    parser.add_argument("--axis_labels",help="Toggle axes labels, only visible when showing axes",action="store_false")
    parser.add_argument("--pdos",help="Use .pdos_bin to color fermi surface",action="store_true")
    parser.add_argument("--species",help="Project pdos onto species rather than orbitals",action='store_true')
    parser.add_argument("--gif",help="Option to generate an orbital .gif",action='store_true')
    parser.add_argument("-d",'--dryrun',help='Fermi surface analysis without displaying results',action="store_true")
    args = parser.parse_args()
    seed=args.seed
    save=args.save
    #plot_paths=args.paths
    #plot_labels=args.labels
    fermi=args.fermi
    col=args.colour
    start_time = time.time()
    show=args.show
    n_surf=args.nsurf
    prim=args.primitive
    smooth=args.smooth
    velocity=args.velocity
    opacity=args.opacity
    verbose=args.verbose
    cam_pos=args.position
    show_faces=args.faces
    background=args.background
    z=np.float(args.zoom)
    offset=args.offset
    show_axes=args.axes
    show_labels=args.axis_labels
    pdos=args.pdos
    species=args.species
    gif=args.gif
    dry=args.dryrun
    if not show_axes:
        show_labels=False








    def pdos_read(seed,species,bs):
        from scipy.io import FortranFile as FF

        f=FF(seed+'.pdos_bin', 'r','>u4')
        
        version=f.read_reals('>f8')
        header=f.read_record('a80')[0]
        num_kpoints=f.read_ints('>u4')[0]
        num_spins=f.read_ints('>u4')[0]
        num_popn_orb=f.read_ints('>u4')[0]
        max_eigenvalues=f.read_ints('>u4')[0]
        
        orbital_species=f.read_ints('>u4')
        orbital_ion=f.read_ints('>u4')
        orbital_l=f.read_ints('>u4')
        
        kpoints=np.zeros((num_kpoints,3))
        pdos_weights=np.zeros((num_popn_orb,max_eigenvalues,num_kpoints,num_spins))
        for nk in range(0,num_kpoints):
            record=f.read_record('>i4','>3f8')
            kpt_index,kpoints[nk,:]=record
            for ns in range(0,num_spins):
                spin_index=f.read_ints('>u4')[0]
                num_eigenvalues=f.read_ints('>u4')[0]
                
                for nb in range(0,num_eigenvalues):
                    pdos_weights[0:num_popn_orb,nb,nk,ns]=f.read_reals('>f8')
                    
                    #norm=np.sqrt(np.sum((pdos_weights[0:num_popn_orb,nb,nk,ns])**2))
                    norm=np.sum((pdos_weights[0:num_popn_orb,nb,nk,ns]))
                    pdos_weights[0:num_popn_orb,nb,nk,ns]=pdos_weights[0:num_popn_orb,nb,nk,ns]/norm
                    
        if species:
            num_species=len(np.unique(orbital_species))
            pdos_weights_sum=np.zeros((num_species,max_eigenvalues,num_kpoints,num_spins))
            
            for i in range(0,num_species):
                loc=np.where(orbital_species==i+1)[0]
                pdos_weights_sum[i,:,:,:]=np.sum(pdos_weights[loc,:,:,:],axis=0)
            pdos_weights_reorder=np.zeros((num_species,max_eigenvalues,len(bs.kpoints),num_spins))                        
            
        else:
            num_orbitals=4
            pdos_weights_sum=np.zeros((num_orbitals,max_eigenvalues,num_kpoints,num_spins))
            pdos_colours=np.zeros((3,max_eigenvalues,num_kpoints,num_spins))
            
            r=np.array([1,0,0])
            g=np.array([0,1,0])
            b=np.array([0,0,1])
            k=np.array([0,0,0])
            
            
            
            for i in range(0,num_orbitals):
                loc=np.where(orbital_l==i)[0]
                if len(loc)>0:
                
                    pdos_weights_sum[i,:,:,:]=np.sum(pdos_weights[loc,:,:,:],axis=0)
            pdos_weights_reorder=np.zeros((num_orbitals,max_eigenvalues,len(bs.kpoints),num_spins))                        

        #print(kpoints[1])
        #for nb in range(num_eigenvalues):
        #    print(pdos_weights_sum[:,nb,1,0])

        pdos_weights_sum=np.where(pdos_weights_sum>1,1,pdos_weights_sum)
        pdos_weights_sum=np.where(pdos_weights_sum<0,0,pdos_weights_sum)

        # reorder the thing


        for kp in range(len(bs.kpoints)):
            pdos_weights_reorder[:,:,kp,:]=pdos_weights_sum[:,:,bs.kpoint_map[kp],:]
            '''
            for n in range(len(pdos_weights_reorder[0,:,0,0])):        
                for s in range(len(pdos_weights_reorder[0,0,0,:])):
                    max_l=np.max(pdos_weights_reorder[:,n,kp,s])

                    if pdos_weights_reorder[0,n,kp,s]<max_l:
                        pdos_weights_reorder[0,n,kp,s]=0
                    if pdos_weights_reorder[1,n,kp,s]<max_l:
                        pdos_weights_reorder[1,n,kp,s]=0
                    if pdos_weights_reorder[2,n,kp,s]<max_l:
                        pdos_weights_reorder[2,n,kp,s]=0
                    if pdos_weights_reorder[3,n,kp,s]<max_l:
                        pdos_weights_reorder[3,n,kp,s]=0
            '''


        pdos_weights=np.zeros((max_eigenvalues,len(kpoints),num_spins))
        for kp in range(len(kpoints)):
            for n in range(max_eigenvalues):        
                for s in range(num_spins):
                    #print(pdos_weights_sum.shape,n,kp,s,len(kpoints))
                    max_l=np.where(pdos_weights_sum[:,n,kp,s]==np.max(pdos_weights_sum[:,n,kp,s]))[0]
                    #print(max_l)

                    pdos_weights[n,kp,s]=max_l
                    
    

            
                        
        #pdos_weights_reorder[1,:,:,:]=0
        np.savetxt(seed+".kpt_pdos",kpoints)
        return np.round(pdos_weights_reorder,13),kpoints,pdos_weights

    
    
    line_color="black"
    if background=="default" or background =="night" or background=="ParaView":
        line_color="white"
    if opacity>1 or opacity<0:
        print("\u001b[31mError: Invalid opacity\u001b[0m")
        sys.exit()
    
    # Aux functions
    def blockPrint():
        sys.stdout = open(os.devnull, 'w')
    def enablePrint():
        sys.stdout = sys.__stdout__
    
    
    def castep_read_out_sym(seed):
    
        out_cell=open(seed+"-out.cell","r")
        out_lines=out_cell.readlines()
        spec_grid=[1,1,1]
        for i in range(len(out_lines)):
            if "%BLOCK symmetry_ops" in out_lines[i]:
                start_line=i
            if "%ENDBLOCK symmetry_ops" in out_lines[i]:
                end_line=i
            if "spectral_kpoint_mp_grid" in out_lines[i] or "bs_kpoint_mp_grid" in out_lines[i]:
                spec_grid=np.array(out_lines[i].split()[-3:],dtype=float)
    
        n_ops=int((end_line-start_line-1)/5)
        rotations=np.zeros((n_ops,3,3))
        translations=np.zeros((n_ops,3))
    
        for i in range(n_ops):
            rotations[i,:,0]=[float(j) for j in out_lines[start_line+2+i*5].split()]
            rotations[i,:,1]=[float(j) for j in out_lines[start_line+3+i*5].split()]
            rotations[i,:,2]=[float(j) for j in out_lines[start_line+4+i*5].split()]
            translations[i,:]=[float(j) for j in out_lines[start_line+5+i*5].split()]
    
        return rotations,translations,spec_grid
    
       
    colours=cycle(("blue",'yellow','purple','pink','green','orange'))
    colours=cycle(('0081a7','eb8258','f6f740','3cdbd3','e3d7ff'))
    #colours=cycle(("FC9E4F",'4A0D67','50514F','59FFA0','FF8CC6'))
    if col!="default":
        cmap=plt.get_cmap(col)
        colours=cycle(cmap(np.linspace(0.1,1,5)))
    else:
        col="rainbow"
    
    #Open the files: Cell and bands
    blockPrint()
    try:
        cell=io.read(seed+".cell")
    except:
        raise Exception("No file "+seed+".cell")
    
    
    enablePrint()
    positions=cell.get_positions()
    numbers=cell.get_atomic_numbers()
    latt=cell.get_cell()
    atoms=np.unique(cell.get_chemical_symbols())[::-1]
    # Get the BZ information
    bril_zone=BZ.BZ(cell)
    recip_latt=bril_zone.recip_latt
    
    if species:
        n_cat=len(atoms)
    else:
        n_cat=4
        
    # Try and read a castep <seed>-out.cell, if not will have to use the ase symmetries
    try:
        symmetry=castep_read_out_sym(seed)
    except:
        print("Can't find <seed>-out.cell, proceeding with ASE, results may be inaccuracte")
        spacegroup=Spacegroup(bril_zone.sg)
        rot,trans=spacegroup.get_op()
        symmetry=(rot,trans,[1,1,1])
    
    
    # Get the bands information if needed
    if fermi:
        bs=bands.BandStructure(seed,recip_latt,np.array(latt),bril_zone.bz_vert,symmetry,prim,offset)

    
    # Set up the plotting stuff
    pv.set_plot_theme(background)
    
    
    if save:
        p=pv.Plotter(off_screen=True,lighting="three lights")
    else:
        p = pv.Plotter(lighting="three lights")
    p.enable_parallel_projection()
    
    
    #light = pv.Light()
    #light.set_direction_angle(30, 0)
    specular_power=3000
    specular=2
    ambient=0.3
    diffuse=1.0
    

    # Run the pdos if needed
    if pdos:
        pdos_weights,full_kp,pdos_norm=pdos_read(seed,species,bs)
    '''
        # Do some plotting
        up_ids=bs.up_ids[int(n_surf[0])]
        import matplotlib.pyplot as plt
        kpoints=bs.kpoints
        ids=[]
        kx=[]
        s=[]
        pl=[]
        d=[]
        f=[]
        print(kpoints)
        for k in range(len(kpoints)):

            #if np.isclose(kpoints[k][1],0.01345666) and np.isclose(kpoints[k][2],0.03814634):
            #if np.isclose(kpoints[k][1],-0.0745) and np.isclose(kpoints[k][2],-0.0447):
            if np.isclose(kpoints[k][1],-0.0) and np.isclose(kpoints[k][2],-0.0):
                                
                ids.append(k)
                kx.append(kpoints[k][0])
        for k in ids:
            s.append(pdos_weights[0,up_ids,k,0])
            pl.append(pdos_weights[1,up_ids,k,0])
            d.append(pdos_weights[2,up_ids,k,0])
            f.append(pdos_weights[3,up_ids,k,0])
        
        plt.plot(kx,s,label='s')
        plt.plot(kx,pl,label='p')
        plt.plot(kx,d,label='d')
        plt.plot(kx,f,label='f')
        plt.legend()
        plt.show(block=False)

        line_kpt=kpoints[ids]

        def lines_from_points(points):
            """Given an array of points, make a line set"""
            poly = pv.PolyData()
            poly.points = points
            cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
            cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
            cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
            poly.lines = cells
            return poly
        line=lines_from_points(line_kpt)
        line['scalars']=d
        line=line.tube(radius=0.005)
        p.add_mesh(line,cmap='Oranges')
    '''
    # Add box for BZ
    if not prim:
        for i in range(len(bril_zone.edges)):
            p.add_lines(bril_zone.edges[i],color=line_color,width=1.5)
    
    
    # Add prim box if wanted    
    edges  = np.array([[[0.,0.,0.],[0.,0.,1.]],
    		   [[0.,0.,0.],[0.,1.,0.]],
                       [[0.,0.,0.],[1.,0.,0.]],
    	           [[1.,1.,1.],[1.,0.,1.]],
                       [[1.,1.,1.],[1.,1.,0.]],
    	           [[1.,1.,1.],[0.,1.,1.]],
    	           [[0.,1.,1.],[0.,0.,1.]],
                       [[0.,1.,1.],[0.,1.,0.]],
                       [[1.,1.,0.],[1.,0.,0.]],
                       [[1.,1.,0.],[0.,1.,0.]],
    		   [[1.,0.,0.],[1.,0.,1.]],
    		   [[0.,0.,1.],[1.,0.,1.]]])
    
    prim_vert=np.array([[0,0,0],   # 0 0
                        [1,0,0],   # 1 1
                        [0,1,0],   # 3 2
                        [1,1,0],   # 2 3
                        [0,0,1],   # 4 4
                        [1,0,1],   # 5 5
                        [0,1,1],   # 7 6
                        [1,1,1]],dtype=float)  # 6 7
    prim_faces=np.hstack([[4,0,2,3,1], # bottom /
                          [4,0,1,5,4], # front / 
                          [4,0,2,6,4], # left /
                          [4,4,5,7,6], # top   /
                          [4,1,3,7,5], # right /
                          [4,2,3,7,6]]) # back /
    
    for i,main in enumerate(edges):
        for j,sub in enumerate(main):
            edges[i,j]=np.matmul(recip_latt.T,sub)
    
    if prim:
        for i in range(0,12):
            if not save:
                p.add_lines(edges[i],width=1.5,color=line_color)
            else:
                p.add_lines(edges[i],width=2.5,color=line_color)
    
    # mesh edges for making a boundary surface
    if prim:
        verts=np.append(edges[:,0],edges[:,1],axis=0)
        for i in range(len(prim_vert)):
            prim_vert[i]=np.matmul(recip_latt.T,prim_vert[i])
    
            verts=pv.PolyData(prim_vert,prim_faces)
    else:
        verts=np.append(bril_zone.edges[:,0],bril_zone.edges[:,1],axis=0)
    
        verts=np.unique(verts,axis=0)
        outer=bril_zone.bz_vert
        faces=[]
        
        for f in range(len(outer)):
            v_face=outer[f][0]
            faces.append(len(v_face))
            for i in range(len(v_face)):
                for j in range(len(verts)):
                    if np.allclose(verts[j],v_face[i]):
                        faces.append(j)
                        
    
        verts=pv.PolyData(verts,faces)
    
    
    # Add recip lattice vecs
    #axis_lab=np.array(["$k_x$","$k_y$","$k_z$"])
    #print("test")
    axis_lab=np.array(["Kx","Ky","Kz"])
    min_k=np.max(np.linalg.norm(recip_latt,axis=1))
    if show_axes:
        for i in range(0,3):
            l=np.linalg.norm(recip_latt[i])
            
            r=min_k/l
            arrow=pv.Arrow([0,0,0],0.75*recip_latt[i],shaft_radius=r*0.003,tip_radius=r*0.02,tip_length=r*0.05,scale="auto")
            p.add_mesh(arrow,color="blue")
        if show_labels:
            if not save:
                p.add_point_labels(0.75*recip_latt,axis_lab,shape=None,always_visible=True,show_points=False,font_family="courier",font_size=24,italic=True)
            else:
                p.add_point_labels(0.75*recip_latt,axis_lab,shape=None,always_visible=True,show_points=False,font_family="times",font_size=120)
                
    '''
    #Special points labels
    if plot_labels:
        p.add_point_labels(bril_zone.bz_points,bril_zone.bz_labels,shape=None,always_visible=True,show_points=False,font_size=24)
    
    if plot_paths:
        for i in range(len(bril_zone.bz_path)):
            p.add_lines(bril_zone.bz_path[i],color="red",width=1.5)
    # Check if we are metallic in any channel
    if fermi:
        fermi=bs.metal
        if not bs.metal:
            print('\033[93m Material is insulating, no Fermi surfaces to display.  \u001b[0m')
    '''




    
    # Test the kpoints
    if fermi:
        point_cloud = pv.PolyData(bs.kpoints)
        point_cloud = pv.PolyData(bs.kpt_irr)

        basis=[]
        n_colors=cycle(['#0000FF','#FF0000','#00FF00','black','purple','orange','yellow','cyan'])


        

        for n in range(n_cat):
            basis.append(np.array(colors.to_rgba(next(n_colors))))


            #if pdos:
            #    if species:
            #        print(atoms[n],":",basis[n])
            #    else:
            #        print(n,":",basis[n])
    
        # Get the kpoints
        kpoints=bs.kpoints

        # Check kpoint density
        if len(kpoints)<600 :
            print('\033[93m K-point density is relatively low, results may not be accurate..  \u001b[0m')
    
        
        # Calculate the spacing
        frac_spacing=1/np.array(symmetry[2])
        recip_spacing=np.matmul(recip_latt,frac_spacing)
        max_spacing=2*np.max(recip_spacing)
        if max_spacing>0.2:
            max_spacing=0.2
    
        # Get the spin_polarisation
        sp=bs.spin_polarised
        eV=1
        if sp:
            #Extract all the right stuff
            energy_up=bs.energy_up
            energy_down=bs.energy_down        
    
            
            #Number of fermi surfaces
            fermi_up=bs.n_fermi_up
            fermi_down=bs.n_fermi_down
            up_ids=bs.up_ids
            down_ids=bs.down_ids
    
    
            # Print the report
            print("+=========================================================+")
            print("| Electron   Spin   Min. (eV)  Max. (eV)   Bandwidth (eV) |")
            print("+=========================================================+")
            
            for i in range(len(up_ids)):
                print("|    {:04d}      up     {:6.3f}     {:6.3f}         {:6.3f}    |".format(up_ids[i],np.min(energy_up[:,i])*eV,np.max(energy_up[:,i])*eV,(np.max(energy_up[:,i])-np.min(energy_up[:,i]))*eV))
    
            for i in range(len(down_ids)):
                print("|    {:04d}    down     {:6.3f}     {:6.3f}         {:6.3f}    |".format(down_ids[i],np.min(energy_down[:,i])*eV,np.max(energy_down[:,i])*eV,(np.max(energy_down[:,i])-np.min(energy_down[:,i]))*eV))
    
            print("+=========================================================+")
            if dry:
                sys.exit()
            
            cloud=pv.PolyData(bs.kpoints)

            interp = cloud.delaunay_3d(alpha=100,progress_bar=verbose)
            total_vol=interp.volume
            #cloud['pdos']=pdos_norm[up_ids[0],:,0]
            #p.add_mesh(cloud,scalars='pdos')


            #test_cloud=pv.PolyData(full_kp)
            #test_cloud['pdos']=pdos_norm[up_ids[0],:,0]
            #p.add_mesh(test_cloud,scalars='pdos')


            
                
            # get the indices to plot
            if n_surf!=None:
                n_surf=np.array(n_surf,dtype=int)
            else:
                n_surf=range(np.max([fermi_up,fermi_down]))
            
            if show=="both" or show=="up":

                for i in range(fermi_up):
                    c=next(colours)
                    if i in n_surf:
                        interp.point_arrays["values"]=energy_up[:,i]
                        
                        contours=interp.contour([offset],scalars="values")
                        contours=contours.smooth(n_iter=smooth)
            
                        
                        if not prim:
                            for face in bril_zone.bz_vert:
                                origin=face[0][0]
                                direction=face[1]
                                contours=contours.clip(origin=origin,normal=direction)
                        
                        cont_vol=contours.volume
                        surf_vol=100*cont_vol/total_vol
                        if verbose:
                            print("%2d  up    %2.3f %% " %(i,surf_vol))
    
    
                        if surf_vol<5 and smooth>10:
                            print('\033[93m'+"Small Fermi surfaces may become distorted with large 'smooth' parameter, consider reducing.\u001b[0m")
    
                        try:    
                            if velocity:
                            
                                grad=interp.compute_derivative(scalars="values")
                                grad['Fermi Velocity (m/s)']=np.sqrt(np.sum(grad['gradient']**2,axis=1))*1.6e-19*1e-10/(1.05e-34)
                                
                                
                                std=np.std(grad['Fermi Velocity (m/s)'])
                                mean=np.mean(grad['Fermi Velocity (m/s)'])
                                above=np.where(grad['Fermi Velocity (m/s)']>mean+1*std)[0]
                                grad['Fermi Velocity (m/s)'][above]=0#mean+1*std
                                contours=contours.interpolate(grad,radius=max_spacing)
                                
                                
                                p.add_mesh(contours,scalars="Fermi Velocity (m/s)",cmap=col,smooth_shading=True,show_scalar_bar=True,lighting=True,pickable=False,specular=specular,specular_power=specular_power,ambient=ambient,diffuse=diffuse,opacity=opacity)
        
        
                            elif pdos:
                                cmap_array=np.zeros((len(kpoints),4))

                                for n in range(n_cat):
                                
                                    cmap_array[:,0]+=pdos_weights[n,up_ids[i],:,0]*basis[n][0]
                                    cmap_array[:,1]+=pdos_weights[n,up_ids[i],:,0]*basis[n][1]
                                    cmap_array[:,2]+=pdos_weights[n,up_ids[i],:,0]*basis[n][2]
                                cmap_array[:,3]=1
                                max=np.max(cmap_array[:,0:3],axis=1)
                                max_array=np.zeros((len(kpoints),3))
                                max_array[:,0]=max
                                max_array[:,1]=max
                                max_array[:,2]=max
                                cmap_array[:,0:3]=cmap_array[:,0:3]/max_array
                                
                                cmap_array=np.where(cmap_array>1,1,cmap_array)
        
        
                                cmap = ListedColormap(cmap_array)
                                z = np.linspace(0, 1, len(kpoints))
        
                                
                                interp.point_arrays["pdos"]=z
                                contours=interp.contour([offset],scalars="values")
                                #contours=contours.smooth(n_iter=smooth)
                                if not prim:
                                    for face in bril_zone.bz_vert:
                                        origin=face[0][0]
                                        direction=face[1]
                                        contours=contours.clip(origin=origin,normal=direction)
        
        
                                p.add_mesh(contours,scalars="pdos",cmap=cmap,smooth_shading=False,show_scalar_bar = False,lighting=True,pickable=False,specular=specular,specular_power=specular_power,ambient=ambient,diffuse=diffuse,opacity=opacity)
                                p.add_mesh_slice(interp,show_scalar_bar=False,cmap=cmap,show_edges=False,implicit=False)
                                
                            else:
                                p.add_mesh(contours,color=c,smooth_shading=True,show_scalar_bar = False,lighting=True,pickable=False,specular=specular,specular_power=specular_power,ambient=ambient,diffuse=diffuse,opacity=opacity)
                        except:
                                pass
    
            if show=="both" or show=="down":        

                for i in range(fermi_down):
                    c=next(colours)
                    if i in n_surf:
                        interp.point_arrays["values"]=energy_down[:,i]
                        contours = interp.contour([offset])
                        contours=contours.smooth(n_iter=smooth)
                        cont_vol=contours.volume
                        surf_vol=100*cont_vol/total_vol
                        cont_vol=contours.volume
                        surf_vol=100*cont_vol/total_vol
                        if verbose:
                            print("%2d  down  %2.3f %%" %(i,surf_vol))
                        if surf_vol<5 and smooth>10:
                            print('\033[93m'+"Small Fermi surfaces may become distorted with large 'smooth' parameter, consider reducing.\u001b[0m")
    
                        if not prim:
                            for face in bril_zone.bz_vert:
                                origin=face[0][0]
                                direction=face[1]
                                contours=contours.clip(origin=origin,normal=direction)
                        try:
                            if velocity:
                                grad=interp.compute_derivative(scalars="values")
                                grad['Fermi Velocity (m/s)']=np.sqrt(np.sum(grad['gradient']**2,axis=1))*1.6e-19*1e-10/(1.05e-34)
                                
                                std=np.std(grad['Fermi Velocity (m/s)'])
                                mean=np.mean(grad['Fermi Velocity (m/s)'])
                                above=np.where(grad['Fermi Velocity (m/s)']>mean+1*std)[0]
                                grad['Fermi Velocity (m/s)'][above]=0#mean+1*std
                                contours=contours.interpolate(grad,radius=max_spacing)
                                
                                
                                p.add_mesh(contours,scalars="Fermi Velocity (m/s)",cmap=col,smooth_shading=True,show_scalar_bar=True,lighting=True,specular=specular,specular_power=specular_power,ambient=ambient,diffuse=diffuse,opacity=opacity)

                            elif pdos:


                                cmap_array=np.zeros((len(kpoints),4))

                                for n in range(n_cat):
                                
                                    cmap_array[:,0]+=pdos_weights[n,up_ids[i],:,0]*basis[n][0]
                                    cmap_array[:,1]+=pdos_weights[n,up_ids[i],:,0]*basis[n][1]
                                    cmap_array[:,2]+=pdos_weights[n,up_ids[i],:,0]*basis[n][2]
                                cmap_array[:,3]=1
                                max=np.max(cmap_array[:,0:3],axis=1)
                                max_array=np.zeros((len(kpoints),3))
                                max_array[:,0]=max
                                max_array[:,1]=max
                                max_array[:,2]=max
                                cmap_array[:,0:3]=cmap_array[:,0:3]/max_array
                                
                                cmap_array=np.where(cmap_array>1,1,cmap_array)


                                cmap = ListedColormap(cmap_array)
                                z = np.linspace(0, 1, len(kpoints))

                            
                                #interp.point_arrays["pdos"]=z
                                interp.point_arrays["pdos"]=pdos_weights[1,up_ids[i],:,0]#z
                                contours=interp.contour([offset],scalars="values")
                                #contours=contours.smooth(n_iter=smooth)
                                if not prim:
                                    for face in bril_zone.bz_vert:
                                        origin=face[0][0]
                                        direction=face[1]
                                        contours=contours.clip(origin=origin,normal=direction)
                                
                                
                                p.add_mesh(contours,scalars="pdos",cmap='Oranges',smooth_shading=True,show_scalar_bar = True,lighting=True,pickable=False,specular=specular,specular_power=specular_power,ambient=ambient,diffuse=diffuse,opacity=opacity,pbr=True)
                                #p.add_mesh_slice(interp,show_scalar_bar=False,cmap=cmap,show_edges=False,implicit=False)
                            else:
                                p.add_mesh(contours,color=c,smooth_shading=True,show_scalar_bar = False,lighting=True,pickable=False,specular=specular,specular_power=specular_power,ambient=ambient,diffuse=diffuse,opacity=opacity)
    
                        except:
                            pass
                
    
        else:
            energy_up=bs.energy_up
            fermi_up=bs.n_fermi
            up_ids=bs.up_ids
    
            print("+=========================================================+")
            print("| Electron   Spin   Min. (eV)  Max. (eV)   Bandwidth (eV) |")
            print("+=========================================================+")
     
            for i in range(len(up_ids)):
                print("|    %04d      up     %2.3f     %2.3f         %2.3f      |"%(up_ids[i],np.min(energy_up[:,i])*eV,np.max(energy_up[:,i])*eV,(np.max(energy_up[:,i])-np.min(energy_up[:,i]))*eV))
    
    
            print("+=========================================================+")
                
    
            cloud=pv.PolyData(kpoints)
            interp = cloud.delaunay_3d(progress_bar=verbose)
            total_vol=interp.volume
    
            interp.point_arrays["values"]=energy_up[:,0]
            contours = interp.contour([0.])
    
            if n_surf!=None:
                n_surf=np.array(n_surf,dtype=int)
            else:
                n_surf=range(fermi_up)
                
    
            if show=="both" or show=="up":
                for i in range(fermi_up):
                    c=next(colours)
                    if i in n_surf:
                        interp.point_arrays["values"]=energy_up[:,i]
     
                        contours=interp.contour([offset],scalars="values")
                        contours=contours.smooth(n_iter=smooth)
                        
                        if not prim:
                            for face in bril_zone.bz_vert:
                                origin=face[0][0]
                                direction=face[1]
                                contours=contours.clip(origin=origin,normal=direction)
     
                        cont_vol=contours.volume
                        surf_vol=100*cont_vol/total_vol
                        if verbose:
                            print("%2d  up    %2.3f %% " %(i,surf_vol))
    
    
                        if surf_vol<5 and smooth>10:
                            print('\033[93m'+"Small Fermi surfaces may become distorted with large 'smooth' parameter, consider reducing.\u001b[0m")
    
    
    
                        #try:    
                        if velocity:
                            
                            grad=interp.compute_derivative(scalars="values")
                            grad['Fermi Velocity (m/s)']=np.sqrt(np.sum(grad['gradient']**2,axis=1))*1.6e-19*1e-10/(1.05e-34)
                            
                            
                            std=np.std(grad['Fermi Velocity (m/s)'])
                            mean=np.mean(grad['Fermi Velocity (m/s)'])
                            above=np.where(grad['Fermi Velocity (m/s)']>mean+1*std)[0]
                            grad['Fermi Velocity (m/s)'][above]=0#mean+1*std
                            contours=contours.interpolate(grad,radius=max_spacing)
                            
                            
                            p.add_mesh(contours,scalars="Fermi Velocity (m/s)",cmap=col,smooth_shading=True,show_scalar_bar=True,lighting=True,pickable=False,specular=specular,specular_power=specular_power,ambient=ambient,diffuse=diffuse,opacity=opacity)
                            
                        elif pdos:

                            
                            
                            cmap_array=np.zeros((len(kpoints),4))
                            for n in range(n_cat):
                                
                                cmap_array[:,0]+=pdos_weights[n,up_ids[i],:,0]*basis[n][0]
                                cmap_array[:,1]+=pdos_weights[n,up_ids[i],:,0]*basis[n][1]
                                cmap_array[:,2]+=pdos_weights[n,up_ids[i],:,0]*basis[n][2]
                            cmap_array[:,3]=1
                            max=np.max(cmap_array[:,0:3],axis=1)
                            max_array=np.zeros((len(kpoints),3))
                            max_array[:,0]=max
                            max_array[:,1]=max
                            max_array[:,2]=max
                            cmap_array[:,0:3]=cmap_array[:,0:3]/max_array

                            cmap_array=np.where(cmap_array>1,1,cmap_array)

                            
                            cmap = ListedColormap(cmap_array)
                            z = np.linspace(0, 1, len(kpoints))
                            

                            
                            interp.point_arrays["pdos"]=pdos_weights[3,up_ids[i],:,0]#z

                            
                            contours=interp.contour([offset],scalars="values")
                            #contours=contours.smooth(n_iter=smooth)
                            if not prim:
                                for face in bril_zone.bz_vert:
                                    origin=face[0][0]
                                    direction=face[1]
                                    contours=contours.clip(origin=origin,normal=direction)
                                        

                            p.add_mesh(contours,scalars="pdos",cmap='Oranges',smooth_shading=True,show_scalar_bar = True,lighting=True,pickable=False,specular=specular,specular_power=specular_power,ambient=ambient,diffuse=diffuse,opacity=opacity)
                            #p.add_mesh_slice(interp,show_scalar_bar=False,cmap=cmap,show_edges=False,implicit=False)
                                
                        else:
                            p.add_mesh(contours,color=c,smooth_shading=True,show_scalar_bar = False,lighting=True,pickable=False,specular=specular,specular_power=specular_power,ambient=ambient,diffuse=diffuse,opacity=opacity)

                         #except:
                        #    pass
    
    
    
    p.window_size = 1000, 1000
    
    if prim:
        focus=np.matmul(recip_latt.T,[0.5,0.5,0.5])
    else:
        focus=[0,0,0]
    if show_faces:
        p.add_mesh(verts,opacity=0.3,color="grey",smooth_shading=True,specular=specular,specular_power=specular_power,ambient=ambient,diffuse=diffuse,lighting=True)
            
    p.set_focus(focus)
    p.view_isometric()
    
    def button_a():
        o=recip_latt[2]
        vpvec=recip_latt[0]/np.linalg.norm(recip_latt[0])    
        vp=focus+15*vpvec
        p.camera_position=[vp,focus,o]
        
        
    def button_b():
        
        o=recip_latt[2]
        vpvec=recip_latt[1]/np.linalg.norm(recip_latt[1])
        vp=focus+15*vpvec
        p.camera_position=[vp,focus,o]
        
        
    def button_c():
        
        o=recip_latt[1]
        vpvec=recip_latt[2]/np.linalg.norm(recip_latt[2])
        
        vp=focus+15*vpvec
        p.camera_position=[vp,focus,o]
    
    def button_sd():
        o=recip_latt[2]
        T=0.9*recip_latt[0]+0.4*recip_latt[1]+0.6*recip_latt[2]
        vpvec=T/np.linalg.norm(T)
        vp=focus+15*vpvec
        p.camera_position=[vp,focus,o]
    p.add_key_event("a",button_a)
    p.add_key_event("b",button_b)
    p.add_key_event("c",button_c)
    p.add_key_event("o",button_sd)

    if not prim:
        o=recip_latt[2]
        T=0.9*recip_latt[0]+0.4*recip_latt[1]+0.6*recip_latt[2]
        vpvec=T/np.linalg.norm(T)
        vp=focus+100*vpvec
        p.camera_position=[vp,focus,o]

    if np.sum(cam_pos)!=0:
        vp=(cam_pos[0],cam_pos[1],cam_pos[2])
        o=(cam_pos[3],cam_pos[4],cam_pos[5])
        p.camera_position=[vp,focus,o]
    try:
        p.camera.zoom(z)
    except:
        print("Zoom not implemented in this version of PyVista.")

    
    
    end_time=time.time()-start_time
    print("Time %3.3f s"%end_time)
    
    if save:
        p.window_size=[5000,5000]
        p.show(title=seed,screenshot=seed+"_BZ.png")
    else:
        p.show(title=seed,auto_close=False)
        
    if verbose:
        print("Final Camera Position:")
        print(p.camera_position[0][0],p.camera_position[0][1],p.camera_position[0][2],p.camera_position[2][0],p.camera_position[2][1],p.camera_position[2][2])
    

    #path = p.generate_orbital_path(factor=2.0, shift=10000, viewup=viewup, n_points=36)
    #p.open_gif("orbit.gif")
    #p.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1])
    if gif:
        gif_time=time.time()
        print("Writing movie to %s.gif..."%seed)
        
        path = p.generate_orbital_path(n_points=50, shift=0,viewup=vp)

        p.open_gif(seed+".gif")
        p.orbit_on_path(path, write_frames=True,focus=focus)
        p.close()
        movie_time=time.time()-gif_time
        print("Export time %3.3f s"%movie_time)
if __name__=='__main__':
    main()

import numpy as np
import sys

class BandStructure:
    '''Class containing bands information for calculating fermi surfaces'''
    def __init__(self,seed,recip_cell,cell,vert,sym,prim):
        
        # Calculate the plane info
        planes=np.zeros((len(vert),3))
        ds=np.zeros((len(vert)))
        for i,face in enumerate(vert):
            lface=face[0]
            n=face[1]
            planes[i]=n
            ds[i]=-1.3*np.dot(n,lface[0])

            
        def is_outside(points,planes,ds):
            #make the loci_r mat
            test_array=np.zeros((len(points),len(planes)))
            for i in range(len(points)):
                for j in range(len(planes)):
                    test_array[i,j]=np.dot(points[i],planes[j])+ds[j]
                
            test_array=np.sign(test_array+1e-8)
            sum_array=np.sum(test_array,axis=1)
            return sum_array==-len(planes)
            
        eV=27.2114
        
        # First we try to open the file 
        
        # Open the bands file
        try:
            bands_file=seed+".bands"
            bands=open(bands_file,'r')
        except:
            raise Exception("No .bands file")

        lines=bands.readlines()

        no_spins=int(lines[1].split()[-1])
        no_kpoints=int(lines[0].split()[-1])
        fermi_energy=float(lines[4].split()[-1])
        
        if no_spins==1:
            fermi_energy=float(lines[4].split()[-1])
            no_electrons =float(lines[2].split()[-1])
            no_eigen  = int(lines[3].split()[-1])
            no_eigen_2=None
            spin_polarised=False
        if no_spins==2:
            spin_polarised=True
            no_eigen  = int(lines[3].split()[-2])
            no_eigen_2=int(lines[3].split()[-1])
            n_up=float(lines[2].split()[-2])
            n_down=float(lines[2].split()[-1])
        # Set all of the bands information
        self.spin_polarised=spin_polarised
        self.Ef=fermi_energy
        self.n_kpoints=no_kpoints
        if spin_polarised:
            self.nup=n_up
            self.ndown=n_down
            self.electrons=n_up+n_down
        else:
            self.nup=None
            self.ndown=None
            self.electrons=no_electrons
        self.eig_up=no_eigen
        self.eig_down=no_eigen_2
        self.n_kpoints=no_kpoints

        
        rot,trans,spec_grid=sym
        
        kpoints=np.zeros((no_kpoints,3))
        if not spin_polarised:
            temp_kpts=lines[9:-1:no_eigen+2]
            for i in range(len(temp_kpts)):
                temp_kpts_array=np.array(temp_kpts[i].split()[2:5],dtype=float)
                #temp_kpts_array=np.matmul(recip_cell.T,temp_kpts_array)
                kpoints[i]=temp_kpts_array
                #kpoints[i+no_kpoints]=-temp_kpts_array
        else:
            temp_kpts=lines[9:-1:no_eigen+no_eigen_2+3]
            for i in range(len(temp_kpts)):
                temp_kpts_array=np.array(temp_kpts[i].split()[2:5],dtype=float)
                #temp_kpts_array=np.matmul(recip_cell.T,temp_kpts_array)
                kpoints[i]=temp_kpts_array
                #kpoints[i+no_kpoints]=-temp_kpts_array

        kpoints=np.array(kpoints)
                
        
        unfold_kpoints=[]#np.zeros((2*no_kpoints,3))  
        kpoint_map=[]
        
        # Define the recip lattice vecs
        kx=recip_cell[0]
        ky=recip_cell[1]
        kz=recip_cell[2]
        k_len=np.array([np.linalg.norm(kx),np.linalg.norm(ky),np.linalg.norm(kz)])

        
        
        for i in range(len(trans)):
            trans[i]=np.matmul(recip_cell.T,trans[i])

        for i in range(len(rot)):
            #for i in [0]:
            temp_mat=np.matmul(np.matmul(cell,rot[i]),recip_cell.T)#/(2*np.pi)
            temp_mat=np.round(temp_mat)

            for j in range(len(kpoints)):
                ks=kpoints[j]

                ks=np.matmul(temp_mat,ks)
 
                ks = ks - np.floor(ks)
                ks=np.matmul(recip_cell.T,ks)
                
                unfold_kpoints.append(ks)
                kpoint_map.append(j)

        # Now we have them unfolded by rotations, we need to translate by the reciprocal lattice vectors
        unfold_kpoints=np.array(unfold_kpoints)
        kpoint_map=np.array(kpoint_map)
        kpt_copy=unfold_kpoints
        map_copy=np.array(kpoint_map)


        if not prim:
            for i in [-1,0]:
                for j in [-1,0]:
                    for l in [-1,0]:
                        if i==0 and j==0 and l==0: continue
                        T=kpt_copy+i*kx+j*ky+l*kz
                        unfold_kpoints=np.append(unfold_kpoints,T,axis=0)
                        kpoint_map=np.append(kpoint_map,map_copy)
            
        
        # Start masking
        # Find the unique ones
        unfold_kpoints=np.round(unfold_kpoints,4)

        uni_ind,ind=np.unique(unfold_kpoints,axis=0,return_index=True)
        unfold_kpoints=unfold_kpoints[ind]
        kpoint_map=np.array(kpoint_map)[ind]

        # Distance of the points from Gamma
        r=np.sqrt(np.sum(unfold_kpoints**2,axis=1))
        mask=(r<2*np.max(k_len)/3)
        
        # Gets rid of anything too far away
        if not prim:
            unfold_kpoints=unfold_kpoints[mask]
            kpoint_map=np.array(kpoint_map)[mask]
            r=r[mask]
        

        # Cut outside BZ
        if not prim:
            empty_mask=is_outside(unfold_kpoints,planes,ds)
            unfold_kpoints=unfold_kpoints[empty_mask]
            kpoint_map=kpoint_map[empty_mask]
        
        # After the reducing, add in the negatives, need moving if prim
        if not prim:
            unfold_kpoints=np.append(unfold_kpoints,-unfold_kpoints,axis=0)
            kpoint_map=np.append(kpoint_map,kpoint_map)
        else:
            unfold_kpoints=np.append(unfold_kpoints,-unfold_kpoints+kx+ky+kz,axis=0)
            kpoint_map=np.append(kpoint_map,kpoint_map)

        
        # Put the folded ones here
        folded=[]
        temp_mat=np.matmul(cell.T,recip_cell)
        temp_mat=np.round(temp_mat)
        for j in range(len(kpoints)):
            ks=kpoints[j]
            if ks[0]<0: ks[0]=1+ks[0]
            if ks[1]<0: ks[1]=1+ks[1]
            if ks[2]<0: ks[2]=1+ks[2]
            #ks=np.matmul(temp_mat,kpoints[j])
            ks=np.matmul(recip_cell.T,ks)
            folded.append(ks)


        self.kpt_irr=folded
        #print("kpts: ",len(unfold_kpoints))
        nkpts_unfolded=len(unfold_kpoints)
        self.nkpts_unfolded=len(unfold_kpoints)
    

        # Extract the energy for each kpoint

        if not spin_polarised:
            energy_array=np.zeros((no_eigen,no_kpoints))
            fermi_map=np.zeros((no_eigen),dtype=bool)
            electron_ids=[]
            for i in range(no_eigen-1):                
                energy_array[i]=lines[11+i:-1:no_eigen+2]
                energy_array[i]=energy_array[i]-fermi_energy
                if np.max(energy_array[i])>0 and  np.min(energy_array[i])<0:
                    electron_ids.append(i)
                  
                    fermi_map[i]=True

            #if np.sum(fermi_map)==0:
            #    fermi_map[int(no_electrons)-1]=True
            energy_array=energy_array[fermi_map]
            #print("Number of Fermi surfaces: ",len(energy_array))
            n_fermi=len(energy_array)
            self.n_fermi=n_fermi


            #Transpose the energy array to make it kpoint oriented
            energy_array=energy_array.T
            unfolded_energy_up=np.zeros((nkpts_unfolded,n_fermi))

            for i in range(nkpts_unfolded):
                unfolded_energy_up[i]=energy_array[kpoint_map[i]]

            self.energy_up=unfolded_energy_up*eV
            self.energy_down=None
            self.ids=electron_ids


            
        else:
            energy_array=np.zeros((no_eigen,no_kpoints))
            energy_array_do=np.zeros((no_eigen_2,no_kpoints))
            fermi_map=np.zeros((no_eigen),dtype=bool)
            fermi_map_do=np.zeros((no_eigen_2),dtype=bool)
            up_ids=[]
            for i in range(no_eigen-1):
                energy_array[i]=lines[11+i:-1:no_eigen+3+no_eigen_2]
                energy_array[i]=energy_array[i]-fermi_energy                
                if np.max(energy_array[i])>0 and np.min(energy_array[i])<0:
                    fermi_map[i]=True
                    up_ids.append(i)
            #if np.sum(fermi_map)==0:
            #    fermi_map[int(n_up)-1]=True
            energy_array=energy_array[fermi_map]
            
            
            n_fermi_up=len(energy_array)
            down_ids=[]
            for i in range(no_eigen_2-1):
                energy_array_do[i]=lines[12+i+no_eigen:-1:no_eigen+3+no_eigen_2]
                energy_array_do[i]=energy_array_do[i]-fermi_energy

                if np.max(energy_array_do[i])>0 and  np.min(energy_array_do[i])<0:
                    fermi_map_do[i]=True
                    down_ids.append(i)
            #if np.sum(fermi_map_do)==0:
            #    fermi_map_do[int(n_down)-1]=True
            energy_array_do=energy_array_do[fermi_map_do]
            
            n_fermi_down=len(energy_array_do)


            self.n_fermi_up=n_fermi_up
            self.n_fermi_down=n_fermi_down
            self.n_fermi=None
            self.up_ids=up_ids
            self.down_ids=down_ids
            #Transpose the energy array to make it kpoint oriented
            energy_array=energy_array.T
            energy_array_do=energy_array_do.T


            unfolded_energy_up=np.zeros((nkpts_unfolded,n_fermi_up))
            unfolded_energy_down=np.zeros((nkpts_unfolded,n_fermi_down))

            for i in range(nkpts_unfolded):
                unfolded_energy_up[i]=energy_array[kpoint_map[i]]
                unfolded_energy_down[i]=energy_array_do[kpoint_map[i]]


            self.energy_up=unfolded_energy_up*eV
            self.energy_down=unfolded_energy_down*eV
            

        self.kpoints=unfold_kpoints
        self.kpoint_map=kpoint_map
        if spin_polarised:
            if self.n_fermi_up+self.n_fermi_down==0:
                self.metal=False
            else:
                self.metal=True
        else :
            if self.n_fermi==0:
                self.metal=False
            else:
                self.metal=True
            

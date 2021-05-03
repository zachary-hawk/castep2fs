import numpy as np
import warnings
from ase.spacegroup import Spacegroup
from ase.build  import cut
import ase
import ase.dft.bz as bz

class BZ:
    '''Class containing all of the reciprocal lattice information'''
    def __init__(self, cell):
        warnings.filterwarnings("ignore")
        def primitive_from_conventional_cell(atoms):
            """Returns primitive cell given an Atoms object for a conventional
            cell and it's spacegroup."""
            spacegroup=ase.spacegroup.get_spacegroup(atoms)
            sg = Spacegroup(spacegroup)
            prim_cell = sg.scaled_primitive_cell  # Check if we need to transpose
            return cut(atoms, a=prim_cell[0], b=prim_cell[1], c=prim_cell[2])


        # Get the bv lattice
        bv_latt=cell.cell.get_bravais_lattice()

        vertices=bz.bz_vertices(cell.cell.reciprocal())
        edges=[]
        vert=[]
        for face in vertices:
            lface=face[0]
            for j in range(1,len(lface)):
                edge=[lface[j],lface[j-1]]
                edges.append(edge)
                vert.append(lface[j])
            edges.append([lface[0],lface[-1]])
            vert.append(lface[0])
        edges=np.array(edges)
        self.edges=edges
        self.vertices=vert
        self.bz_vert=vertices

        #Set the spacegroup
        self.sg=ase.spacegroup.get_spacegroup(cell)
        prim_cell=cell.get_cell()#primitive_from_conventional_cell(cell).get_cell()
        recip_latt=np.zeros((3,3))
        recip_latt[0]=np.cross(prim_cell[1],prim_cell[2])/np.dot(prim_cell[0],np.cross(prim_cell[1],prim_cell[2]))
        recip_latt[1]=np.cross(prim_cell[2],prim_cell[0])/np.dot(prim_cell[0],np.cross(prim_cell[1],prim_cell[2]))
        recip_latt[2]=np.cross(prim_cell[0],prim_cell[1])/np.dot(prim_cell[0],np.cross(prim_cell[1],prim_cell[2]))

        self.recip_latt=recip_latt
        
        
        # Special points
        scaled_points=bv_latt.get_special_points_array()
        special_point_names=bv_latt.special_point_names
        for i in range(len(scaled_points)):
            scaled_points[i]=np.matmul(recip_latt.T,scaled_points[i])        

            if special_point_names[i]=="G":
                special_point_names[i]="$\Gamma$"
            else:
                special_point_names[i]="$"+special_point_names[i]+"$"
        bz_path=[]
        for i in range(len(scaled_points)):
            for j in range(len(scaled_points)):
                if i==j:
                    continue
                bz_path.append([scaled_points[i],scaled_points[j]])
        
        self.bz_points=scaled_points
        self.bz_labels=special_point_names
        self.bz_path=np.array(bz_path)



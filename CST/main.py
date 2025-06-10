import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import os

from nodes import Node
from elements import CST
from solve import Solve


def modificar_geo(archivo_entrada, archivo_salida, nuevo_n, nuevo_r):
    with open(archivo_entrada, "r") as f:
        lineas = f.readlines()

    nuevas_lineas = []
    for linea in lineas:
        if linea.strip().startswith("n ="):
            nuevas_lineas.append(f"n = {nuevo_n};\n")
        elif linea.strip().startswith("r ="):
            nuevas_lineas.append(f"r = {nuevo_r};\n")
        elif "Transfinite Curve{1, 2}" in linea:
            nuevas_lineas.append(f"Transfinite Curve{{1, 2}} = {nuevo_n} Using Progression {nuevo_r};\n")
        elif "Transfinite Curve{4, 3}" in linea:
            nuevas_lineas.append(f"Transfinite Curve{{4, 3}} = {nuevo_n} Using Progression 1/{nuevo_r};\n")
        else:
            nuevas_lineas.append(linea)

    with open(archivo_salida, "w") as f:
        f.writelines(nuevas_lineas)


def load_mesh_objects(geo_file="geo.geo", msh_file="mesh.msh"):
    gmsh.initialize()
    gmsh.open(geo_file)
    gmsh.model.mesh.generate(2)
    gmsh.write(msh_file)
    gmsh.finalize()

    # Leer malla
    mesh = meshio.read(msh_file)

    # Crear nodos
    nodes = [Node(i+1, x, y) for i, (x, y, _) in enumerate(mesh.points)]

    # Crear elementos CST
    cst_elements = []
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            for i, node_ids in enumerate(cell_block.data):
                node_ids = [int(id) + 1 for id in node_ids]  # Convertir a índices base 1
                
                cst_elements.append(CST(i+1 , list(node_ids)))
            break

    # Detectar nodos en líneas físicas ("Diritchlet 1" a "Diritchlet 4")
    boundary_nodes = {1: set(), 2: set(), 3: set(), 4: set()}

    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type == "line":
            physical_ids = mesh.cell_data_dict['gmsh:physical']['line']
            for line, phys_id in zip(cell_block.data, physical_ids):
                if phys_id in boundary_nodes:
                    for node_id in line:
                        boundary_nodes[phys_id].add(int(node_id))

    # Añadir etiquetas de borde a los nodos
    for node in nodes:
        node.boundary_label = []
        for label_id, node_set in boundary_nodes.items():
            if node.id in node_set:
                node.boundary_label.append(f"Diritchlet Boundary")

    return nodes, cst_elements

def main(N, R, alpha):   

    Estructure = None
    nodes = None
    elements = None

    geo_file = "CST/geo.geo"
    mesh_file = "CST/mesh.msh"

    #En primero lugar modifico el archivo geo
    modificar_geo(geo_file, geo_file, N, R)

    #Genero la malla
    nodes, elements = load_mesh_objects(geo_file=geo_file, msh_file=mesh_file)

    #Obtengo la solucion numerica por nodo
    for node in nodes:
        node.solve_u(alpha)

    #Resuelvo la estructura
    Estructure = Solve(nodes, elements, alpha)

    Estructure.solve_matrix()

    #errores = error(nodes)
    solucion_analitica = Estructure.semi_norm_H1_0()
    print(f"Solución analítica: {solucion_analitica}")
    solucion_fem = Estructure.femm_solution()
    print(f"Solución FEM: {solucion_fem}")

    error = np.abs(solucion_analitica - solucion_fem)

    # Guardar en .txt

    with open("CST/resultados.txt", "a") as f:
        f.write(f"N = {N}, R = {R}, alpha = {alpha}\n")
        f.write(f"Error: {error:.6e}\n")
        f.write("-" * 40 + "\n")

    print("Resultados guardados en CST/resultados.txt")

    

if __name__ == "__main__":

    open("CST/resultados.txt", "w").close()

    N = []
    for i in range(40):
        N.append(i + 10)

    R = []
    for i in range(10):
        R.append(1.0 + (i) * 0.05)

    alpha = 0.1

    for n in N:
        for r in R:
            main(n, r, alpha)
   
       
        
    

    
    
    
    

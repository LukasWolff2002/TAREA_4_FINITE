import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import os

from nodes import Node
from elements import LST
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

    print(f"Archivo guardado como '{archivo_salida}' con n = {nuevo_n} y r = {nuevo_r}")

def fixed_load_mesh_objects(geo_file="geo.geo", msh_file="mesh.msh"):
    gmsh.initialize()
    gmsh.open(geo_file)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(2)

    # Obtener todos los nodos y sus coordenadas
    all_tags, all_coords, _ = gmsh.model.mesh.getNodes()
    all_coords = all_coords.reshape(-1, 3)
    tag_to_xy = {tag: (round(x, 8), round(y, 8)) for tag, (x, y, _) in zip(all_tags, all_coords)}

    # Leer archivo de malla en paralelo
    mesh = meshio.read(msh_file)

    # Crear nodos con índices base 1
    nodes = [Node(i + 1, x, y) for i, (x, y, _) in enumerate(mesh.points)]

    # Mapear coordenadas → índice base 1
    coord_to_index = {(round(x, 8), round(y, 8)): i + 1 for i, (x, y, _) in enumerate(mesh.points)}

    # Buscar nodos de borde
    boundary_nodes = {}
    for dim in [1]:
        phys_groups = gmsh.model.getPhysicalGroups(dim)
        for _, tag in phys_groups:
            node_indices = set()
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            for entity in entities:
                element_type, element_tags, node_tags = gmsh.model.mesh.getElements(dim, entity)
                for nlist in node_tags:
                    for node_id in nlist:
                        node_indices.add(node_id)  # usamos el ID original, no mapeo por coordenadas
            boundary_nodes[tag] = node_indices

    gmsh.write(msh_file)
    gmsh.finalize()

    # Crear elementos LST
    lst_elements = []
    for cell_block in mesh.cells:
        if cell_block.type in ["triangle6", "triangle"]:
            for i, node_ids in enumerate(cell_block.data):
                if len(node_ids) == 6:
                    node_ids = [int(id) + 1 for id in node_ids]
                    lst_elements.append(LST(i + 1, node_ids))

    # Asignar etiquetas
    for node in nodes:
        node.boundary_label = []
        for tag, ids in boundary_nodes.items():
            if node.id in ids:
                node.boundary_label.append(f"Dirichlet")

    return nodes, lst_elements

def main(N, R, alpha):   

    Estructure = None
    nodes = None
    elements = None

    geo_file = "LST/geo.geo"
    mesh_file = "LST/mesh.msh"

    #En primero lugar modifico el archivo geo
    modificar_geo(geo_file, geo_file, N, R)

    #Genero la malla
    nodes, elements = fixed_load_mesh_objects(geo_file=geo_file, msh_file=mesh_file)

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

    with open("LST/resultados.txt", "a") as f:
        f.write(f"N = {N}, R = {R}, alpha = {alpha}\n")
        f.write(f"Error: {error:.6e}\n")
        f.write("-" * 40 + "\n")

    print("Resultados guardados en LST/resultados.txt")

    

if __name__ == "__main__":

    open("LST/resultados.txt", "w").close()

    N = []
    for i in range(100):
        N.append(i + 15)

    R = []
    for i in range(10):
        R.append(1.0 + (i) * 0.05)

    alpha = 0.1

    for n in N:
        for r in R:
            main(n, r, alpha)
   
       
        
    

    
    
    
    

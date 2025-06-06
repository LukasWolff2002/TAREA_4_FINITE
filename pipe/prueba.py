import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import meshio

def leer_malla_msh(filename):
    """
    Lee un archivo mesh.msh con meshio y extrae:
    - coords: array (N,2) con (x,y) de cada nodo
    - elems: array (M,3) con índices de nodos P1 que forman cada triángulo
    - bordes: lista de índices de nodos en la frontera x=0, x=1, y=0 ó y=1
    """
    # 1) Leer con meshio
    malla = meshio.read(filename)
    # 2) Extraer las coordenadas (meshio guarda puntos en 3D, tomamos x,y)
    pts = malla.points   # forma (N,3) o (N,2)
    coords = pts[:, :2]  # (x,y)
    # 3) Extraer sólo los triángulos (tipo "triangle")
    #    meshio almacena celdas en 'cells' o en 'cells_dict'
    if "triangle" in malla.cells_dict:
        elems = malla.cells_dict["triangle"]
    else:
        # En versiones antiguas: recorrer malla.cells, buscar type=="triangle"
        elems_list = []
        for bloque in malla.cells:
            if bloque.type.lower() == "triangle":
                elems_list.append(bloque.data)
        if len(elems_list) == 0:
            raise ValueError("No se encontraron elementos tipo 'triangle' en el .msh.")
        elems = np.vstack(elems_list)
    elems = np.array(elems, dtype=int)
    # 4) Determinar los nodos de frontera: aquellos con x≈0 ó x≈1 ó y≈0 ó y≈1
    tol = 1e-8
    bordes = []
    for i, (x, y) in enumerate(coords):
        if abs(x - 0.0) < tol or abs(x - 1.0) < tol or abs(y - 0.0) < tol or abs(y - 1.0) < tol:
            bordes.append(i)
    bordes = np.array(bordes, dtype=int)
    return coords, elems, bordes

# ------------------------------
# Manufactured solution y f(x)
# ------------------------------
def u_exact(x, alpha):
    """
    Solución manufacturada: u(x) = ||x||^alpha
    x: array shape (n_points, 2)
    alpha: exponente > 0
    """
    return (x[:,0]**2 + x[:,1]**2)**(alpha/2)

def f_manufactured(x, alpha):
    """
    f(x) = -Δ( ||x||^alpha ) = -alpha^2 * ||x||^{alpha-2}
    x: array shape (n_points, 2)
    """
    r2 = x[:,0]**2 + x[:,1]**2
    f = np.zeros_like(r2)
    idx = (r2 > 1e-14)   # evitar singularidad en (0,0)
    f[idx] = - alpha**2 * (r2[idx])**(alpha/2 - 1)
    return f

# ------------------------------
# Ensamblaje P1 (CST) en lectura de malla
# ------------------------------
def ensamblar_P1(coords, elems, f_fun, alpha):
    """
    Ensambla la matriz K (rigidez) y el vector F (carga) para elementos P1,
    a partir de coords (N×2), elems (M×3), función f_fun(x,alpha) y alpha.
    Retorna K (sparse csr) y F (vector numpy de tamaño N).
    """
    N = coords.shape[0]
    I, J, V = [], [], []      # listas para armar la sparse matrix
    F = np.zeros(N)           # vector carga global

    # Recorremos cada triángulo T_e con sus índices de nodos Te = [i,j,k]
    for Te in elems:
        # 1) Coordenadas de los vértices del triángulo
        x1, y1 = coords[Te[0]]
        x2, y2 = coords[Te[1]]
        x3, y3 = coords[Te[2]]

        # 2) Cálculo del área |T| = 0.5 * |(x2-x1)(y3-y1) - (x3-x1)(y2-y1)|
        area = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))

        # 3) Obtener (a_i, b_i, c_i) para cada phi_i, resolviendo el sistema:
        #    [1 xj yj] [a_i]   = [delta_{i1}, delta_{i2}, delta_{i3}]^T
        #    Etc. para i = 1,2,3.  (Mª en forma matricial).
        Msys = np.array([
            [1.0, x1, y1],
            [1.0, x2, y2],
            [1.0, x3, y3]
        ])
        # RHS para phi_1, phi_2, phi_3
        RHS = np.eye(3)  # 3×3; cada columna es [delta_{1j},delta_{2j},delta_{3j}]
        ABC = np.linalg.solve(Msys, RHS)  # dimensión (3×3)
        # grads[i] = (b_i, c_i), porque ∇φ_i = (b_i, c_i)
        grads = ABC[1:,:].T  # queda (3×2): grads[0] = (b_1, c_1), etc.

        # 4) Matriz elemental Ke (3×3): Ke[i,j] = ∇φ_i ⋅ ∇φ_j * |T|
        Ke = np.zeros((3, 3))
        for i_loc in range(3):
            bi, ci = grads[i_loc]
            for j_loc in range(3):
                bj, cj = grads[j_loc]
                Ke[i_loc, j_loc] = (bi * bj + ci * cj) * area

        # 5) Vector elemental Fe (3): usamos cuadratura de 1 punto (centroide T)
        xc = (x1 + x2 + x3) / 3.0
        yc = (y1 + y2 + y3) / 3.0
        f_cent = f_fun(np.array([[xc, yc]]), alpha)[0]
        # φ_i(centroide) = 1/3 para cada i en P1 => Fe[i] = f_cent * (1/3) * |T|
        Fe = np.ones(3) * (f_cent * area / 3.0)

        # 6) Ensamblaje en K y F globales
        for i_loc in range(3):
            I.append(Te[i_loc])
            J.append(Te[i_loc])
            V.append(Ke[i_loc, i_loc])
            for j_loc in range(i_loc + 1, 3):
                # entrada (i_loc, j_loc) y (j_loc, i_loc) por simetría
                I.extend([Te[i_loc], Te[j_loc], Te[j_loc], Te[i_loc]])
                J.extend([Te[j_loc], Te[i_loc], Te[i_loc], Te[j_loc]])
                V.extend([
                    Ke[i_loc, j_loc],
                    Ke[i_loc, j_loc],
                    Ke[j_loc, i_loc],
                    Ke[j_loc, i_loc]
                ])
            F[Te[i_loc]] += Fe[i_loc]

    # Construir K sparse en formato CSR
    K = sp.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    return K, F

# ------------------------------
# Aplicación de Dirichlet y solución del sistema
# ------------------------------
def aplicar_dirichlet_y_resolver(K, F, coords, bordes, alpha):
    """
    - K: matriz rigidez global (sparse CSR NxN)
    - F: vector carga global (long. N)
    - coords: arreglo (N×2) con coordenadas de cada nodo
    - bordes: lista/array de índices de nodos en la frontera
    - alpha: exponente de la solución exacta
    Retorna U (vector tamaño N) con la solución P1.
    """
    N = coords.shape[0]
    # 1) Inicializar U y poner valor de Dirichlet en nodos frontera
    U = np.zeros(N)
    if bordes.size > 0:
        U[bordes] = u_exact(coords[bordes], alpha)

    # 2) Ajustar F[i] -= sum_{j∈bordes} K[i,j] * U[j], para i ∈ nodos libres
    libres = np.setdiff1d(np.arange(N), bordes)
    F_mod = F.copy()
    # Restar influencia de Dirichlet
    # (si K es sparse, K[i, bordes] es rápido)
    for i in libres:
        if bordes.size > 0:
            F_mod[i] -= K[i, bordes].dot(U[bordes])

    # 3) Extraer submatriz K_ff y F_f
    K_ff = K[libres][:, libres]
    F_f  = F_mod[libres]

    # 4) Resolver K_ff * U_f = F_f
    U_f = spla.spsolve(K_ff, F_f)
    U[libres] = U_f
    return U

# ------------------------------
# Cálculo del error en seminorma H^1
# ------------------------------
def error_H1(K, U, alpha, cuad=200):
    """
    Calcula |u - u_h|_{H^1} = sqrt( |u|^2_{H^1} - U^T K U ),
    donde |u|^2_{H^1} = alpha^2 ∫_{Ω} (x^2+y^2)^{α-1} dx. 
    Aprox. con cuadratura tensorial de cuad×cuad.
    """
    # 1) Primer término: U^T K U
    UhKUh = U.dot(K.dot(U))

    # 2) Aproximar ∫_{Ω=(0,1)^2} alpha^2 * (x^2 + y^2)^{α-1} dx
    q = cuad
    xs = np.linspace(0, 1, q)
    ys = np.linspace(0, 1, q)
    hx = 1.0/(q - 1)
    hy = 1.0/(q - 1)
    Xg, Yg = np.meshgrid(xs, ys, indexing='xy')
    R2 = Xg**2 + Yg**2
    integrando = np.zeros_like(R2)
    idx = (R2 > 1e-14)
    integrando[idx] = (R2[idx])**(alpha - 1)
    integral = alpha**2 * np.sum(integrando) * hx * hy

    # 3) Error H1² = integral - UhKUh  (hacer max(0,.) para evitar ligeros negativos numéricos)
    e2 = integral - UhKUh
    if e2 < 0:
        e2 = 0.0
    return np.sqrt(e2)

# ------------------------------
# Función “driver” para leer .msh, ensamblar y resolver
# ------------------------------
def ejecutar_P1_con_msh(nombre_archivo_msh, alpha):
    """
    Lee el archivo .msh, arma K y F, aplica Dirichlet y resuelve P1.
    Retorna: U_h, coords, elems, bordes, K, F, error_H1
    """
    # 1) Leer malla
    coords, elems, bordes = leer_malla_msh(nombre_archivo_msh)
    print(f"--- Malla cargada desde '{nombre_archivo_msh}' ---")
    print(f" # nodos totales   = {coords.shape[0]}")
    print(f" # triángulos P1   = {elems.shape[0]}")
    print(f" # nodos frontera  = {bordes.size}")

    # 2) Ensamblar K y F
    K, F = ensamblar_P1(coords, elems, f_manufactured, alpha)
    print("Matriz K ensamblada (sparse):", K.shape)

    # 3) Resolver con Dirichlet
    U_h = aplicar_dirichlet_y_resolver(K, F, coords, bordes, alpha)
    print("Sistema resuelto, obtenida solución U_h.")

    # 4) Calcular error en seminorma H1
    err = error_H1(K, U_h, alpha, cuad=200)
    print(f"Error H^1  = {err:.6e}")

    return U_h, coords, elems, bordes, K, F, err

# ------------------------------
# Ejemplo de uso
# ------------------------------
if __name__ == "__main__":
    # Nombre del archivo .msh que se entrega
    archivo_msh = "pipe/geo.msh"

    # Escoger alpha para la solución manufacturada
    alpha = 0.5

    # Ejecutar todo el pipeline
    U_h, coords, elems, bordes, K, F, err = ejecutar_P1_con_msh(archivo_msh, alpha)

    # Opcional: guardar U_h en archivo de texto / plotear o procesar resultados
    # Por ejemplo, podemos guardar la solución en un .txt:
    np.savetxt("pipe\solucion_P1.txt", U_h)

    # Si se quiere visualizar la malla y la solución (por ejemplo, con matplotlib + tri):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        triang = mtri.Triangulation(coords[:,0], coords[:,1], elems)
        plt.figure(figsize=(6,5))
        tpc = plt.tricontourf(triang, U_h, levels=20, cmap="viridis")
        plt.colorbar(tpc, label=r"$u_h(x,y)$")
        plt.triplot(triang, color="k", lw=0.3, alpha=0.5)
        plt.title(f"Solución P1 (α={alpha}), Error H1 = {err:.2e}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.gca().set_aspect("equal", "box")
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

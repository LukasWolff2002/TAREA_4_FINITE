import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

class Solve:
    def __init__(self, nodes, lst_elements, alpha):
        self.nodes = nodes
        self.elements = lst_elements  # cambio de nombre más general
        self.alpha = alpha
        self.nnodes = len(nodes)

        self.make_elements_stiffness_matrices()
        self.K_global = self.assemble_global_matrix()
        self.u = np.zeros(self.nnodes)
        self.f = np.zeros(self.nnodes)

    def make_elements_stiffness_matrices(self):
        for element in self.elements:
            element.get_stiffness_matrix(self.nodes)

    def assemble_global_matrix(self):
        row_idx = []
        col_idx = []
        data = []

        for elem in self.elements:
            node_ids = [int(i) for i in elem.node_ids]  # base 1
            K_local = elem.K

            for i_local in range(6):
                for j_local in range(6):
                    global_i = node_ids[i_local] - 1
                    global_j = node_ids[j_local] - 1

                    row_idx.append(global_i)
                    col_idx.append(global_j)
                    data.append(K_local[i_local, j_local])

        K_global = coo_matrix((data, (row_idx, col_idx)), shape=(self.nnodes, self.nnodes)).tocsr()

        #print(f"✅ Norm(K_global) = {np.linalg.norm(K_global.toarray()):.3e}")
        return K_global


    def solve_matrix(self):
        fixed_dofs = []
        fixed_values = []

        for node in self.nodes:
            if hasattr(node, "boundary_label") and any("Dirichlet" in label for label in node.boundary_label):
                
                fixed_dofs.append(node.id - 1)
                fixed_values.append(node.u)

        fixed_dofs = np.array(fixed_dofs, dtype=int)
        fixed_values = np.array(fixed_values, dtype=float)
        #print(f"🔒 Nodos con Dirichlet: {fixed_dofs}")
        all_dofs = np.arange(self.nnodes)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

        K = self.K_global
        #print(f"🔧 Tamaño de K_global: {K.shape}")
        f = self.f.copy()
        #print(f"🔧 Tamaño de f: {f.shape}")
        #print(f)
        nodes = self.nodes
        
        n_dirichlet = sum("Dirichlet" in getattr(n, "boundary_label", []) for n in nodes)
        #print(f"🟩 Nodos marcados como Dirichlet: {n_dirichlet}")


        #print(f"📦 Norm(f) = {np.linalg.norm(self.f):.3e}")
        #print(f"🔒 Nodos con Dirichlet = {len(fixed_dofs)}")

        f_reduced = f[free_dofs] - K[free_dofs][:, fixed_dofs] @ fixed_values
        #print(f_reduced)
        u_free = spsolve(K[free_dofs][:, free_dofs], f_reduced)

        u_full = np.zeros(self.nnodes)
        u_full[fixed_dofs] = fixed_values
        u_full[free_dofs] = u_free

        self.u = u_full

        for node in self.nodes:
            node.u_fem = u_full[node.id - 1]

    def real_solution(self):
        for node in self.nodes:
            node.solve_u(self.alpha)

    def semi_norm_H1_0(self):
        alpha = self.alpha
        orden = 5

        puntos, pesos = np.polynomial.legendre.leggauss(orden)
        puntos = 0.5 * (puntos + 1)
        pesos = 0.5 * pesos

        total = 0.0

        for i in range(orden):
            for j in range(orden):
                x = puntos[i]
                y = puntos[j]
                w = pesos[i] * pesos[j]

                r2 = x**2 + y**2
                if r2 == 0 and alpha < 1:
                    grad2 = 0.0
                else:
                    grad2 = alpha**2 * r2**(alpha - 1)

                total += grad2 * w

        return total

    def femm_solution(self):
        """
        Calcula la energía FEM asociada a la solución u_h usando K_global.

        Retorna:
            fem_energy (float): Energía computada como x^T K x, donde x es el vector de desplazamientos.
        """
        # Vector de solución: u en nodos
        x = np.zeros(self.nnodes)
        for node in self.nodes:
            if hasattr(node, "boundary_label") and any("Dirichlet" in label for label in node.boundary_label):
                x[node.id - 1] = node.u   # condición de Dirichlet exacta
            else:
                x[node.id - 1] = node.u_fem  # solución FEM

        K = self.K_global

        # Validación básica de dimensiones
        assert K.shape == (self.nnodes, self.nnodes), "⚠️ Dimensiones de K_global incorrectas"

        fem_energy = x @ K @ x  # producto escalar x^T K x

        return fem_energy

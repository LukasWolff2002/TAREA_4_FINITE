import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

class Solve:
    def __init__(self, nodes, cst_elements, alpha):
        self.nodes = nodes
        self.cst_elements = cst_elements
        self.alpha = alpha
        self.nnodes = len(nodes)

        self.make_elements_stiffness_matrices()
        self.K_global = self.assemble_global_matrix()
        self.f = np.zeros(self.nnodes)
        self.assemble_global_vector()  # <<<<<< AGREGA el vector de carga aquí

        self.u = np.zeros(self.nnodes)

    def make_elements_stiffness_matrices(self):
        for element in self.cst_elements:
            element.get_stiffness_matrix(self.nodes)

    def assemble_global_matrix(self):
        row_idx = []
        col_idx = []
        data = []

        for elem in self.cst_elements:
            node_ids = [int(i) for i in elem.node_ids]  # base 1
            K_local = elem.K

            for i_local in range(3):
                for j_local in range(3):
                    global_i = node_ids[i_local] - 1
                    global_j = node_ids[j_local] - 1

                    row_idx.append(global_i)
                    col_idx.append(global_j)
                    data.append(K_local[i_local, j_local])

        return coo_matrix((data, (row_idx, col_idx)), shape=(self.nnodes, self.nnodes)).tocsr()

    def assemble_global_vector(self):
        self.f = np.zeros(self.nnodes)
        for elem in self.cst_elements:
            f_local = elem.get_load_vector(self.nodes, self.alpha)
            for i_local, node_id in enumerate(elem.node_ids):
                self.f[node_id - 1] += f_local[i_local]

    def solve_matrix(self):
        # DOFs fijos por Dirichlet
        fixed_dofs = [
            node.id - 1
            for node in self.nodes
            if hasattr(node, "boundary_label") and any("Dirichlet" in label for label in node.boundary_label)
        ]
        fixed_dofs = np.array(fixed_dofs, dtype=int)

        fixed_values = np.array([self.nodes[dof].u for dof in fixed_dofs], dtype=float)
        all_dofs = np.arange(self.nnodes)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

        K = self.K_global
        f_reduced = self.f[free_dofs] - K[free_dofs][:, fixed_dofs] @ fixed_values
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
        x = np.zeros(self.nnodes)
        for node in self.nodes:
            x[node.id - 1] = node.u_fem
        Kx = self.K_global @ x
        fem_energy = x @ Kx
        return fem_energy

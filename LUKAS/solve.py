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
        self.u = np.zeros(self.nnodes)
        self.f = np.zeros(self.nnodes)

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
                    global_i = node_ids[i_local] - 1  # corregir base
                    global_j = node_ids[j_local] - 1

                    row_idx.append(global_i)
                    col_idx.append(global_j)
                    data.append(K_local[i_local, j_local])

        return coo_matrix((data, (row_idx, col_idx)), shape=(self.nnodes, self.nnodes)).tocsr()

    def solve_matrix(self):
        fixed_dofs = []
        fixed_values = []

        for node in self.nodes:
            if hasattr(node, "boundary_label") and any("Diritchlet" in label for label in node.boundary_label):
                fixed_dofs.append(node.id - 1)  # corregir base
                fixed_values.append(node.u)

        fixed_dofs = np.array(fixed_dofs, dtype=int)
        fixed_values = np.array(fixed_values, dtype=float)

        all_dofs = np.arange(self.nnodes)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

        K = self.K_global
        f = self.f.copy()

        f_reduced = f[free_dofs] - K[free_dofs][:, fixed_dofs] @ fixed_values

        u_free = spsolve(K[free_dofs][:, free_dofs], f_reduced)

        u_full = np.zeros(self.nnodes)
        u_full[fixed_dofs] = fixed_values
        u_full[free_dofs] = u_free

        self.u = u_full

        # Almacenar solución en cada nodo
        # Almacenar solución en cada nodo (ajuste base 1 → base 0)
        for node in self.nodes:
            node.u_fem = u_full[node.id - 1]


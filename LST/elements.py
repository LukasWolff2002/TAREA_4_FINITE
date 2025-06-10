import numpy as np

class LST:
    def __init__(self, id, node_ids):
        self.id = id
        self.node_ids = node_ids  # Lista de 6 nodos (orden: 1-2-3-4-5-6)
        self.K = None  # Se calculará con get_stiffness_matrix

    def shape_function_derivatives(self, xi, eta):
        L1 = 1 - xi - eta
        L2 = xi
        L3 = eta

        dN_dxi = np.array([
            -4 * L1 + 1,          # ∂N1/∂xi
            4 * L2 - 1,           # ∂N2/∂xi
            0.0,                  # ∂N3/∂xi
            4 * (L1 - L2),        # ∂N4/∂xi
            4 * L3,               # ∂N5/∂xi
            -4 * L3               # ∂N6/∂xi
        ])

        dN_deta = np.array([
            -4 * L1 + 1,          # ∂N1/∂eta
            0.0,                  # ∂N2/∂eta
            4 * L3 - 1,           # ∂N3/∂eta
            -4 * L2,              # ∂N4/∂eta
            4 * L2,               # ∂N5/∂eta
            4 * (L1 - L3)         # ∂N6/∂eta
        ])

        return dN_dxi, dN_deta




    def get_B_matrix(self, nodes, xi, eta):
        """
        Calcula la matriz B en un punto (xi, eta) del triángulo de referencia
        usando interpolación cuadrática (6 nodos).
        """
        # Coordenadas reales de los nodos
        coords = np.array([[nodes[i - 1].x, nodes[i - 1].y] for i in self.node_ids])  # (6,2)

        # Derivadas de funciones de forma en coordenadas de referencia
        dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)

        # Jacobiano (2x2)
        J = np.zeros((2, 2))
        for i in range(6):
            J[0, 0] += dN_dxi[i] * coords[i, 0]
            J[0, 1] += dN_dxi[i] * coords[i, 1]
            J[1, 0] += dN_deta[i] * coords[i, 0]
            J[1, 1] += dN_deta[i] * coords[i, 1]

        detJ = np.linalg.det(J)
        if abs(detJ) < 1e-12:
            return np.zeros((2, 6)), 0.0  # Elemento degenerado

        # Inversa del Jacobiano
        Jinv = np.linalg.inv(J)

        # Derivadas de N respecto a x, y
        dN_dx = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
        dN_dy = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta

        # Construir B (2 x 6)
        B = np.vstack((dN_dx, dN_dy))

        return B, detJ

    def get_stiffness_matrix(self, nodes):
        """
        Calcula la matriz de rigidez del elemento LST usando cuadratura de 3 puntos de Gauss
        sobre un triángulo de referencia. La matriz B se evalúa en cada punto de integración.
        """
        # Puntos de cuadratura de orden 2 para triángulos (debería ser al menos orden 2 para LST)
        gauss_points = [
            (1/6, 1/6, 1/6),
            (2/3, 1/6, 1/6),
            (1/6, 2/3, 1/6)
        ]

        K = np.zeros((6, 6))
        I = np.identity(2)  # matriz constitutiva simple

        for xi, eta, w in gauss_points:
            B, detJ = self.get_B_matrix(nodes, xi, eta)
            if detJ == 0:
                continue  # Saltar elementos degenerados
            K += w * detJ * (B.T @ I @ B)

        self.K = K
        return self.K

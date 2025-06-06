import numpy as np

class CST:
    def __init__(self, id, node_ids):
        self.id = id
        self.node_ids = node_ids  # Lista de 3 enteros
        self.K = None  # Se calcula luego con get_stiffness_matrix

    def get_B_matrix(self, nodes):
        """
        Calcula la matriz B del elemento CST.
        nodes: Lista completa de nodos.
        """
        x1, y1 = nodes[self.node_ids[0]].x, nodes[self.node_ids[0]].y
        x2, y2 = nodes[self.node_ids[1]].x, nodes[self.node_ids[1]].y
        x3, y3 = nodes[self.node_ids[2]].x, nodes[self.node_ids[2]].y

        # Área del triángulo
        area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

        # Matriz B (para problema escalar tipo ∇²u = f)
        B = np.array([
            [y2 - y3, y3 - y1, y1 - y2],
            [x3 - x2, x1 - x3, x2 - x1]
        ]) / (2 * area)

        return B, area

    def get_stiffness_matrix(self, nodes):
        """
        Calcula y almacena la matriz de rigidez local del elemento CST.
        """
        B, area = self.get_B_matrix(nodes)
        I = np.identity(2)
        self.K = area * B.T @ I @ B  # Matriz 3x3
        return self.K

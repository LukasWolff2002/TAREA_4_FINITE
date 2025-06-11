import numpy as np

class CST:
    def __init__(self, id, node_ids):
        self.id = id
        self.node_ids = node_ids  # Lista de 3 enteros
        self.K = None  # Se calcula luego con get_stiffness_matrix

    def get_B_matrix(self, nodes):
        n1, n2, n3 = [nodes[i - 1] for i in self.node_ids]
        x1, y1 = n1.x, n1.y
        x2, y2 = n2.x, n2.y
        x3, y3 = n3.x, n3.y

        area_signed = 0.5 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        area = abs(area_signed)

        B = np.array([
            [y2 - y3, y3 - y1, y1 - y2],
            [x3 - x2, x1 - x3, x2 - x1]
        ]) / (2*area)

        return B, area

    def get_stiffness_matrix(self, nodes):
        B, area = self.get_B_matrix(nodes)
        self.K = area * (B.T @ B)
        return self.K

    def get_load_vector(self, nodes, alpha):
        n1, n2, n3 = [nodes[i - 1] for i in self.node_ids]
        x1, y1 = n1.x, n1.y
        x2, y2 = n2.x, n2.y
        x3, y3 = n3.x, n3.y

        xc = (x1 + x2 + x3) / 3
        yc = (y1 + y2 + y3) / 3

        r2 = xc**2 + yc**2
        if r2 == 0 and alpha < 1:
            f_val = 0.0
        else:
            f_val = -alpha**2 * r2**(alpha / 2 - 1)

        _, area = self.get_B_matrix(nodes)
        return np.full(3, f_val * area / 3)


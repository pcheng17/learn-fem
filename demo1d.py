import matplotlib.pyplot as plt
from math import pi, sin, cos
import numpy as np
import random

def solve(nodes, elements, left_bc, right_bc, rhs):
    """
    Solve 1D Poisson equation with mixed boundary conditions.

    Parameters:
    -----------
    nodes : array_like
        Node coordinates
    elements : list of lists
        Element connectivity [[node1, node2], ...]
    left_bc : dict
        Left boundary condition: {'type': 'dirichlet'/'neumann', 'value': float}
    right_bc : dict
        Right boundary condition: {'type': 'dirichlet'/'neumann', 'value': float}
    rhs : callable
        Right-hand side function f(x)

    Returns:
    --------
    solution : array
        Solution values at all nodes
    """

    node_count = len(nodes)

    constrained_nodes = []
    if left_bc['type'] == 'dirichlet':
        constrained_nodes.append(0)
    if right_bc['type'] == 'dirichlet':
        constrained_nodes.append(node_count - 1)

    dof_nodes = [i for i in range(node_count) if i not in constrained_nodes]
    dof_count = len(dof_nodes)
    node_to_dof = {node: i for i, node in enumerate(dof_nodes)}

    K = np.zeros((dof_count, dof_count))
    b = np.zeros(dof_count)

    for elem_nodes in elements:
        node1, node2 = elem_nodes
        x1, x2 = nodes[node1], nodes[node2]
        h = x2 - x1

        # Element stiffness matrix
        K_elem = (1.0 / h) * np.array([[ 1, -1], [-1,  1]])

        # Trapezoidal rule
        f_elem = 0.5 * h * np.array([rhs(x1), rhs(x2)])

        # Assemble
        for i, node_i in enumerate(elem_nodes):
            if node_i in dof_nodes:
                dof_i = node_to_dof[node_i]
                b[dof_i] += f_elem[i]

                for j, node_j in enumerate(elem_nodes):
                    if node_j in dof_nodes:
                        dof_j = node_to_dof[node_j]
                        K[dof_i, dof_j] += K_elem[i, j]

        # Handle Dirichlet boundary conditions
        for i, node_i in enumerate(elem_nodes):
            if node_i in dof_nodes:
                dof_i = node_to_dof[node_i]

                for j, node_j in enumerate(elem_nodes):
                    if node_j in constrained_nodes:
                        if node_j == 0:
                            dirichlet_value = left_bc['value']
                        else:  # node_j == node_count - 1
                            dirichlet_value = right_bc['value']

                        # Move known terms to RHS
                        b[dof_i] -= K_elem[i, j] * dirichlet_value

    # Apply Neumann boundary conditions
    if left_bc['type'] == 'neumann':
        left_dof = node_to_dof[0]
        b[left_dof] -= left_bc['value']
    if right_bc['type'] == 'neumann':
        right_dof = node_to_dof[node_count - 1]
        b[right_dof] += right_bc['value']

    # Solve system
    u = np.linalg.solve(K, b)

    # Reconstruct full solution
    solution = np.zeros(node_count)

    # Fill in free DOF values
    for node, dof in node_to_dof.items():
        solution[node] = u[dof]

    # Fill in Dirichlet values
    if left_bc['type'] == 'dirichlet':
        solution[0] = left_bc['value']
    if right_bc['type'] == 'dirichlet':
        solution[node_count - 1] = right_bc['value']

    return solution

if __name__ == "__main__":
    random.seed(42)

    interior_nodes = 100

    problems = [
        {
            "a": 0,
            "b": 2 * pi,
            "lbc": 0,
            "rbc": 0,
            "lbc_type": "dirichlet",
            "rbc_type": "dirichlet",
            "rhs": lambda x: sin(x),
            "exact_solution": lambda x: sin(x),
        },
        {
            "a": 0,
            "b": 2 * pi,
            "lbc": 1,
            "rbc": 1,
            "lbc_type": "dirichlet",
            "rbc_type": "dirichlet",
            "rhs": lambda x: cos(x),
            "exact_solution": lambda x: cos(x),
        },
        {
            "a": 0,
            "b": 2 * pi,
            "lbc": 1,
            "rbc": 0,
            "lbc_type": "dirichlet",
            "rbc_type": "neumann",
            "rhs": lambda x: cos(x),
            "exact_solution": lambda x: cos(x),
        }
    ]

    pIdx = 0

    problem = problems[pIdx]
    a = problem["a"]
    b = problem["b"]
    nodes = np.concatenate([[a], [random.uniform(a, b) for _ in range(interior_nodes)], [b]])
    nodes.sort()
    elements = [(i, i+1) for i in range(len(nodes) - 1)]

    left_bc = {
        "type": problem["lbc_type"],
        "value": problem["lbc"]
    }
    right_bc = {
        "type": problem["rbc_type"],
        "value": problem["rbc"]
    }

    rhs = problem["rhs"]
    exact_solution = problem["exact_solution"]

    solution = solve(nodes, elements, left_bc, right_bc, rhs)

    plt.figure(figsize=(12, 5))

    # First subplot - Solutions
    plt.subplot(1, 2, 1)
    plt.plot(nodes, solution, label='FEM Solution', color='blue', marker='o', markersize=3)
    plt.plot(nodes, [exact_solution(xi) for xi in nodes], label='Exact Solution', color='red', linestyle='--')
    plt.title('FEM Solution vs Exact Solution')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Second subplot - Error
    plt.subplot(1, 2, 2)
    exact_values = [exact_solution(xi) for xi in nodes]
    error = [abs(fem - exact) for fem, exact in zip(solution, exact_values)]
    plt.plot(nodes, error, label='Error (FEM - Exact)', color='green', marker='s', markersize=3)
    plt.title('Pointwise Error')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 1.0
E = 200e9
F = 1000
A0 = 0.01
alpha_modified = 1

# Analytical displacement for the modified design
def analytical_displacement_modified(x):
    return (F / (E * A0 * alpha_modified)) * (1 - np.exp(-alpha_modified * x))

# Modified cross-sectional area
def area_modified(x):
    return A0 * np.exp(alpha_modified * x)

# Calculate the local stiffness matrix
def local_stiffness_matrix(A_mid, dx):
    return (E * A_mid / dx) * np.array([[1, -1], [-1, 1]])

# Assemble the global stiffness matrix from local stiffness matrices
def assemble_global_stiffness_matrix(N):
    dx = L / N
    K = np.zeros((N+1, N+1))  # Global stiffness matrix initialized to zeros

    for i in range(N):
        x_i = i * dx
        x_ip1 = (i + 1) * dx
        x_mid = 0.5 * (x_i + x_ip1)  # Gauss quadrature 1-point rule
        A_mid = area_modified(x_mid)

        # Local stiffness matrix for the element
        k_local = local_stiffness_matrix(A_mid, dx)

        # Assemble into the global stiffness matrix
        K[i:i+2, i:i+2] += k_local

    return K

# Apply boundary condition (u(0) = 0)
def apply_boundary_conditions(K, f):
    K[0, 0] += 1e20  # Simulate the fixed boundary condition by adding a large value
    return K, f

# FEM displacement for the modified design
def fem_displacement_modified(N):
    dx = L / N
    K = assemble_global_stiffness_matrix(N)  # Global stiffness matrix
    f = np.zeros(N+1)  # Force vector initialized to zeros
    f[-1] = F  # Force applied at the right end

    # Apply boundary conditions
    K, f = apply_boundary_conditions(K, f)

    # Solve the linear system Ku=f
    u = np.linalg.solve(K, f)

    return u

# Calculate stress from displacement
def stress(u, N, dx):
    du_dx = np.diff(u) / dx  # Finite difference for gradient
    return (E / dx) * du_dx

# Main program
x_values = np.linspace(0, L, 250)
u_analytical_modified = [analytical_displacement_modified(x) for x in x_values]

elements_list = [4, 8, 32, 128]

# Plotting Displacement
plt.figure()
for N in elements_list:
    u_fem_modified = fem_displacement_modified(N)
    x_fem = np.linspace(0, L, N+1)

    plt.plot(x_fem, u_fem_modified, label=f'FEM Modified (N={N})')

plt.plot(x_values, u_analytical_modified, '--k', label='Analytical')
plt.xlabel("Length (x) [m]")
plt.ylabel("Displacement (u) [m]")
plt.title("Comparison of Analytical and FEM Solutions (Modified Design)")
plt.legend()
plt.grid(True)
plt.show()

# Plotting Stress
plt.figure()
for N in elements_list:
    u_fem_modified = fem_displacement_modified(N)
    x_fem = np.linspace(0, L, N+1)

    dx = L / N
    sigma_fem_modified = stress(u_fem_modified, N, dx)

    plt.plot(x_fem[:-1], sigma_fem_modified, label=f'FEM Modified (N={N})')

plt.xlabel("Length (x) [m]")
plt.ylabel("Stress σx [Pa]")
plt.title("Stress Variation over the Length of the Rod (Modified Design)")
plt.legend()
plt.grid(True)
plt.show()

# Plotting Error
plt.figure()
for N in elements_list:
    u_fem_modified = fem_displacement_modified(N)
    x_fem = np.linspace(0, L, N+1)

    u_analytical_at_nodes = [analytical_displacement_modified(x) for x in x_fem]
    error = np.abs(u_fem_modified - u_analytical_at_nodes)

    plt.plot(x_fem, error, label=f'Error (N={N})')

plt.xlabel("Length (x) [m]")
plt.ylabel("Error |uN - uA| [m]")
plt.title("Error between FEM and Analytical Solutions (Modified Design)")
plt.legend()
plt.grid(True)
plt.show()
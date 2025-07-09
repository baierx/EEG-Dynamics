from numpy import linspace, zeros, tanh, ones, arange, fill_diagonal
from numpy.random import default_rng
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Model equations

def sigmoid(u):
    return tanh(u)


def N_oscillators(t, y, N, h_ex_rand, h_in_rand, coupling_matrix_E, pars):

    tau_ex, tau_in, c1, c2, c3, c4 = pars

    # Separate Variables
    y_ex = y[:-1:2]
    y_in = y[1::2]

    dy_ex, dy_in = zeros(N), zeros(N)
    dydt = zeros(2*N)

    for osc in arange(N):
        
        coup_E = sum(coupling_matrix_E[:, osc] * y_ex)

        dy_ex[osc] = (h_ex_rand[osc] - y_ex[osc] + c1*sigmoid(y_ex[osc]) - c2*sigmoid(y_in[osc]) + coup_E)*tau_ex 
        dy_in[osc] = (h_in_rand[osc] - y_in[osc] + c3*sigmoid(y_ex[osc]) - c4*sigmoid(y_in[osc])         )*tau_in

    # Combine Variables

    dydt[:-1:2] = dy_ex
    dydt[1: :2] = dy_in

    return dydt


# Number of oscillators
N = 2

# Excitatory input parameter
h_ex_0 = -2 # -3.3
h_in_0 = -4

eps    = 0.0001
SEED   = 1234

rng = default_rng(SEED)

h_ex_rand = h_ex_0 + eps*rng.normal(0,1,size=N)
h_in_rand = h_in_0 + eps*rng.normal(0,1,size=N)

pars = (1, 1, 4, 6, 6, 0)

# Coupling matrices
coupling_matrix_E_ini = ones(shape=(N, N))

fill_diagonal(coupling_matrix_E_ini, 0)

coupling_strength_E = 0.14

coupling_matrix_E = coupling_strength_E * coupling_matrix_E_ini

# Initial conditions
SEED = 12

y_ini = rng.uniform(size=2*N)
# y_ini = y[-1, :]


# Time array
time_stop = 1000
sr        = 100
time      = linspace(start=0, stop=time_stop, num=time_stop*sr)


# Solve the ODEs
solution = solve_ivp(N_oscillators, (0, time_stop), y_ini,
          args=(N, h_ex_rand, h_in_rand, coupling_matrix_E, pars), method='BDF', max_step=1)


t, trajectory = solution.t[::3], solution.y[:, ::3].T

print(t.size, trajectory.shape)

# Set up the figure with 2 subplots
fig = plt.figure(figsize=(6, 8))
ax1 = fig.add_subplot(211, projection='3d')  # 3D state space
ax1.set_xlim(-2, 3)
ax1.set_ylim(-2, 3)
ax1.set_zlim(-1, 1)

ax2 = fig.add_subplot(212)                   # 2D time series

# Subplot 1: 3D trajectory (with tail)
ax1.set_xlabel('Ex 1'); ax1.set_ylabel('In 1'); ax1.set_zlabel('In 2')
ax1.set_title('State Space', fontsize=12)
traject_point, = ax1.plot([], [], [], 'ro', markersize=20)
trail, = ax1.plot([], [], [], 'b-', linewidth=2, alpha=0.5)
trail_length = 300

# Subplot 2: Time series of c1 (or c2/c3)
ax2.set_xlabel('Time'); ax2.set_ylabel('`Ex 1`')
# ax2.set_title('Simulated EEG', fontsize=12)
ax2.set_ylim(-1, 3)
time_line, = ax2.plot([], [], 'b-', label='Ex')
current_time_marker, = ax2.plot([], [], 'ro', markersize=8, label='Current time')
ax2.legend(loc='upper left')

fig.tight_layout()

# Initialize both subplots
def init():
    # 3D plot
    traject_point.set_data([], [])
    traject_point.set_3d_properties([])
    trail.set_data([], [])
    trail.set_3d_properties([])
    
    # Time series plot
    time_line.set_data([], [])
    current_time_marker.set_data([], [])  # Initialize with empty lists
    
    return [traject_point, trail, time_line, current_time_marker]

def update(frame):
    frame = min(frame, len(trajectory) - 1)
    
    # --- Update 3D plot ---
    x, y, z = trajectory[frame, 0], trajectory[frame, 1], trajectory[frame, 3]
    traject_point.set_data([x], [y])
    traject_point.set_3d_properties([z])
    
    # Update tail
    start = max(0, frame - trail_length)
    trail_x = trajectory[start:frame+1, 0]
    trail_y = trajectory[start:frame+1, 1]
    trail_z = trajectory[start:frame+1, 3]
    trail.set_data(trail_x, trail_y)
    trail.set_3d_properties(trail_z)
    
    # --- Update time series plot ---
    time_line.set_data(t[:frame+1], trajectory[:frame+1, 0]) # Plot Ex 1 vs time
    
    # Fix: Pass lists/arrays to set_data
    current_time_marker.set_data([t[frame]], [trajectory[frame, 0]])  # Red dot at current time
    
    # Adjust time series axis limits dynamically (optional)
    ax2.relim()
    ax2.autoscale_view()
    
    return [traject_point, trail, time_line, current_time_marker]

# Animate
ani = FuncAnimation(fig, update, frames=len(trajectory),
                   init_func=init, blit=False, interval=25)

# ani.save('cytosolic_calcium_oscillation.mp4', writer='ffmpeg', fps=30, dpi=200)

plt.tight_layout()
plt.show()
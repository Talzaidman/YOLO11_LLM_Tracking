import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV
df = pd.read_csv(r"C:\Users\zaita\PycharmProjects\yolosss\results\stain_positions.csv")

# Extract columns
x = df['x']
y = df['y']
z = df['z']  # Time

# Create 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Remap axes:
#   x -> height (Z axis)
#   y -> depth (Y axis)
#   z -> horizontal (X axis)
ax.plot(z, y, x, label='Polar Motion Path (Z is horizontal)', color='blue')

# Label axes based on the remapping
ax.set_xlabel('Time (Z)')
ax.set_ylabel('Y')
ax.set_zlabel('X')

# Optional: control aspect ratio
ax.set_box_aspect([6, 1, 1])

# Set view to look down the new Z-axis
ax.view_init(elev=20, azim=-60)

# Title and legend
ax.set_title('3D Polar Motion (Z Axis is Horizontal)')
ax.legend()

plt.tight_layout()
plt.show()

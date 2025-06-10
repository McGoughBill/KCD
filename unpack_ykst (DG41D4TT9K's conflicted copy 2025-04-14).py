import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# Set up the figure and axis.
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

# ---------------------------
# Central AI Icon (Circuit-Inspired)
# ---------------------------
ai_center = (0, 0)
ai_circle = patches.Circle(ai_center, radius=1, fill=False, lw=4, edgecolor='darkblue')
ax.add_patch(ai_circle)

# Draw circuit-style "spokes" inside the central AI icon
for angle in range(0, 360, 45):  # every 45 degrees
    x_end = 1 * math.cos(math.radians(angle))
    y_end = 1 * math.sin(math.radians(angle))
    ax.plot([0, x_end], [0, y_end], color='darkblue', lw=2)

# ---------------------------
# Detection Icon: Magnifying Glass
# ---------------------------
mag_center = (-3, 1)
mag_circle = patches.Circle(mag_center, radius=0.5, fill=False, lw=3, edgecolor='darkgreen')
ax.add_patch(mag_circle)
# Handle of the magnifying glass
ax.plot([mag_center[0] + 0.35, mag_center[0] + 1.2],
        [mag_center[1] - 0.35, mag_center[1] - 1.2],
        color='darkgreen', lw=3)

# ---------------------------
# Cost Savings Icon: Stacked Coins
# ---------------------------
coin_centers = [(3, 1.3), (3, 1), (3, 0.7)]
for center in coin_centers:
    coin = patches.Ellipse(center, width=0.8, height=0.4, fill=False, lw=3, edgecolor='purple')
    ax.add_patch(coin)

# ---------------------------
# Radiation Icon: Simplified Hazard Symbol
# ---------------------------
rad_center = (0, -3)
rad_circle = patches.Circle(rad_center, radius=0.6, fill=False, lw=3, edgecolor='red')
ax.add_patch(rad_circle)

# Create three triangular "rays" around the radiation icon.
for angle in [90, 210, 330]:
    tip = (rad_center[0] + 1.0 * math.cos(math.radians(angle)),
           rad_center[1] + 1.0 * math.sin(math.radians(angle)))
    base1 = (rad_center[0] + 0.6 * math.cos(math.radians(angle + 10)),
             rad_center[1] + 0.6 * math.sin(math.radians(angle + 10)))
    base2 = (rad_center[0] + 0.6 * math.cos(math.radians(angle - 10)),
             rad_center[1] + 0.6 * math.sin(math.radians(angle - 10)))
    triangle = patches.Polygon([tip, base1, base2], closed=True, fill=False, lw=3, edgecolor='red')
    ax.add_patch(triangle)

# ---------------------------
# Connecting Arrows
# ---------------------------
def draw_arrow(start, end, color):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=2, color=color))

# Connect the central AI icon with each surrounding element.
draw_arrow(ai_center, mag_center, 'darkblue')
draw_arrow(ai_center, (3, 1), 'darkblue')
draw_arrow(ai_center, rad_center, 'darkblue')

plt.show()

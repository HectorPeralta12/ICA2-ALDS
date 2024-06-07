import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tkinter as tk
from tkinter import ttk, messagebox

# Define state colors and connections color in BGR format
state_colors = {
    "Baja California Sur": (225, 62, 110),
    "Baja California Norte": (88, 140, 126),
    "Chihuahua": (104, 228, 170),
    "Sonora": (219, 107, 92),
    "Coahuila": (245, 146, 110),
    "Nuevo Leon": (32, 129, 165),
    "Tamaulipas": (243, 98, 4),
    "Sinaloa": (136, 107, 183),
    "Durango": (156, 24, 254),
    "Zacatecas": (24, 206, 254),
    "Nayarit": (207, 219, 102),
    "San Luis": (121, 132, 191),
    "Jalisco": (69, 48, 118),
    "Guanajuato": (146, 142, 65),
    "Michoacan": (216, 33, 0),
    "Ciudad de Mexico": (208, 0, 216),
    "Guerrero": (40, 58, 15),
    "Veracruz": (84, 157, 49),
    "Oaxaca": (175, 64, 0),
    "Tabasco": (44, 178, 162),
    "Chiapas": (193, 177, 84),
    "Guatemala": (43, 255, 180),
    "Campeche": (167, 120, 197),
    "Yucatan": (98, 16, 15),
    "Quintana Roo": (56, 222, 1)
}

# Connection color in BGR format
connection_color = (153, 99, 90)

# Load the map image
map_image = cv2.imread('/Users/mp/Desktop/ALDS/PROYECTO/mapMexico2.png')

# Convert image to RGB
map_image_rgb = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)

# Create an empty graph
G = nx.Graph()

# Function to detect states and add nodes to the graph
def detect_states(map_image_rgb, state_colors):
    positions = {}
    for state, color in state_colors.items():
        mask = cv2.inRange(map_image_rgb, color, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                positions[state] = (cX, cY)
                G.add_node(state, pos=(cX, cY))
                print(f"Detected {state} at position: {cX}, {cY}")
            else:
                print(f"Failed to detect {state}: no contours with non-zero area found.")
        else:
            print(f"Failed to detect {state}: no contours found.")
    return positions

# Detect states
positions = detect_states(map_image_rgb, state_colors)

# Function to detect connections and add edges to the graph
def detect_connections(map_image_rgb, connection_color):
    lower_bound = np.array(connection_color) - 20
    upper_bound = np.array(connection_color) + 20
    mask = cv2.inRange(map_image_rgb, lower_bound, upper_bound)
    edges = cv2.Canny(mask, 30, 100)  # Adjusted for finer edge detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=5)  # Adjusted for finer line detection
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            node1 = find_closest_node(positions, (x1, y1))
            node2 = find_closest_node(positions, (x2, y2))
            if node1 != node2 and not G.has_edge(node1, node2):
                dist = np.linalg.norm(np.array(positions[node1]) - np.array(positions[node2]))
                G.add_edge(node1, node2, weight=dist)
                print(f"Connected {node1} to {node2} with distance {dist}")

# Function to find the closest node to a given point
def find_closest_node(positions, point):
    min_dist = float('inf')
    closest_node = None
    for node, pos in positions.items():
        dist = np.linalg.norm(np.array(pos) - np.array(point))
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    return closest_node

# Detect connections
detect_connections(map_image_rgb, connection_color)

# Function to find the shortest path using Dijkstra's algorithm
def find_shortest_path(G, origin, destination):
    try:
        return nx.dijkstra_path(G, origin, destination), nx.dijkstra_path_length(G, origin, destination)
    except nx.NetworkXNoPath:
        messagebox.showerror("Error", f"No path between {origin} and {destination}")
        return None, None
    except nx.NodeNotFound as e:
        messagebox.showerror("Error", str(e))
        return None, None

# Function to draw the shortest path on the map
def draw_shortest_path(map_image_rgb, positions, path):
    for i in range(len(path) - 1):
        pos1 = positions[path[i]]
        pos2 = positions[path[i + 1]]
        cv2.line(map_image_rgb, pos1, pos2, (255, 255, 0), 2)  # Yellow line
    plt.imshow(map_image_rgb)
    plt.title("Shortest Path Found")
    plt.show()

# Create the Tkinter interface
def main():
    root = tk.Tk()
    root.title("Map of Mexican States")

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(frame, text="Origin State:").grid(column=0, row=0, sticky=tk.W)
    origin = ttk.Combobox(frame, values=list(state_colors.keys()))
    origin.grid(column=1, row=0, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="Destination State:").grid(column=0, row=1, sticky=tk.W)
    destination = ttk.Combobox(frame, values=list(state_colors.keys()))
    destination.grid(column=1, row=1, sticky=(tk.W, tk.E))

    def search_route():
        origin_state = origin.get()
        destination_state = destination.get()
        if origin_state in state_colors and destination_state in state_colors:
            path, distance = find_shortest_path(G, origin_state, destination_state)
            if path:
                messagebox.showinfo("Shortest Path", f"Distance: {distance:.2f} units")
                draw_shortest_path(map_image_rgb.copy(), positions, path)  # Use a copy to avoid drawing multiple paths
        else:
            messagebox.showerror("Error", "Please select valid states.")

    ttk.Button(frame, text="Search Route", command=search_route).grid(column=0, row=2, columnspan=2)

    for child in frame.winfo_children():
        child.grid_configure(padx=5, pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()


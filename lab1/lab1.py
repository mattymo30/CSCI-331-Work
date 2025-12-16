import heapq
import math
import sys
from PIL import Image


"""
file: lab1.py
CSCI-331
author: Matthew Morrison msm8275

Find the shortest path from a start point and end point in an
orienteering terrain map using A* algorithm
"""

class Node:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.parent = None
        self.total_cost = sys.maxsize # replicate f(n)
        self.path_cost = 0 # replicate g(n)
        self.terrain_cost = 0 # replicate h(n)
        self.length_in_m = 0 # g(n) without terrain cost

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)


# 1 px of map is equivalent to 10.29m x 7.55m
x_dist = 10.29
y_dist = 7.55

# maps are 395x500 images
longitude = 395
latitude = 500

path_color = (161, 70, 221, 255)

pixels = [[Node() for _ in range(longitude)] for _ in range(latitude)]


# cost values based on the terrain we are on
terrain_cost = {
    '#f89412': 3, # open land
    '#ffc000': 4, # rough meadow
    '#ffffff': 6, # easy movement forest
    '#02d03c': 8, # slow run forest
    '#028828': 10, # walk forest
    '#054918': sys.maxsize, # impassible vegetation
    '#0000ff': 100, # lake/swamp/marsh
    '#473303': 1, # paved road
    '#000000': 2, # footpath
    '#cd0065': sys.maxsize  # out of bounds
}

def reset_path_total_cost(explored_points):
    """
    reset the f(n), g(n), and length in meters of all explored points
    in the visited set
    :param explored_points: the set of all finalized nodes from running
    a* algorithm
    """
    for coords in explored_points:
        curr_point = pixels[coords[1]][coords[0]]
        curr_point.total_cost, curr_point.path_cost, curr_point.length_in_m = sys.maxsize, 0, 0

def reset_open_nodes(frontier):
    """
    reset the f(n), g(n), and length in meters of all visited (but not finalized)
    nodes that remain in the frontier
    :param frontier: the visited but nor finalized node priority queue
    """
    while frontier:
        curr_frontier_point = heapq.heappop(frontier)
        curr_node = curr_frontier_point[1]
        curr_node.total_cost, curr_node.path_cost, curr_node.length_in_m = sys.maxsize, 0, 0

def create_nodes_of_map(elevation_map):
    """
    initially create all the nodes of the map and add the x, y, and z values
    :param elevation_map: the elevation map file consisting of 500 lines of
    400 values (400x500). The last 5 values are ignored, making it the
    395x500 representation as the terrain map
    """
    with open(elevation_map) as file:
        row = 0
        for line in file:
            # needs to convert to float first to prevent and int ValueError
            row_elevation = [int(float(num)) for num in
                                   line.split()[:longitude]]
            for x in range(len(row_elevation)):
                curr_cell = pixels[row][x]
                curr_cell.x = x
                curr_cell.y = row
                curr_cell.z = row_elevation[x]
            row += 1

def get_hex_vals_of_terrain(terrain_img, goal_points):
    """
    get the hex color values of the given terrain image
    :param goal_points: an array of points that resemble where the user needs
    to traverse to in chronological order
    :param terrain_img: the image file of the terrain map
    """
    terrain = Image.open(terrain_img)
    terrain.convert('RGB')

    for y in range(latitude):
        for x in range(longitude):
            curr_pixel = [x, y]
            if curr_pixel in goal_points:
                continue
            rgb_tuple = terrain.getpixel((x, y))
            curr_cell = pixels[y][x]
            # ignore the alpha value, it's always 255
            hex_color = '#%02x%02x%02x' % rgb_tuple[0:3]
            curr_cell.terrain_cost = terrain_cost.get(hex_color)


def get_goal_points(path_file):
    """
    convert the goal point(s) as an array of [x,y] coordinates in order
    :param path_file: the file of point values that need to be visited
    :return: an array of [x,y] coordinates that represents the goal points
    that need to be traversed to
    """
    goal_points = []
    with open(path_file) as file:
        for line in file:
            goal_points.append([int(num) for num in line.split()])
    return goal_points


def draw_goal_path(final_node, terrain_img, output_file):
    """
    draw on the output file the final path found from the start node
    and final node
    :param final_node: the coordinates of the final node
    :param terrain_img: the terrain image to open and read from
    :param output_file: the output file to display the path on the original
    terrain image
    """
    curr_node = final_node
    t_img = Image.open(terrain_img)
    t_img = t_img.convert('RGB')

    # go until you find the start node of the path
    while curr_node is not None:
        parent_node = curr_node.parent

        t_img.putpixel((curr_node.x, curr_node.y), path_color)
        curr_node = parent_node

    t_img.save(output_file)

def get_neighbors(x, y):
    """
    get the neighboring coordinates of the given coordinate
    :param x: the longitude position
    :param y: the latitude position
    :return: an array of the 4 direct neighbors of the current coordinate
    """
    neighbors = []

    north = (x, y-1)
    if north[1] >= 0:
        neighbors.append(north)

    south = (x, y+1)
    if south[1] < latitude:
        neighbors.append(south)

    east = (x+1, y)
    if east[0] < longitude:
        neighbors.append(east)

    west = (x-1, y)
    if west[0] >= 0:
        neighbors.append(west)

    northeast = (x+1, y-1)
    if northeast[0] < longitude and northeast[1] >= 0:
        neighbors.append(northeast)

    northwest = (x-1, y-1)
    if northwest[1] >= 0 and northwest[0] >= 0:
        neighbors.append(northwest)

    southeast = (x+1, y+1)
    if southeast[1] < latitude and southeast[0] < longitude:
        neighbors.append(southeast)

    southwest = (x-1, y+1)
    if southeast[1] < latitude and southwest[0] >= 0:
        neighbors.append(southwest)


    return neighbors


def calc_3d_distance(start_node, end_node):
    """
    calculate the Euclidean distance between two points on the map
    :param start_node: the node to start on
    :param end_node: the node to finish on
    :return: the Euclidean distance
    """
    delta_x = (end_node.x - start_node.x)
    delta_y = (end_node.y - start_node.y)
    delta_z = (end_node.z - start_node.z)
    return math.sqrt((delta_x * x_dist) ** 2 + (delta_y * y_dist) ** 2 + delta_z ** 2)


def find_astar_path(start_point_coords, end_point_coords):
    """
    calculate the shortest path from start_point_coords to end_point_coords
    using the A* algorithm
    :param start_point_coords: the coordinates of the start point
    :param end_point_coords: the coordinates of the end point
    :return: the total length in meters of the path to the end point OR NONE
    if a path cannot be found
    """
    visited = set()
    frontier = []

    start_node = pixels[start_point_coords[1]][start_point_coords[0]]
    start_node.parent = None
    end_node = pixels[end_point_coords[1]][end_point_coords[0]]
    end_node.parent = None

    heapq.heappush(frontier, (0, start_node))

    # while priority queue is not empty
    while frontier:
        curr_frontier_point = heapq.heappop(frontier)

        # separate to get the actual node object
        curr_node = curr_frontier_point[1]

        # ensure it is not visited again
        visited.add((curr_node.x, curr_node.y))

        # end node has been discovered in queue, end the loop and return
        if ((curr_node.x, curr_node.y) ==
                (end_point_coords[0], end_point_coords[1])):
            tot_dist = curr_node.length_in_m
            reset_path_total_cost(visited)
            reset_open_nodes(frontier)
            return tot_dist

        neighbors = get_neighbors(curr_node.x, curr_node.y)
        for neighbor in neighbors:
            # ignore neighbor if already seen via frontier
            if neighbor not in visited:
                neighbor_node = pixels[neighbor[1]][neighbor[0]]

                # neighbor already visited or impassible? ignore if true
                if neighbor_node.terrain_cost == sys.maxsize:
                    continue

                # calculate hn, gn, length_in_m, and fn for this node
                dist_to_neighbor_node = calc_3d_distance(curr_node,
                                                         neighbor_node)
                neighbor_hn = calc_3d_distance(neighbor_node, end_node)
                neighbor_gn = (curr_node.path_cost + dist_to_neighbor_node
                               + neighbor_node.terrain_cost)
                neighbor_fn = neighbor_gn + neighbor_hn
                neighbor_length_in_meters = (dist_to_neighbor_node
                                             + curr_node.length_in_m)

                # first time seeing this node? auto add to the frontier
                if neighbor_node.total_cost == sys.maxsize:
                    neighbor_node.length_in_m = neighbor_length_in_meters
                    neighbor_node.path_cost = neighbor_gn
                    neighbor_node.total_cost = neighbor_fn
                    neighbor_node.parent = curr_node
                    heapq.heappush(frontier, (neighbor_fn, neighbor_node))
                    continue

                # this is not the first time seeing this node, compare fn values
                neighbor_prev_cost = neighbor_node.total_cost
                # found a better path? add new node to frontier so it is seen
                #earlier
                if neighbor_fn < neighbor_prev_cost:
                    neighbor_node.length_in_m = neighbor_length_in_meters
                    neighbor_node.path_cost = neighbor_gn
                    neighbor_node.total_cost = neighbor_fn
                    neighbor_node.parent = curr_node
                    heapq.heappush(frontier, (neighbor_fn, neighbor_node))


    return None


def orienteering(terrain_image, elevation_file, path_file, output_file):
    """
    Using the A* algorithm, calculate the shortest path from the starting point
    to the end point, passing through each goal point provided in the path file
    :param terrain_image: the image of the terrain map
    :param elevation_file: a file containing the elevation values of all pixels
    :param path_file: the file of point values that need to be visited
    :param output_file: the file to output the finished path on the
    original terrain map image

    Will output the total path length in meters to stdout
    """
    overall_path_length = 0

    goal_points = get_goal_points(path_file)
    create_nodes_of_map(elevation_file)
    get_hex_vals_of_terrain(terrain_image, goal_points)

    start_node = goal_points.pop(0)
    first_iteration = True
    while len(goal_points) >= 1:
        end_node = goal_points.pop(0)
        overall_path_length += find_astar_path(start_node, end_node)
        # draw the final path for this iteration
        if first_iteration:
            draw_goal_path(pixels[end_node[1]][end_node[0]], terrain_image, output_file)
            first_iteration = False
        else:
            draw_goal_path(pixels[end_node[1]][end_node[0]], output_file, output_file)
        start_node = end_node

    final_img = Image.open(output_file)
    final_img.save(output_file)
    print(overall_path_length)


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 lab1.py terrain-image elevation-file path-file output-image-filename")
        return

    terrain_image = sys.argv[1]
    elevation_file = sys.argv[2]
    path_file = sys.argv[3]
    output_image_filename = sys.argv[4]

    orienteering(terrain_image, elevation_file, path_file, output_image_filename)


if __name__ == '__main__':
    main()
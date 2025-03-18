import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from math               import pi, atan, sin, tan, dist, radians
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon
from shapely.prepared   import prep



######################################################################
#
#   Parameters
#
#   Define the step size.  Also set the maximum number of nodes.
#
DSTEP = 1

#   Activates/deactivates time based distance metrics.
USETIME = True

#   Define SPEEDXY distance the robot can move in one time-step
SPEEDXY = 10
#   Define robot speed (angle of search cone) (must be < 90 and > 0)
SPEED = atan(1/SPEEDXY)

#   Percentage to select goal as growth target
GOALPER = 0.05

#   Maximum number of steps (attempts) or nodes (successful steps).
SMAX = 50000
NMAX = 1500

#   Map bounds
(xmin, xmax) = (0, 10)
(ymin, ymax) = (0, 12)

#   Time Bound
TB = 100

#   Start and Goal locations
(xstart, ystart) = (5, 1)
(xgoal,  ygoal)  = (5, 11)
(tstart, tgoal)  = (5, TB)


#   Define Obstacles
side_len = 1
# Coordinates of a square
x = lambda z: 5 + sin(z/10*2*pi)
y = lambda z: 3 + sin(z/10*2*pi)
square = lambda z: [[x(z),y(z)], [x(z), y(z) + side_len], [x(z) + side_len, y(z) + side_len], [x(z) + side_len, y(z)]]
squarez = lambda z: [[x(z),y(z), z], [x(z), y(z) + side_len, z], [x(z) + side_len, y(z) + side_len, z], [x(z) + side_len, y(z), z]]

# Coordinates of a second square
x2 = lambda z: 8 + sin(z/10*2*pi)
y2 = lambda z: 8 + sin(z/10*2*pi)
square2 = lambda z: [[x2(z),y2(z)], [x2(z), y2(z) + side_len], [x2(z) + side_len, y2(z) + side_len], [x2(z) + side_len, y2(z)]]
square2z = lambda z: [[x2(z),y2(z), z], [x2(z), y2(z) + side_len, z], [x2(z) + side_len, y2(z) + side_len, z], [x2(z) + side_len, y2(z), z]]

# Wall with opening and closing door
def wall(z):
    if z<80 and z>60:
        return [[0, 2], [0, 4], [8, 4], [8, 2]]
    return [[0, 2], [0, 4], [10, 4], [10, 2]]

def wallz(z):
    if z<80 and z>60:
        return [[0, 2, z], [0, 4, z], [8, 4, z], [8, 2, z]]
    return [[0, 2, z], [0, 4, z], [10, 4, z], [10, 2, z]]

def wall2(z):
    if z<80 and z>30:
        return [[0, 5], [0, 6], [8, 6], [8, 5]]
    return [[0, 5], [0, 6], [10, 6], [10, 5]]

def wall2z(z):
    if z<80 and z>30:
        return [[0, 5, z], [0, 6, z], [8, 6, z], [8, 5, z]]
    return [[0, 5, z], [0, 6, z], [10, 6, z], [10, 5, z]]
    

#   Prepare obstacle graphics and define obstacle positions w.r.t time and prep
polygons = []
obstacles = []
for t in range(0, TB):
    prepped = prep(MultiPolygon([
                Polygon(wall2(t)),
                Polygon(wall(t))]))
    obstacles.append(prepped)

    polygons.append(wall2z(t))
    polygons.append(wallz(t))
    
#####################################################################
#
#   Utilities: Visualization
#
# Visualization Class
class Visualization:
    def __init__(self):
        # Clear the current, or create a new figure.

        # Create a new axes, enable the grid, and set axis limits.
        ax = plt.figure().add_subplot(projection='3d')
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

        # Plot the obstacles
        poly = Poly3DCollection(polygons, alpha=1)
        ax.add_collection3d(poly)

        # Show.
        self.show()

    def show(self, text = ''):
        # Show the plot.
        plt.pause(0.001)
        # If text is specified, print and wait for confirmation.
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, *args, **kwargs):
        plt.plot(node.x, node.y, node.t, *args, **kwargs)

    def drawEdge(self, head, tail, *args, **kwargs):
        plt.plot((head.x, tail.x),
                 (head.y, tail.y),
                 (head.t, tail.t), 
                 *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], *args, **kwargs)


######################################################################
#
#   Node Definition
#
class Node:
    def __init__(self, x, y, t):
        # Define a parent (cleared for now).
        self.parent = None

        # Define/remember the state/coordinates (x,y).
        self.x = x
        self.y = y
        self.t = t

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<Point %5.2f,%5.2f,%5.2f>" % (self.x, self.y, self.t))

    # Compute/create an intermediate node.  This can be useful if you
    # need to check the local planner by testing intermediate nodes.
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                    self.y + alpha * (other.y - self.y),
                    int(self.t + alpha * (other.t - self.t)))

    # Return a tuple of coordinates, used to compute Euclidean distance.
    def coordinates(self):
        return (self.x, self.y)
    
    # Compute the relative Euclidean distance to another node.
    def distance(self, other):
        return dist(self.coordinates(), other.coordinates())
    
    # Return a tuple of coordinates, used to compute Euclidean distance w.r.t time.
    def coordinatesT(self):
            return (self.x, self.y, self.t)
    
    # Compute the relative time distance to another node.
    def distanceT(self, other):
        return dist(self.coordinates(), other.coordinates())

    ################
    # Collision functions:
    # Check whether in free space.
    def inFreespace(self):
        if (self.x <= xmin or self.x >= xmax or
            self.y <= ymin or self.y >= ymax):
            return False
        return obstacles[self.t].disjoint(Point(self.coordinates()))
            
    def inCone(self, other):
        def inCircle(circle_x, circle_y, rad, x, y):
            # Compare radius of circle
            # with distance of its center
            # from given point
            if (dist([x, y], [circle_x, circle_y]) <= rad):
                return True
            else:
                return False
            
        delta_t = other.t - self.t 
        if delta_t > 0:
            return inCircle(self.x, self.y, delta_t * tan(radians(90)-SPEED), other.x, other.y)
        return False
    
    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other):
            line = LineString([self.coordinates(), other.coordinates()])
            disjoint = True
            for t in range(self.t, other.t):
                if not obstacles[t].disjoint(line):
                    disjoint = False
                    break

            return disjoint

######################################################################
#
#   TVRRT Functions
#
def tvrrt(startnode, goalnode, visual):
    def generateNode():
        t = np.random.randint(tstart+1, tgoal)
        offset = (t-tstart) * tan(radians(90)-SPEED)

        xlow = startnode.x-offset
        xhigh = startnode.x+offset
        if xlow < xmin:
            xlow = xmin
        if xhigh > xmax:
            xhigh = xmax

        x = np.random.uniform(xlow, xhigh)
        
        ylow = startnode.y-offset
        yhigh = startnode.y+offset
        if ylow < ymin:
            ylow = ymin
        if yhigh > ymax:
            yhigh = ymax
        y = np.random.uniform(ylow, yhigh)
        return Node(x, y, t)
    
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode            
        tree.append(newnode)
        visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
        visual.show()

    # Loop - keep growing the tree.
    steps = 0
    while True:
        # Determine the target state.
        if np.random.uniform() <= GOALPER:
            targetnode = goalnode
        else:
            # Generate legal node within start node's search cone
            targetnode = generateNode()

        # Directly determine the distances to the target node.
        legaltree = [node for node in tree if node.inCone(targetnode)]
        
        # Switches on/off time based distance metrics
        if USETIME:
            distances = np.array([node.distanceT(targetnode) for node in legaltree])
        else:
            distances = np.array([node.distance(targetnode) for node in legaltree])

        index     = np.argmin(distances)
        nearnode  = legaltree[index]
        d         = distances[index]

        # Determine the next node.
        if DSTEP >= d:
            nextnode = targetnode
        else:
            nextnode = nearnode.intermediate(targetnode, (DSTEP/d))
        
        # Check whether to attach.
        if nextnode.inFreespace() and nearnode.connectsTo(nextnode):
            addtotree(nearnode, nextnode)

            # If within DSTEP, also try connecting to the goal.  If
            # the connection is made, break the loop to stop growing.
            if nextnode.distance(goalnode) <= DSTEP and nextnode.connectsTo(goalnode):
                addtotree(nextnode, goalnode)
                break

        # Check whether we should abort - too many steps or nodes.
        steps += 1
        if (steps >= SMAX) or (len(tree) >= NMAX):
            print("Aborted after %d steps and the tree having %d nodes" %
                  (steps, len(tree)))
            return None

    # Build the path.
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    # Report and return.
    print("Finished after %d steps and the tree having %d nodes" %
          (steps, len(tree)))
    return path


def PostProcess(path):
    i = 0
    while (i < len(path)-2):
        if path[i].connectsTo(path[i+2]):
            path.pop(i+1)
        else:
            i = i+1

######################################################################
#
#  Main Code
#
def main():
    # Report the parameters.
    print(f"Running with step size {DSTEP} speed {SPEED} and up to {NMAX} nodes.")

    # Create the figure.
    visual = Visualization()

    # Create the start/goal nodes.
    startnode = Node(xstart, ystart, 0)
    goalnode  = Node(xgoal,  ygoal, TB)

    # Show the start/goal nodes.
    visual.drawNode(startnode, color='orange', marker='o')
    visual.drawNode(goalnode,  color='purple', marker='o')
    visual.show("Showing basic world")


    # Run the RRT planner.
    print("Running RRT...")
    path = tvrrt(startnode, goalnode, visual)

    # If unable to connect, just note before closing.
    if not path:
        visual.show("UNABLE TO FIND A PATH")
        return

    # Show the path.
    visual.drawPath(path, color='r', linewidth=2)
    visual.show("Showing the raw path")


    # Post process the path.
    PostProcess(path)

    # Show the post-processed path.
    visual.drawPath(path, color='b', linewidth=2)
    visual.show("Showing the post-processed path")


if __name__== "__main__":
    main()

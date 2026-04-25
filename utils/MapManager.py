import numpy as np
import time
from heapq import heapify, heappop, heappush
np.set_printoptions(precision=4)

from utils.UtilsMisc import *

class World:
    def __init__(self, clip_prec=4, start_weight=4096, cull_threshold=10000):
        self.pose = None
        self.clip_prec = clip_prec
        self.raw_points = None
        self.start_weight = start_weight
        self.grid = CellGrid()
        self.cull_threshold = cull_threshold

    def update(self, pts):
        tstart = time.process_time()
        print("Running update operation")
        if self.raw_points is None:
            self.raw_points = pts
        else:
            # single vstack is o(n) rather than appending in a loop, which is o(n^2)
            self.raw_points = np.vstack((self.raw_points, pts))

        tstart = time.process_time()
        times = []
        for point in pts:
            # pixelate
            clipped_x = self.clip(point[0])
            clipped_y = self.clip(point[1])
            times.append(self.grid.addCell(clipped_x, clipped_y, self.start_weight))
        avg_time = sum(times) / len(times)
        print("Avg march time was " + str(avg_time))
        print("# of pts: " + str(len(times)))

        pt_ct = self.raw_points.shape[0]
        if pt_ct > self.cull_threshold:
            victims = np.random.choice(pt_ct, size=pt_ct - self.cull_threshold, replace=False)
            # should also be o(n) operation
            self.raw_points = np.delete(self.raw_points, victims, axis=0)

        print("Update took " + str(time.process_time() - tstart))

    def clip(self, z):
        return math.floor(z * math.pow(10, self.clip_prec))

    def export(self, path):
        with open(path + "raw.npz", "wb+") as f:
            np.save(f, self.raw_points)
        wpts = []
        for _, val in self.grid.cellmap.items():
            for _, c in val.items():
                wpts.append([c.x, c.y, c.weight])
        with open(path + "pixels.npz", "wb+") as f:
            np.save(f, np.array(wpts))



class CellGrid:
    def __init__(self):
        self.cellmap = {}

    def addCell(self, x, y, w):
        if x not in self.cellmap:
            self.cellmap[x] = {}
        if y not in self.cellmap[x]:
            c = Cell(x, y, w)
            self.cellmap[x][y] = c
        else:
            # cell already created, but was either a placeholder or marched/infected
            c = self.getCell(x, y)
            if c.weight < w:
                # overwrite
                c.weight = w
        tstart = time.process_time()
        self.march(c, w)
        return time.process_time() - tstart

    def getNeighbors(self, c):
        return [self.getCell(c.x + 1, c.y + 1),
                self.getCell(c.x + 1, c.y),
                self.getCell(c.x + 1, c.y - 1),
                self.getCell(c.x, c.y + 1),
                self.getCell(c.x, c.y - 1),
                self.getCell(c.x - 1, c.y + 1),
                self.getCell(c.x - 1, c.y),
                self.getCell(c.x - 1, c.y - 1)]

    # returns empty by default, assumes LiDAR has not discovered it or the location is unoccupied
    def getCell(self, x, y):
        if x not in self.cellmap:
            self.cellmap[x] = {}
        if y not in self.cellmap[x]:
            self.cellmap[x][y] = Cell(x, y, 0)
        return self.cellmap[x][y]

    def march(self, c, start_weight):
        if (start_weight < 1):
            return
        for cell in self.getNeighbors(c):
            if cell.weight < start_weight:
                cell.weight = start_weight
                self.march(cell, self.decay(start_weight))

    def decay(self, w):
        return w/4

    def dijkstra(self, start_cell):
        pq = [(0), start_cell]
        heapify(pq)
        visited = set()

        distances = {}
        distances[start_cell] = 0

        while pq:
            curr_dist, curr_cell = heappop(pq)

            if curr_cell in visited:
                continue
            visited.add(curr_cell)

            for neighbor in self.getNeighbors(curr_cell):
                tentative_dist = curr_dist + neighbor.weight
                if neighbor not in distances:
                    distances[neighbor] = float("inf")
                if tentative_dist < distances[neighbor]:
                    distances[neighbor] = tentative_dist
                    heappush(pq, (tentative_dist, neighbor))

        predecessors = {cell: None for cell in distances.keys()}
        for cell, distance in distances.items():
            for neighbor in self.getNeighbors(cell):
                if distances[neighbor] == distance + neighbor.weight:
                    predecessors[neighbor] = cell
        return distances, predecessors

    def shortest_path(self, start, end):
        _, predecessors = self.dijkstra(start)
        path = []
        curr_cell = end
        while curr_cell:
            path.append(curr_cell)
            curr_cell = predecessors[curr_cell]
        path.reverse()
        return path


class Cell:
    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        self.weight = weight
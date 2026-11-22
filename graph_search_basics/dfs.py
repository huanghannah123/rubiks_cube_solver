from collections import defaultdict

# Adapted from the standard GeeksforGeeks DFS example
# (https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/).
# Change: a visited set, so vertices that only appear as edge targets
# (or have non-integer labels) no longer cause an IndexError.


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def dfs_util(self, v, visited, order):
        visited.add(v)
        order.append(v)
        print(v)

        for i in self.graph[v]:
            if i not in visited:
                self.dfs_util(i, visited, order)

    def DFS(self, v):
        order = []
        self.dfs_util(v, set(), order)
        return order


if __name__ == "__main__":

    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(2, 3)
    g.add_edge(3, 3)

    node = 2
    print("DFS starting from vertex %d:" % node)
    g.DFS(node)

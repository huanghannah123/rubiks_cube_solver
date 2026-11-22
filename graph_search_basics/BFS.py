from collections import defaultdict, deque

# Adapted from the standard GeeksforGeeks BFS example
# (https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/).
# Changes: a visited set (works for any vertex labels, including vertices
# that only appear as edge targets) and a deque for O(1) queue pops.


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def BFS(self, s):
        visited = {s}
        queue = deque([s])
        order = []

        while queue:
            s = queue.popleft()
            order.append(s)
            print(s)

            for i in self.graph[s]:
                if i not in visited:
                    visited.add(i)
                    queue.append(i)
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
    print("BFS starting from vertex %d:" % node)
    g.BFS(node)

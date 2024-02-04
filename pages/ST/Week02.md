---
title: "Graph Theory Fundamentals"
---

# Basics of Graphs

## Introduction to Graphs

Graphs, as fundamental data structures, play a crucial role in software testing. The inception of graph theory dates back to 1736, when Leonard Euler addressed the "Seven Bridges of Konigsberg" problem. Graphs find applications not only in computer science and data science but also in diverse fields such as sociology, economics, chemistry, and biology.

## Graph Components

A graph comprises vertices (nodes) denoted by the set $V$ and edges denoted by the set $E$, where $E$ is a subset of $V \times V$ (the cartesian product of $V$ with itself). Graphs can be classified as undirected (lacking arrows on edges) or directed (with edges having directions). Self-loops, edges connecting a vertex to itself, add a special characteristic. While graphs can be finite or infinite, finite graphs are preferred for testing purposes.

### Degree of a Vertex

The degree of a vertex is defined as the number of edges incident to it. In directed graphs, the degree is further categorized into in-degree (count of incoming edges) and out-degree (count of outgoing edges).

## Control Flow Graphs

Control flow graphs are essential in software testing for modeling program control flow. An illustrative example includes a control flow graph for an if-else statement.

## Path, Length, and Reachability

- **Path:** A path represents a sequence of vertices connected by edges.
- **Length of a Path:** It corresponds to the number of edges in a given path.
- **Reachability:** This concept determines whether a vertex or edge is reachable from another within the graph.

## Depth First Search (DFS) and Breadth First Search (BFS)

DFS and BFS are algorithms crucial for reachability analysis in graphs.

- **DFS (Depth First Search):** This algorithm explores as far as possible before backtracking.
- **BFS (Breadth First Search):** BFS explores level by level in a graph.

These algorithms are instrumental in solving various reachability problems in graph theory.

## Test Path and Feasibility

A test path is a sequence of vertices and edges starting from an initial vertex and ending at a final vertex. Feasible test paths are executable with valid test cases, while infeasible ones cannot be achieved.

## Visiting and Touring

- A test path **visits** a vertex or edge when it includes them in the sequence.
- **Touring** is an equivalent concept for vertices and edges.

## Test Requirements and Criteria

- **Test Requirements:** These specifications define properties to be tested, such as covering every if statement or loop.
- **Test Criteria:** Rules outlining how test requirements should be satisfied.

## Structural Coverage Criteria

Structural coverage criteria concentrate on graph structure without considering variables. An example is **Branch Coverage,** aiming to cover all branches in a graph.


# Graph Representation and Breadth-First Search

## Introduction to Graph Representation

### Graph Data Structures
In the context of software testing, graphs serve as fundamental data structures for implementing algorithms related to test case design. The lecture emphasizes the significance of representing graphs using matrices and lists, specifically, the adjacency matrix and the adjacency list.

### Representation Methods
Graphs can be represented using two primary methods: matrices and lists.

#### Adjacency List Representation
For each vertex in the graph, an array of lists is employed. This array contains lists corresponding to each vertex, enumerating its adjacent vertices. This representation proves advantageous for sparse graphs where not all vertices have extensive connections.

##### Example
Consider vertices $u$, $v$, and $w$. The adjacency list representation could be:

- $u$: {$v$, $w$}
- $v$: {$u$, $w$}
- $w$: {$u$}

#### Adjacency Matrix Representation
This method utilizes an $n \times n$ matrix, where $n$ is the number of vertices. A 0 or 1 is assigned to each matrix entry based on the presence or absence of an edge between the corresponding vertices.

##### Example
For the graph with vertices $u$, $v$, and $w$:
$$
\begin{matrix}
 & u & v & w \\
u & 0 & 1 & 1 \\
v & 1 & 0 & 1 \\
w & 1 & 1 & 0 \\
\end{matrix}
$$

## Breadth-First Search (BFS)

### Algorithm Overview
BFS is a traversal algorithm used to explore a graph in a breadth-first manner. The algorithm commences by assigning colors, distances, and parent pointers to vertices. It then employs a queue for traversing the graph, exploring adjacency lists, enqueuing adjacent vertices, and updating attributes.

#### BFS Tree
The algorithm constructs a BFS tree, representing the shortest paths from the source vertex. Each edge in the tree corresponds to the shortest path between vertices.

#### Queue Operations
The BFS algorithm relies on two fundamental operations: enqueue (insert) and dequeue (remove). These operations manage the vertices during the traversal process.

#### Vertex Attributes
- **Colors:** Vertices are initially white (unexplored), turn blue when enqueued, and finally black when explored.
- **Distance Attribute ($d$):** Represents the length of the shortest path from the source.
- **Parent Attribute ($\pi$):** Points to the predecessor vertex in the BFS tree.

### Example Execution of BFS
The lecture provides a step-by-step illustration of BFS execution using a sample graph. It outlines the process of enqueueing, dequeuing, and updating vertex attributes, resulting in the construction of the BFS tree.

### Analysis of BFS
The efficiency of BFS is analyzed in terms of its running time, which is linear, $O(v + e)$, where $v$ is the number of vertices and $e$ is the number of edges. BFS guarantees the identification of shortest paths in unweighted graphs.

#### Correctness Theorem
A correctness theorem is presented, asserting that BFS correctly explores all reachable vertices from the source and returns the shortest paths.

# Depth First Search (DFS) in Graphs

Depth First Search (DFS) is an algorithm used for traversing and exploring graphs in a systematic manner. It starts from a designated source vertex and explores as far as possible along each branch before backtracking. This exploration strategy is in contrast to the breadth-first search (BFS) algorithm. DFS provides valuable insights into the structure and connectivity of a graph.

## Overview of DFS

DFS operates by systematically exploring edges out of the most recently discovered vertex with unexplored edges. The algorithm assigns colors to vertices to track their exploration status: 

- **White:** Undiscovered
- **Gray:** Discovered but not fully explored
- **Black:** Fully explored

Additionally, timestamps in the form of discovery and finish times are assigned to vertices during the process, offering further information about the graph.

### Pseudocode for DFS

```plaintext
DFS(G):
  for each vertex u in G:
    color[u] = \text{white}
    parent[u] = \text{nil}
  time = 0
  for each vertex u in G:
    if color[u] is \text{white}:
      DFS-Visit(u)

DFS-Visit(u):
  time = time + 1
  discovery[u] = time
  color[u] = \text{gray}
  for each vertex v adjacent to u:
    if color[v] is \text{white}:
      parent[v] = u
      DFS-Visit(v)
  color[u] = \text{black}
  time = time + 1
  finish[u] = time
```

### Properties of DFS

1. **Parenthesis Theorem:**
   - Discovery times are always less than finish times, creating nested parenthesis intervals.

2. **White Path Theorem:**
   - When first encountering a vertex, there exists a path of white-colored vertices leading to it.

3. **Edge Classification:**
   - **Tree Edges:** Form the DFS tree.
   - **Forward Edges:** Connect descendants to ancestors.
   - **Backward Edges:** Connect ancestors to descendants.
   - **Cross Edges:** Connect vertices unrelated in the DFS tree.

# Strongly Connected Components (SCC)

Strongly Connected Components are subsets of vertices in a directed graph where every pair of vertices is reachable from each other. DFS can be employed to efficiently identify these components.

## Algorithm for SCC

1. **Run DFS on the graph to compute finish times.**
   - The finish times denote the order in which vertices complete their exploration.

2. **Compute the transpose of the graph.**
   - Reverse the direction of edges in the graph.

3. **Run DFS on the transpose graph in reverse finish time order.**
   - Explore vertices in the order of decreasing finish times obtained in step 1.

4. **Identify SCCs based on DFS trees in the second run.**
   - Each DFS tree represents a strongly connected component.

# Structural Coverage Criteria

## Introduction

In the realm of software testing, structural coverage criteria play a pivotal role in ensuring the thorough examination of software artifacts. This lecture delves into various structural coverage criteria applied to graphs, elucidating their significance in the testing process. The focus lies on node coverage, edge coverage, edge pair coverage, and prime path coverage. Additionally, the challenges associated with achieving complete path coverage, especially in the presence of loops, are explored.

## Nodes, Edges, and Paths

A graph modeling a software artifact comprises nodes (or vertices) and edges, representing the structural entities. Paths in the graph manifest as sequences of nodes and edges, forming the basis for coverage criteria.

### Node Coverage

**Definition**: The test requirement for node coverage entails generating test cases that visit every node in the graph at least once. A test set, denoted as $T$, satisfies node coverage if, for every reachable node, there exists a test path in $T$ that visits that node.

### Edge Coverage

**Definition**: Edge coverage necessitates visiting every edge in the graph at least once. The test requirement can be expressed as executing each reachable path of length up to 1. It aims to subsume node coverage, ensuring that the paths of length 0 (nodes) and length 1 (edges) are covered.

### Edge Pair Coverage

**Definition**: This criterion extends coverage to pairs of edges. Test paths of length 2 (pairs of edges) are considered, ensuring coverage of all possible edge pairs. Edge pair coverage aims to encompass both edge and node coverage.

## Prime Path Coverage

Prime paths are maximal simple paths within a graph, devoid of internal loops. Enumerating prime paths provides an effective coverage criterion, addressing challenges associated with loops in control flow graphs.

**Definition**: A prime path is a simple path that is not a proper subpath of any other simple path. It serves as a maximal simple path within the graph.

### Touring with Side Trips and Detours

To address scenarios where prime paths might necessitate traversing loops, two concepts are introduced:

1. **Side Trips**:

   - **Definition**: A test path $p$ is considered to have a side trip towards a subpath $q$ if every edge in $q$ appears in $p$ in the same order.
   
   - **Explanation**: Side trips allow for the traversal of a subpath $q$ within the main test path $p$, ensuring that the edges in $q$ are followed in the same sequence as they appear in $p$.
   
   - **Purpose**: The concept of side trips is particularly useful when dealing with loops in control flow graphs. It allows for the inclusion of loop-related paths within the main test path, contributing to a more practical and feasible testing scenario.

2. **Detours**:

   - **Definition**: A test path $p$ is considered to have a detour towards a subpath $q$ if every node in $q$ appears in $p$ in the same order.
   
   - **Explanation**: Detours enable the traversal of a subpath $q$ within the main test path $p$, ensuring that the nodes in $q$ are visited in the same sequence as they appear in $p$.
   
   - **Purpose**: Similar to side trips, detours offer a mechanism to accommodate loops in control flow graphs during testing. They provide flexibility by allowing the inclusion of paths related to loops, contributing to a more realistic testing approach.

These concepts help mitigate infeasibility concerns, allowing for more practical testing scenarios.

## Round Trip Coverage

Round trips are prime paths that commence and culminate at the same node. Coverage criteria for round trips include:

- **Simple Round Trip Coverage**: Ensures at least one round trip for each reachable node.

- **Complete Round Trip Coverage**: Requires coverage of all possible round trip paths within the graph.


# Graphs for Structural Coverage Criteria

## Overview
This section delves into the meticulous process of deriving test requirements and paths to achieve structural coverage criteria within software testing. The primary focus is on graphs representing software artifacts.

## Structural Coverage Criteria

### 1. Node Coverage and Edge Coverage

#### Test Requirements
- **Node Coverage:** Set of nodes in the graph.
- **Edge Coverage:** Set of edges in the graph.

#### Test Paths
Utilize Breadth-First Search (BFS) from an initial node to cover reachable nodes and edges systematically.

### 2. Edge Pair Coverage

#### Test Requirements
- All paths of length $2$ in the graph.

#### Algorithm
Enumerate pairs of edges by traversing nodes and adjacency lists. This involves considering nodes $u$ and $v$, exploring their adjacency lists, and forming pairs $u \to v \to w$, where $w$ is in the adjacency list of $v$.

### 3. Specified Path Coverage

#### Test Requirements
- Set of specified paths provided by a tester.

#### Algorithm
Modify BFS for graphs without loops to achieve specified path coverage. This entails adapting BFS to include specified paths in the traversal.

## Prime Path Coverage

### Prime Paths
Prime paths are defined as maximal simple paths in a graph.

### Test Requirements
Enumerate all prime paths in the graph.

### Prime Path Enumeration Algorithm

#### 1. Algorithm Overview
- Enumerate simple paths in ascending order of length.
- Choose prime paths among the enumerated paths.

#### 2. Enumeration Process
- Paths of length $0$ (vertices) are considered, marking unextendable paths with "!".
- Paths of length $1$ (edges) are enumerated, marking unextendable and simple cycle paths with "!" and "*".
- Extension of paths to obtain length $2$ paths is performed, and paths are marked accordingly.
- The process continues until paths of length $\text{mod } v - 1$ are reached, with markings indicating path characteristics.

#### 3. Result
Obtain all prime paths as test requirements.

## Test Paths for Prime Path Coverage

### Algorithm Overview
- Start with the longest prime path.
- Extend each path to the initial and final vertices.
- Utilize traversal algorithms to extend paths systematically.

### Example
For a graph with multiple loops, initiate the process with the longest prime path and extend it to cover all instances of loops.

### Optimality Challenge
Achieving optimal test paths is generally intractable. Symbolic execution, an advanced technique, can be explored for improved test path generation.

# Conclusion

The module provide a comprehensive overview of graph basics, control flow graphs, DFS, BFS, graph representation, and structural coverage criteria in the context of software testing. The detailed explanations and examples make these complex topics accessible, emphasizing their significance in designing effective test cases.

## Points to Remember

1. **Graph Basics:**
   - Graphs are fundamental data structures with applications in various fields.
   - Graphs consist of vertices and edges, and their components include degree, control flow graphs, paths, and reachability.

2. **DFS and BFS:**
   - DFS and BFS are essential algorithms for reachability analysis in graphs.
   - DFS explores as far as possible before backtracking, while BFS explores level by level.
   - They play a crucial role in solving reachability problems.

3. **Graph Representation:**
   - Graphs can be represented using adjacency matrices or lists.
   - Adjacency list representation is advantageous for sparse graphs.

4. **Breadth-First Search (BFS):**
   - BFS explores a graph in a breadth-first manner, constructing a BFS tree.
   - It guarantees the identification of shortest paths in unweighted graphs.

5. **Depth First Search (DFS):**
   - DFS explores graphs systematically, assigning colors to vertices.
   - It provides insights into the structure and connectivity of a graph.

6. **Strongly Connected Components (SCC):**
   - SCCs are subsets of vertices in a directed graph where every pair of vertices is reachable from each other.
   - DFS is used to efficiently identify SCCs.

7. **Structural Coverage Criteria:**
   - Node coverage, edge coverage, edge pair coverage, prime path coverage, best effort touring, and round trip coverage are discussed.
   - These criteria ensure thorough evaluation of software artifacts in testing.

8. **Test Paths for Prime Path Coverage:**
   - Enumerating prime paths involves considering simple paths in ascending order of length.
   - The algorithm systematically extends paths to cover initial and final vertices.

9. **Optimality Challenge:**
   - Achieving optimal test paths is generally intractable.
   - Symbolic execution is an advanced technique that can be explored for improved test path generation.


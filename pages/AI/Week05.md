# Finding Optimal TSP Tours

## Introduction to Branch and Bound Algorithm

The Branch and Bound algorithm is a fundamental technique employed in optimization problems, particularly in combinatorial optimization, to efficiently find the optimal solution within a search space. This algorithm systematically explores the solution space by dividing it into smaller subspaces and effectively pruning unpromising branches to focus on regions likely to contain the optimal solution.

## Overview of the Example Problem

To illustrate the workings of the Branch and Bound algorithm, let's consider a specific optimization problem: finding the shortest tour among five cities. The objective is to determine the optimal route that visits each city exactly once and returns to the starting point, minimizing the total distance traveled.

## Initialization and Lower Bound Estimates

The Branch and Bound algorithm begins with an initial set of all possible candidates, denoted as $S_0$. Each candidate represents a potential solution or partial tour. A crucial aspect of the algorithm is the computation of lower bound estimates, which serve as a guide for pruning the search space.

The lower bound estimates are calculated based on certain constraints, such as:

1. **Constraint 1**: Each city should be visited exactly once in the tour.
2. **Constraint 2**: Subtours that violate optimality should be excluded from consideration.

## Trade-off Between Accuracy and Computation Time

One of the key considerations in the Branch and Bound algorithm is the trade-off between the accuracy of estimates and the computation time required to compute them. More accurate estimates may necessitate additional computational effort but can lead to better pruning of the search space, resulting in faster convergence to the optimal solution.

## Iterative Refinement of Candidates

The algorithm iteratively refines the candidates in the search space by either including or excluding edges in the tour. This process continues until fully refined nodes, representing complete tours, are identified.

### Inclusion and Exclusion of Edges

At each iteration, the algorithm considers whether to include or exclude specific edges in the tour. This decision is based on the impact it has on the overall cost of the tour and the satisfaction of optimality constraints.

### Pruning Unpromising Branches

The Branch and Bound algorithm prunes unpromising branches of the search tree based on the lower bound estimates. By eliminating regions of the search space that cannot contain the optimal solution, the algorithm focuses its efforts on exploring more promising areas.

## Termination Condition

The algorithm terminates when a fully refined node with the lowest estimated cost is found. This fully refined node represents an actual tour with its actual cost and guarantees that no other solution in the search space can have a lower actual cost.

## Trade-off Between Search and Estimation

A fundamental trade-off exists between the search process and the accuracy of estimation in the Branch and Bound algorithm. Higher estimated costs on open nodes delay exploration, as nodes with lower actual costs are prioritized for refinement.

## Branch and Bound within the State Space Search Framework

The Branch and Bound algorithm operates within the framework of State Space Search, which involves systematically exploring the search space while utilizing lower bound estimates to guide the search towards the optimal solution. This approach ensures that the algorithm efficiently converges to the optimal solution while minimizing computational overhead.

# Optimal Pathfinding Algorithms

## Introduction
In the realm of artificial intelligence (AI), the exploration of search methods is crucial for problem-solving. These methods aim to navigate through complex state spaces to find optimal solutions efficiently. In this discussion, we delve into the concepts and algorithms associated with search methods, focusing particularly on the Branch and Bound algorithm and its comparison with Dijkstra's Algorithm. Additionally, we explore the integration of these concepts in the A\* algorithm for enhanced efficiency in problem-solving.

## Branch and Bound Algorithm

### Overview
The Branch and Bound algorithm serves as a fundamental approach for finding optimal solutions in state spaces with edge costs. Unlike blind search algorithms, which may overlook the optimality of solutions, Branch and Bound ensures the exploration of all potential solutions while organizing the search space systematically.

### Objective
The primary objective of Branch and Bound is to identify the optimal solution within a state space characterized by edge costs. By extending or refining partial paths systematically, the algorithm aims to prune parts of the search space that are unlikely to contain better solutions.

### Comparison with Blind Search Algorithms
Branch and Bound distinguishes itself from blind search algorithms such as Depth First Search (DFS) and Breadth First Search (BFS) in terms of its approach to solution exploration.

#### Depth First Search (DFS)
DFS ventures optimistically into the search space, prioritizing deep exploration without guaranteeing the optimality of solutions.

#### Breadth First Search (BFS)
BFS, on the other hand, focuses on shallow candidates, aiming for the shortest path from the start node to the goal node. However, it may not always guarantee the optimal solution.

#### Depth First Iterative Deepening
Depth First Iterative Deepening combines elements of DFS and BFS, mimicking BFS while operating as a depth-first algorithm. Despite its similarities to BFS, it may still lack the assurance of optimality.

### Heuristic Functions
In the quest for optimal solutions, heuristic functions play a crucial role in estimating distances to the goal. By leveraging these estimates, algorithms can expedite the search process by prioritizing nodes that appear to be closer to the goal.

## Dijkstra's Algorithm

### Overview
Dijkstra's Algorithm represents another prominent approach to finding optimal paths, particularly in graph-based problems. It focuses on determining the shortest paths from a source node to all other nodes within the graph.

### Key Steps
Dijkstra's Algorithm follows a systematic procedure to identify the shortest paths within a graph, encompassing the following key steps:

1. **Initialization**: Assign infinite cost estimates to all nodes except the start node.
2. **Exploration**: Select the cheapest white node and update its cost, marking it as black to signify its exploration.
3. **Relaxation**: Examine the neighbors of the newly explored node and update their costs if a cheaper path is discovered.
4. **Path Tracking**: Maintain parent pointers to track the optimal path from the start node to each explored node.

### Application to Problem Solving
In problem-solving scenarios, Dijkstra's Algorithm proves effective in systematically exploring paths within a graph to identify the shortest path to each node. By iteratively updating node costs and parent pointers, the algorithm converges towards the optimal solution efficiently.

### Comparison with Branch and Bound
While both Branch and Bound and Dijkstra's Algorithm aim for optimal solutions, they exhibit differences in their approach and applicability to various problem domains.

#### Directionality
Branch and Bound lacks a sense of directionality, often exploring the entire search space extensively before reaching the optimal solution. In contrast, Dijkstra's Algorithm systematically explores paths from the start node to all other nodes, ensuring optimality but without a predetermined direction.

#### Efficiency vs. Optimality
While Dijkstra's Algorithm guarantees optimality in pathfinding, it may sacrifice efficiency due to its exhaustive exploration of paths. Branch and Bound, on the other hand, balances optimality with efficiency by organizing the search space strategically and pruning unlikely paths.

# A-star Algorithm

The A-star algorithm is a widely-used search algorithm in artificial intelligence, particularly in problem-solving scenarios where finding an optimal path from a start state to a goal state is required. It combines the efficiency of heuristic search methods with the optimality guarantees of traditional algorithms like Dijkstra's algorithm.

## Introduction to A-star Algorithm

A-star, also known as A* algorithm, was introduced by Hart, Nilsson, and Raphael in 1968 at the Stanford Research Institute (now called SRI International). It is considered an extension of Dijkstra's algorithm, incorporating heuristics to improve performance. The algorithm aims to find the shortest path from a start node to a goal node in a graph, taking into account both the actual cost incurred (g-value) and an estimated cost to reach the goal (h-value).

## Components of A-star Algorithm

The A-star algorithm consists of several key components:

1. **Heuristic Function (h(n)):** A heuristic function estimates the cost from a given node to the goal node. It provides a guiding heuristic to help prioritize nodes during the search process.
   
   $$ h(n) $$

2. **Actual Cost (g(n)):** The actual cost represents the cumulative cost of reaching a node from the start node along the current path. It is used to compute the total cost of a solution path.

   $$ g(n) $$

3. **Combined Cost (f(n)):** The combined cost, denoted as f(n), is the sum of the actual cost (g(n)) and the heuristic cost (h(n)). It represents the estimated total cost of the solution path passing through node n.

   $$ f(n) = g(n) + h(n) $$

4. **Open List and Closed List:** A-star maintains two lists: the open list, which contains nodes that are candidates for expansion, and the closed list, which stores nodes that have already been explored.

## Algorithm Workflow

The A-star algorithm follows a systematic workflow to explore the search space and find the optimal path:

1. **Initialization:**
   - Set the default value of g for every node to positive infinity.
   - Set the parent of the start node to null.
   - Initialize the g-value of the start node to 0.
   - Compute the f-value of the start node as the sum of its g-value and h-value.

   $$ f(S) = g(S) + h(S) $$

   - Add the start node to the open list.

2. **Main Loop:**
   - While the open list is not empty:
     - Select the node with the lowest f-value from the open list (denoted as N) for expansion.
     - Remove N from the open list and add it to the closed list.
     - Check if N is the goal node. If so, return the solution path.
     - Generate the neighbors of N and compute their f-values.
     - Update the g-values and parent pointers for the neighbors if a better path is found.
     - Manage three cases:
       1. If a neighbor node is new, add it to the open list.
       2. If a neighbor node is already on the open list, update its g-value if a better path is found.
       3. If a neighbor node is on the closed list, propagate the improvement to its children.

3. **Propagation of Improvement:**
   - If a node with an improved path is on the closed list, propagate the improvement to its children recursively.
   - For each neighbor of the node, check if the new path has a lower cost than the previous one.
   - Update the parent pointer and g-value of the neighbor node if necessary.
   - Recursively propagate the improvement if the neighbor node is also on the closed list.

## Illustrative Example

Consider a small example to illustrate how A-star expands nodes and updates costs:

1. Default value of $g$ for every node is +∞.
2. $parent(S)$ ← null.
3. $g(S)$ ← 0.
4. $f(S)$ ← $g(S) + h(S)$.
5. $OPEN$ ← $S$: $\left[\right]$.
6. $CLOSED$ ← empty list.

7. While $OPEN$ is not empty:
    - $N$ ← remove node with lowest $f$ value from $OPEN$.
    - Add $N$ to $CLOSED$.

    11. If $GOAL\text{-}TEST(N) = \text{TRUE}$:
        - Return $RECONSTRUCT\text{-}PATH(N)$.

    12. For each $M$ in $MOVE\text{-}GEN(N)$:
        - If $g(M) > g(N) + k(N, M)$:
            * $parent(M)$ ← $N$
            * $g(M)$ ← $g(N) + k(N, M)$
            * $f(M)$ ← $g(M) + h(M)$

        16. If $M$ is in $OPEN$:
            - Continue.

        18. Else if $M$ is in $CLOSED$:
            - PROPAGATE\text{-}IMPROVEMENT(M).

        20. Else add $M$ to $OPEN$ ▶️ $M$ is new.

22. Return empty list.

## Comparison with Best-First Search

A-star differs from traditional best-first search methods in its approach to exploring the search space. While best-first search prioritizes nodes based solely on heuristic values, A-star combines heuristic information with actual path costs to guide the search efficiently towards the goal state. This combination allows A-star to find optimal solutions while also considering the efficiency of the search process.

# Admissibility of A\*

## Introduction to A star Algorithm
The A star algorithm, a fundamental method in artificial intelligence, amalgamates the efficacies of Dijkstra's shortest path algorithm with heuristic approaches to expedite the discovery of optimal solutions within search spaces. This algorithm is chiefly employed in problems where finding the optimal path from a start node to a goal node is paramount.

## Conditions for Guaranteeing Optimal Solutions
In the quest for optimality, it's imperative to discern the conditions under which the A star algorithm operates. The algorithm considers three critical scenarios during its execution:

1. **Adding New Nodes to Open:** This entails the incorporation of new nodes into the open set for further exploration.
2. **Updating Paths to Nodes Already on Open:** Here, the algorithm revisits nodes already present in the open set and updates their paths if more efficient alternatives are discovered.
3. **Finding Newer Paths to Nodes on Closed:** This scenario involves the potential for uncovering superior paths to nodes that have been closed off.

## Optimal Values and Heuristic Functions
To facilitate a deeper understanding of the algorithm's behavior, it's essential to introduce the concept of optimal values and heuristic functions:

- $g^*(n)$: Represents the optimal cost from the start node to node $n$.
- $h^*(n)$: Signifies the optimal cost from node $n$ to the goal node.
- $f^*(n)$: Denotes the optimal cost of a path from the start to the goal via node $n$.

These values serve as guiding principles for the algorithm's decision-making process, albeit their actual values may remain unknown during execution. Notably, the optimal cost from the start to a given node $n$ ($g^*(n)$) typically falls short of the actual cost ($g(n)$), owing to the algorithm's inherent uncertainty regarding the optimal path.

## Admissibility of A star
An algorithm earns the badge of admissibility when it reliably furnishes the optimal solution, provided such a solution exists. The A star algorithm, distinguished by its appended star superscript, is hailed as an admissible method. To qualify as admissible, the algorithm must satisfy the following conditions:

1. **Finite Branching Factor:** The algorithm must grapple with a finite branching factor, even in scenarios where the total number of nodes in the graph spans to infinity.
2. **Cost Condition:** Each edge within the graph must exhibit a cost greater than a predetermined minuscule constant $\epsilon$.
3. **Heuristic Function:** The heuristic function employed by the algorithm must consistently underestimate the distance to the goal ($h(n) \leq h^*(n)$).

## Example Illustrating Heuristic Functions
To elucidate the impact of heuristic functions on the algorithm's decision-making process, consider a hypothetical scenario where the algorithm confronts the choice between two nodes, $P$ and $Q$, each accompanied by known and estimated costs. In the presence of an overestimating function (yielding higher heuristic values), the algorithm may erroneously opt for $Q$ as the proximate node to the goal. Conversely, an underestimating function (with lower heuristic values) would correctly identify $P$ as the closer node, consequently steering the algorithm towards the optimal solution.

# Proof of Admissibility in A* Algorithm

A fundamental aspect in the study of search algorithms is ensuring their termination and correctness, particularly in the context of finite and infinite graphs. The A* algorithm, a popular choice for pathfinding problems, demonstrates such properties through a rigorous proof of admissibility.

## Termination for Finite Graphs

In examining the termination of A* for finite graphs, we consider the behavior of the algorithm within each cycle of its main loop. Central to this discussion is the process by which A* selects nodes from the "open" list and subsequently moves them to the "closed" list. This mechanism ensures that each node in the graph is visited at most once during the algorithm's execution. Crucially, since the number of nodes in the graph is finite, A* will terminate after a finite number of cycles. This termination guarantees that the algorithm will definitively report whether a path to the goal exists or not within the given graph.

## Open List Contains Optimal Path Node

A significant property of A* lies in its ability to maintain an optimal path to the goal node within its "open" list. By construction, the algorithm ensures that at any given point during its execution, there exists a node from the optimal path on the "open" list. This assertion stems from the iterative nature of A*, where nodes along the optimal path are successively added to and removed from the "open" list. Consequently, the "open" list consistently contains a node, denoted as $n'$, from the optimal path, thereby facilitating the algorithm's progress towards identifying the optimal solution.

Moreover, it is imperative to note that the $f$-value of this node $n'$ does not exceed the optimal cost to the goal. This assertion is grounded in the heuristic nature of A*, where the evaluation function $f(n)$ comprises the sum of the actual cost $g(n)$ and the heuristic estimate $h(n)$. As $n'$ lies on the optimal path, its actual cost $g(n')$ coincides with the optimal cost $g^*(n')$. Additionally, the underestimation property of the heuristic function $h(n)$ ensures that $h(n') \leq h^*(n')$, where $h^*(n')$ represents the true cost from $n'$ to the goal. Consequently, the $f$-value of $n'$ remains less than or equal to the optimal cost, thereby affirming its significance in the search process.

## Finding a Path in Infinite Graphs

In contrast to finite graphs, the termination and correctness of A* in infinite graphs pose unique challenges. However, through careful analysis, we establish the algorithm's efficacy in finding a path even in scenarios with infinite graph structures. Central to this discussion is the notion of epsilon, a finite value utilized to ensure progress in the search process.

A* employs a strategy wherein nodes with the lowest $f$-value are prioritized for expansion. As the algorithm explores various paths within the graph, the actual cost $g(n)$ of each partial solution incrementally increases by a finite value greater than epsilon. This incremental increase in cost serves to prevent the existence of infinite paths with finite cost, thereby guiding the algorithm towards a definitive solution. Furthermore, given the finite nature of the branching factor, A* ensures that only a finite number of partial solutions, cheaper than the optimal cost, are considered during its execution.

This meticulous selection and evaluation process, coupled with the epsilon threshold, enable A* to traverse the graph effectively, eventually converging upon a solution path to the goal node. Thus, even in scenarios with infinite graphs, A* exhibits a robust capability to identify a path to the goal, provided finite cost constraints are met.

## Finding the Least Cost Path

The ultimate objective of A* lies in identifying the least cost path to the goal node. In pursuit of this goal, the algorithm employs a proof by contradiction to establish the optimality of its solution. By assuming the termination of A* without finding an optimal cost path, we derive a contradiction that invalidates such an assumption. This contradiction arises from the fundamental properties of A*, wherein the algorithm's selection criteria prioritize nodes with lower $f$-values, thereby guiding it towards the optimal solution. Consequently, A* terminates only upon discovering the optimal cost path to the goal, reaffirming its efficacy in pathfinding scenarios.

## Node Expansion and Heuristic Functions

A critical aspect influencing the behavior of A* is the selection and evaluation of nodes during its execution. Notably, for every node expanded by A*, its $f$-value remains bounded by the optimal cost, irrespective of its position relative to the optimal path. This property underscores the algorithm's adherence to admissible heuristics, where the estimation of node costs remains consistent with the true optimal cost.

Central to this discussion is the role of heuristic functions in guiding A* towards the goal node. These functions provide estimates of the remaining cost from a given node to the goal, thereby influencing the selection of nodes for expansion. Importantly, A* leverages heuristic functions that consistently underestimate the true cost, ensuring the optimality of its solution while traversing the search space.

## Comparison of Heuristic Functions

A noteworthy aspect of A* lies in its sensitivity to the quality of heuristic functions utilized in the search process. By comparing different heuristic functions, we gain insights into their impact on the algorithm's efficiency and performance. Specifically, heuristic functions that provide more informed estimates of node costs tend to result in faster convergence towards the optimal solution.

In comparing two admissible versions of A* utilizing distinct heuristic functions, denoted as $h_1$ and $h_2$, we observe a direct relationship between the informativeness of the heuristic and the algorithm's search efficiency. Notably, if $h_2$ consistently provides higher estimates than $h_1$ for all nodes, it is deemed more informed and closer to the true optimal cost. Consequently, A* employing $h_2$ is expected to explore a smaller search space and converge faster towards the optimal solution compared to its counterpart utilizing $h_1$.

This relationship underscores the significance of heuristic selection in optimizing the performance of A* and highlights the potential trade-offs between search efficiency and heuristic informativeness.

## Variations of A*

While the basic A* algorithm provides a robust framework for pathfinding, variations and adaptations exist to address specific challenges and optimization objectives. These variations aim to enhance the algorithm's efficiency and effectiveness in diverse problem domains, often at the cost of sacrificing certain properties such as admissibility.

One such variation involves the exploration of leaner and meaner versions of A*, wherein the emphasis is placed on minimizing either space or time complexity, or both, while maintaining a degree of admissibility. These adaptations leverage insights from the quality of the available heuristic function to tailor the search process to specific requirements. By striking a balance between computational resources and solution optimality, these variations offer tailored solutions to pathfinding problems in various contexts.

In summary, the proof of admissibility in A* algorithm underscores its effectiveness in traversing finite and infinite graphs to identify optimal paths to the goal node. Through meticulous analysis and adherence to admissible heuristics, A* exhibits robustness and efficiency in pathfinding scenarios, making it a versatile tool in algorithmic problem-solving.

# Conclusion

In this comprehensive exploration, we delved into the intricacies of finding optimal solutions in various problem domains, particularly focusing on the Branch and Bound algorithm and the A* algorithm. We began by elucidating the fundamental principles of Branch and Bound, highlighting its systematic approach to optimizing solutions within a search space. Through an illustrative example, we demonstrated how the algorithm refines candidates iteratively while pruning unpromising branches to converge towards the optimal solution efficiently.

Subsequently, we transitioned to an in-depth discussion on optimal pathfinding algorithms, where we introduced Dijkstra's Algorithm alongside A*. By comparing their methodologies and applications, we unveiled the unique characteristics and trade-offs associated with each algorithm, emphasizing the role of heuristic functions in enhancing search efficiency.

Furthermore, we explored the admissibility of the A* algorithm, elucidating its termination conditions and the properties that guarantee optimality in both finite and infinite graphs. Through meticulous analysis and proof, we established the efficacy of A* in traversing search spaces while maintaining admissibility and efficiency.

In essence, the journey through optimal pathfinding algorithms underscores the synergy between theoretical principles and practical applications, offering valuable insights into algorithmic problem-solving across diverse domains.

# Points to Remember

1. **Branch and Bound Algorithm**:

   - Utilized in optimization problems to find optimal solutions within a search space.
   - Systematically explores the solution space by refining candidates and pruning unpromising branches.
   - Initialization involves generating initial candidates and computing lower bound estimates.
   - Trade-off between accuracy of estimates and computation time influences algorithm performance.
   - Iterative refinement of candidates involves inclusion/exclusion of edges and pruning unpromising branches.
   - Termination condition is met when a fully refined node with the lowest estimated cost is found.

2. **Optimal Pathfinding Algorithms**:

   - Branch and Bound vs. Dijkstra's Algorithm: Branch and Bound focuses on global optimization while Dijkstra's Algorithm finds shortest paths from a source node.
   - A* Algorithm combines the efficiency of heuristic search with the optimality guarantees of Dijkstra's Algorithm.
   - Components of A* include heuristic function, actual cost, combined cost, open list, and closed list.
   - Algorithm workflow involves initialization, main loop, and propagation of improvement.
   - Comparison with Best-First Search highlights the incorporation of actual path costs in A*.
   
3. **Admissibility of A* Algorithm**:

   - A* algorithm guarantees optimal solutions under specific conditions.
   - Termination for finite graphs is ensured by visiting each node at most once.
   - Open list contains node from optimal path, with $f$-value not exceeding optimal cost.
   - Epsilon threshold prevents infinite paths with finite cost in infinite graphs.
   - Proof of admissibility relies on finite branching factor, cost condition, and underestimating heuristic function.
   
4. **Variations of A***:

   - Adaptations of A* algorithm tailor its performance to specific optimization objectives.
   - Leaner and meaner versions focus on minimizing space or time complexity while maintaining admissibility.
   - Heuristic selection impacts search efficiency and convergence towards optimal solutions.


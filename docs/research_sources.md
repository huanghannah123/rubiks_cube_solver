# Research Sources: Solving the 2x2x2 Rubik's Cube with Search and AI

All sources are free to access.

Type key: [PR] = peer-reviewed, [BOOK] = textbook, [PRE] = preprint or technical report (not peer-reviewed), [WEB] = expert website, [DOC] = official documentation.
Every "Free:" link opens the full text without a login or payment.

## Part A: Search foundations (BFS, A*, IDA*)

**1.** Poole, D. L., & Mackworth, A. K. (2023). *Artificial Intelligence: Foundations of Computational Agents* (3rd ed.). Cambridge University Press. [BOOK]
Free: https://artint.info/3e/html/ArtInt3e.Ch3.html (Chapter 3, Searching for Solutions)
Summary: A university AI textbook that its authors make free online. Chapter 3 explains state spaces, graph search, breadth-first search, depth-first and iterative deepening search, A*, and heuristics.
Use: Background and definitions for every search algorithm in the project.

**2.** Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics, 4*(2), 100-107. https://doi.org/10.1109/TSSC.1968.300136 [PR]
Free: https://people.stfx.ca/jdelamer/courses/csci-564/_downloads/b2220c66675ddde471ca1795147b8e86/A_Formal_Basis_for_the_Heuristic_Determination_of_Minimum_Cost_Paths.pdf
Summary: The original A* paper. It ranks nodes by f(n) = g(n) + h(n). It proves that if the estimate h never overestimates, A* always finds the shortest path.
Use: Primary source for A*.

**3.** Korf, R. E. (1985). Depth-first iterative-deepening: An optimal admissible tree search. *Artificial Intelligence, 27*(1), 97-109. https://doi.org/10.1016/0004-3702(85)90084-0 [PR]
Free: https://cse.sc.edu/~mgv/csce580f12/gradPres/korf_IDAStar_1985.pdf
Summary: Introduces iterative deepening and IDA*. It runs repeated depth-first searches with a rising cost limit, so it finds optimal solutions like A* while using very little memory.
Use: Primary source for IDA*.

**4.** Korf, R. E. (2000). Recent progress in the design and analysis of admissible heuristic functions. *Proceedings of the 17th National Conference on Artificial Intelligence (AAAI-2000)*, 1165-1170. [PR]
Free: https://cdn.aaai.org/AAAI/2000/AAAI00-212.pdf
Summary: A short overview by the leading researcher in the field. It explains how pattern databases made it possible to solve Rubik's Cube and the 24-Puzzle optimally. It also shows how to predict IDA*'s running time from the distribution of heuristic values: good heuristics reduce the effective search depth, not the branching factor.
Use: Explains why heuristic quality matters; useful when analyzing your solver's speed.

**5.** Python Software Foundation. (n.d.). collections.deque and heapq. *Python 3 Documentation*. [DOC]
Free: https://docs.python.org/3/library/collections.html#collections.deque and https://docs.python.org/3/library/heapq.html
Summary: Official documentation for the two data structures the code uses. deque is a fast first-in-first-out queue for BFS. heapq is a priority queue that always returns the smallest f-value, which A* needs.
Use: Methods section.

## Part B: The Rubik's Cube as a search problem

**6.** Scherphuis, J. (n.d.). The 2x2x2 Pocket Cube. *Jaap's Puzzle Page*. [WEB]
Free: https://www.jaapsch.net/puzzles/cube2.htm
Summary: Counts the 2x2x2's 3,674,160 positions. It shows that every position can be solved in at most 11 moves in the half-turn metric (14 in the quarter-turn metric), with a full distance table.
Use: The size of your BFS search space and a check on solution lengths.

**7.** Rokicki, T., Kociemba, H., Davidson, M., & Dethridge, J. (2014). The diameter of the Rubik's Cube group is twenty. *SIAM Review, 56*(4), 645-670. https://doi.org/10.1137/14097375X [PR]
Free: https://epubs.siam.org/doi/10.1137/14097375X (marked Free Access) and https://cube20.org/
Summary: Proves every 3x3x3 position can be solved in 20 moves or fewer ("God's Number"). The authors used symmetry, splitting the problem into about 2.2 billion sets, and about 35 CPU-years of computing donated by Google.
Use: Shows how the methods scale from the 2x2x2 to the 3x3x3 and where brute force hits its limits.

**8.** Kunkle, D., & Cooperman, G. (2007). Twenty-six moves suffice for Rubik's Cube. *Proceedings of ISSAC '07*, 235-242. ACM. https://doi.org/10.1145/1277548.1277581 [PR]
Free: https://khoury.northeastern.edu/home/gene/papers/rubik.pdf
Summary: Proves 26 moves always suffice for the 3x3x3, using group theory plus large-scale breadth-first search on a computer cluster, with data stored on disk.
Use: Example of BFS at huge scale.

**9.** Kociemba, H. (n.d.). The two-phase algorithm. [WEB]
Free: https://kociemba.org/math/twophase.htm
Summary: The inventor explains the algorithm behind most fast cube solvers. It works in two phases, each using IDA* with lookup tables, and finds near-optimal solutions in a fraction of a second.
Use: Contrasts optimal solving with fast "good enough" solving.

## Part C: Pattern databases

**10.** Culberson, J. C., & Schaeffer, J. (1998). Pattern databases. *Computational Intelligence, 14*(3), 318-334. https://doi.org/10.1111/0824-7935.00065 [PR]
Free: https://linux.ime.usp.br/~henrick/pattern.pdf
Summary: Introduces pattern databases. Solve every position of a simplified version of the puzzle with BFS, store the move counts in a table, and use the table as a heuristic that never overestimates. On the 15-Puzzle it searched over 1000 times fewer nodes than the Manhattan-distance heuristic.
Use: Primary source for pattern databases.

**11.** Korf, R. E. (1997). Finding optimal solutions to Rubik's Cube using pattern databases. *Proceedings of AAAI-97*, 700-705. [PR]
Free: https://cdn.aaai.org/AAAI/1997/AAAI97-109.pdf
Summary: The first optimal solver for random 3x3x3 cubes. It uses IDA* with corner and edge pattern databases, and the median optimal solution was 18 moves.
Use: Exactly the "pattern-database IDA*" stage of the project.

**12.** Felner, A., Korf, R. E., & Hanan, S. (2004). Additive pattern database heuristics. *Journal of Artificial Intelligence Research, 22*, 279-318. https://doi.org/10.1613/jair.1480 [PR]
Free: https://jair.org/index.php/jair/article/view/10391 (also https://arxiv.org/pdf/1107.0050)
Summary: Shows when values from several pattern databases can be added together, which gives a much stronger heuristic that still never overestimates. This produced large speedups on sliding-tile puzzles.
Use: Ideas for improving the heuristic.

## Part D: Monte Carlo tree search

**13.** Kocsis, L., & Szepesvari, C. (2006). Bandit based Monte-Carlo planning. In *ECML 2006* (LNCS 4212, pp. 282-293). Springer. https://doi.org/10.1007/11871842_29 [PR]
Free: https://cs.nyu.edu/~shasha/papers/uct.pdf
Summary: Introduces UCT, the core of most MCTS programs. It balances trying new moves against repeating moves that already look good.
Use: Primary source for MCTS.

**14.** Browne, C. B., et al. (2012). A survey of Monte Carlo tree search methods. *IEEE Transactions on Computational Intelligence and AI in Games, 4*(1), 1-43. https://doi.org/10.1109/TCIAIG.2012.2186810 [PR]
Free: https://turing.iimas.unam.mx/~luis/cursos/IA2022-1/lecturas/SurveyMCTS.pdf
Summary: A broad review of MCTS. It explains the four steps (selection, expansion, simulation, backpropagation), the main variations, and where MCTS is used.
Use: A readable overview of MCTS.

## Part E: Reinforcement learning and learned heuristics

**15.** Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [BOOK]
Free: http://incompleteideas.net/book/the-book-2nd.html
Summary: The standard RL textbook, free from its authors. It covers agents, rewards, value functions, temporal-difference learning, and combining learning with planning.
Use: Background for the RL stage.

**16.** Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). Mastering the game of Go without human knowledge. *Nature, 550*, 354-359. https://doi.org/10.1038/nature24270 [PR]
Free: https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf (author's accepted version)
Summary: AlphaGo Zero learned Go only by playing itself. A neural network guides MCTS, and MCTS produces better training data for the network.
Use: Inspiration for the DeepCube approach.

**17.** Jabbari Arfaee, S., Zilles, S., & Holte, R. C. (2010). Bootstrap learning of heuristic functions. *Proceedings of the Third Annual Symposium on Combinatorial Search (SoCS 2010)*. [PR]
Free: https://webdocs.cs.ualberta.ca/~holte/Publications/SoCS2010.pdf
Summary: Trains a neural network as a heuristic, step by step. It solves easier puzzles, learns from those solutions, and repeats with harder ones. IDA* with the learned heuristic solved sliding-tile, pancake and blocks-world puzzles about 4000 times faster on the 15-Puzzle, with solutions only about 4.5% longer than optimal.
Use: Bridges hand-made heuristics and machine learning. (This is the free conference version of the authors' 2011 journal article.)

**18.** McAleer, S., Agostinelli, F., Shmakov, A., & Baldi, P. (2018). Solving the Rubik's Cube without human knowledge. arXiv:1805.07470. [PRE]
Free: https://arxiv.org/abs/1805.07470
Summary: Introduces DeepCube. A network learns from positions scrambled backward from the solved cube and then guides MCTS. It solved 100% of random scrambles with no human knowledge.
Use: The first deep-RL cube solver. Not peer-reviewed, so pair it with #19.

**19.** Agostinelli, F., McAleer, S., Shmakov, A., & Baldi, P. (2019). Solving the Rubik's cube with deep reinforcement learning and search. *Nature Machine Intelligence, 1*, 356-363. https://doi.org/10.1038/s42256-019-0070-z [PR]
Free: https://deepcube.igb.uci.edu/static/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf
Summary: DeepCubeA uses weighted A* guided by a learned heuristic. It solved 100% of test cubes and found a shortest solution about 60% of the time. It also solved sliding-tile puzzles, Lights Out and Sokoban.
Use: Connects A* and RL, the final stage of the project.

**20.** Konen, W. (2023). Towards learning Rubik's Cube with n-tuple-based reinforcement learning (Technical report, TH Koln). arXiv:2301.12167. [PRE]
Free: https://arxiv.org/abs/2301.12167
Summary: RL with small lookup-table "n-tuple" networks plus MCTS. It fully solved the 2x2x2 and most 3x3x3 cubes scrambled up to 15 moves, with far less computing power than DeepCube.
Use: One of the few RL studies of the 2x2x2 specifically. Not peer-reviewed.

## Notes for vetting

- Free copies of journal papers: some links (#2, #3, #10, #13, #14) are copies hosted on university course or faculty sites, and #16 is the author's accepted version in UCL's repository. Cite the official journal details above, not the hosting site.
- Not peer-reviewed: #18 and #20 (preprints). #6 and #9 are expert websites, and #5 is software documentation.

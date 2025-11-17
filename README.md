# Pathfinder+ : Enhanced Exploration with Intelligent Breadcrumb-Based A* Optimization

## 🧭 Overview
Pathfinder+ is an intelligent rover simulation environment that demonstrates **hybrid exploration techniques** inspired by classical AI search algorithms.  
The rover first performs **DFS-style uninformed exploration**, leaving behind *breadcrumbs* of visited nodes.  
Later, in **Second Pass Mode**, the rover uses **A\*** search with learned heuristics and breadcrumbs to generate improved, shorter navigation routes.

The system visually simulates:
- Real-time map exploration
- Fog-of-war visibility
- Intelligent path planning
- Breadcrumb tracking
- Camera zoom & rover-centered navigation
- Multiple exploration phases (DFS → A* Optimization)

---

## 🎯 Motivation
Traditional uninformed search (DFS/BFS) is exhaustive but inefficient.  
Pathfinder+ integrates:
- **Exploration Memory**
- **Breadcrumb Learning**
- **Heuristic Optimization**

…to reduce traversal cost, improve revisit paths, and simulate intelligent autonomous rover behavior similar to NASA Mars missions.

---

## 📚 Literature Survey
Pathfinder+ incorporates principles from:
- Russell & Norvig — *Artificial Intelligence: A Modern Approach*
- Classical search algorithms (DFS, BFS, UCS, A*)
- Visibility graphs & grid-based pathfinding
- Multi-phase exploration models used in robotics
- Fog-of-war and sensor-range visibility (similar to SLAM-lite systems)

---

## 🎯 Problem Statement
“**How can a rover explore an unknown environment efficiently, and later optimize its path using the knowledge gained from initial exploration?**”

Challenges addressed:
- Realistic environment scanning
- Incremental knowledge representation
- Handling obstacles and terrain costs
- Combining uninformed search with heuristic optimization
- Generating optimized second-pass traversal routes

---

## ✨ Novelty of the Work
- **Breadcrumb-based heuristic discovery**  
  The rover logs "key turn points" during the first pass, used later to optimize the A* heuristic.

- **Dual-Phase Exploration**
  - Phase 1: DFS-like exploration with local visibility
  - Phase 2: Global optimization with A*

- **Visibility-driven fog system**
  Only scanned areas are revealed — mimicking real sensor constraints.

- **Camera system with rover-centered zoom**  
  Precision zoom based strictly on rover position.

---

## 🎓 CO Mapping (Course Outcomes)
- **CO1:** AI system components demonstrated (agents, sensors, environment, performance measures)  
- **CO2:** Problem solving via search (DFS, BFS, A*, heuristics) implemented  
- **CO3:** Constraints shown via terrain, obstacles, fog-of-war  
- **CO4:** Intelligent reasoning with breadcrumb-based optimization  

---

## 🛠 Methodology (Step-by-step)
1. **Map Generation**
   - Procedural obstacles  
   - Configurable terrain  
   - Randomized seeding  

2. **Rover Initialization**
   - Starts at map center  
   - Limited energy  
   - Scan radius  

3. **First Pass (DFS Exploration)**
   - Moves greedily to unexplored frontier  
   - Records breadcrumbs at meaningful turns  
   - Tracks history for return-to-base  

4. **Return-to-Base Phase**
   - Backtracks strictly along visited nodes  
   - No A* used  

5. **Second Pass (A* Optimization)**
   - Upgrades to heuristic search  
   - Uses breadcrumbs as high-value waypoints  
   - Generates optimized paths  

6. **Visualization**
   - Fog-of-war  
   - Real-time camera  
   - Trace lines per phase  
   - Rover/base sprites  

---

## Project Features

### 🔹 Intelligent First-Pass Exploration (DFS)
The rover explores unknown terrain using DFS-style uninformed search, dropping *breadcrumb nodes* whenever it changes direction.  
These breadcrumbs define an efficient exploration skeleton for later optimization.

### 🔹 Breadcrumb-Driven A* Optimization (Second Pass)
When Second Pass mode is activated:
- Rover switches to a heuristic-driven A* search.
- It targets the optimal sequence of breadcrumb nodes.
- Produces a significantly shorter, smoother, and smarter traversal path.

### 🔹 Fog-of-War Visualization
Unknown terrain is covered by dynamic fog.  
The rover’s sensors reveal circular visibility zones as it moves.

### 🔹 Zoomable Interactive Map
Camera can zoom **into the rover’s center**.  
User-controlled zoom-in/out buttons allow inspection of fine path details.

### 🔹 Dual-Colored Trace System
- Yellow trace → DFS exploration path  
- Cyan trace → A* optimized path  

Both provide visual differentiation between naive exploration vs. intelligent optimization.

### 🔹 Realistic Return-to-Base Backtracking
Upon hitting energy threshold:
- Rover retraces its own breadcrumb-aware route.
- No unrealistic straight-line teleporting.
- Behaves like a real battery-limited robot.

### 🔹 Smart UI Sidebar Controls
- **Start**
- **Return to Base**
- **Second Pass**
- **Recharge**
- **Zoom In / Zoom Out**

All controls visible and interactive in the right sidebar.

---

## 📊 Results
- A* reduces path length versus first-pass DFS  
- Breadcrumbs serve as learned heuristics  
- Return-to-base is reliable and deterministic  
- Rover path becomes significantly more efficient in second pass  

Example metrics (printed in console):
- First-pass distance  
- Second-pass distance  
- Efficiency gained (%)  
- Nodes expanded in DFS vs A*

---

## 🔮 Conclusion & Future Work
Pathfinder+ demonstrates the effectiveness of combining **uninformed exploration** with **heuristic optimization**.  
It provides a visual, intuitive understanding of how agent memory and heuristics dramatically improve performance.

### Possible extensions:
- Full SLAM  
- Dynamic obstacles  
- Neural-network-based heuristics  
- Multi-agent cooperation  
- Energy-aware planning  

---

## Directory Structure

```
mars-rover-pathfinder-plus/
│── assets/
│   ├── mars.jpg
│   ├── drone.png
│   ├── block.png
│   ├── blackSmoke00.png
│   ├── light_350_soft.png
│
│── pro.py
│── visualizer_sprites.py
│── README.md
│── requirements.txt
└── LICENSE
```

---

## Installation & Requirements

```
Python 3.10+
pygame 2.5+
numpy 1.26+
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Simulation

```
python pro.py
```
---

## Authors

Rohan U Nair
SreeHari Sathyajith
Swarna Sri Ashwika


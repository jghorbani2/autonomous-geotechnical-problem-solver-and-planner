# Autonomous Geotechnical Problem Solver and Planner

A sophisticated dependency-graph solver with measurement-suggestion capabilities and symbolic reachability analysis for soil engineering and geotechnical calculations.

## Overview

Solver-Spider implements an advanced dependency resolution system designed specifically for soil mechanics calculations. It uses a multi-stage approach to determine the most efficient path to solve soil engineering problems, minimizing the need for physical measurements while ensuring algebraic accuracy.

## Key Features

### 🔍 **Multi-Stage Resolution**
- **Stage 1**: Uniform-cost search using only algebraic providers (exact providers cost = 0)
- **Stage 2a**: Purely symbolic forward sweep finding parameters derivable without measurements
- **Stage 2b**: Intelligent measurement suggestions excluding symbols from Stage 2a

### 🎯 **Symbolic Reachability**
- Never recommends measuring something that algebra can already supply
- Purely symbolic analysis to identify all parameters that can be derived without physical tests
- Optimizes laboratory and in-situ testing requirements

### 📊 **Smart Measurement Planning**
- Generates experimental plans combining algebraic calculations with necessary measurements
- Prioritizes measurements by cost and duration
- Provides multiple feasible experimental alternatives

### 🧮 **Dependency Graph Analysis**
- Analyzes complex parameter interdependencies in soil calculations
- Handles forward and inverse calculation roles
- Efficient uniform-cost search algorithm for optimal solution paths

## Architecture

The solver consists of several key components:

- **`DependencyGraphSolver`**: Main solver class implementing the multi-stage algorithm
- **`Registry`**: Central repository for equation providers and measurements
- **`EquationProvider`**: Individual calculation providers with cost and dependency information
- **`Measurement`**: Laboratory and field test specifications with pricing and duration data

## Usage

```python
from solver import DependencyGraphSolver, REG

# Initialize solver
solver = DependencyGraphSolver(REG)

# Define known parameters
context = {
    'bulk_density': 1.8,
    'water_content': 0.25,
    # ... other known values
}

# Solve for target parameters
result, plan, cost = solver.solve(context, 'shear_strength', 'bearing_capacity')

# Access measurement suggestions
measurement_options = solver.measurement_choices

# View experimental plans
plans = solver.experimental_plans
```

## Algorithm Details

### Stage 1: Algebraic Path Finding
Uses uniform-cost search to find the cheapest algebraic path to target parameters using only equation providers.

### Stage 2a: Symbolic Reachability
Performs a pure symbolic analysis to identify all parameters that can be derived without any measurements, building upon the initial context.

### Stage 2b: Measurement Optimization
When algebraic paths are insufficient, suggests the most cost-effective combination of measurements and calculations, excluding parameters already determined to be algebraically derivable.

## Example Applications

- **Geotechnical Engineering**: Soil parameter calculations and analysis
- **Foundation Design**: Bearing capacity and settlement predictions
- **Slope Stability**: Factor of safety calculations
- **Earthworks**: Cut/fill volume optimizations
- **Environmental Engineering**: Contaminant transport modeling

## Contributing

We welcome contributions! This project is designed to be extensible and collaborative.

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/jghorbani2/autonomous-geotechnical-problem-solver-and-planner.git`
3. Create a feature branch: `git checkout -b feature-name`
4. Make your changes
5. Run tests: `python -m pytest`
6. Submit a pull request

### Adding New Providers

To add new equation providers or measurements:

1. Define your provider in the registry
2. Specify input/output parameters
3. Set appropriate costs and constraints
4. Test with existing parameter sets

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Autonomous Geotechnical Problem Solver and Planner in your research or engineering work, please cite:

```bibtex
@software{autonomous_geotechnical_solver,
  title = {Autonomous Geotechnical Problem Solver and Planner},
  author = {Javad Ghorbani},
  year = {2024},
  url = {https://github.com/jghorbani2/autonomous-geotechnical-problem-solver-and-planner}
}
```

## Contact

For questions, issues, or contributions, please:

- Open an issue on GitHub
- Submit a pull request
- Contact the maintainers


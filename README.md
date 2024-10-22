# Novel Behavior Discovery in Computation Free Swarms
Contributors: Connor Mattson, Jeremy Clark, Daniel S. Brown

## Required Software
- Python >3.9 & Pip; Python 3.13.0 doesnt support but 3.10.11 works. 
- External Python Packages as defined in [requirements.txt](requirements.txt) 

## Setup
Install Python Packages
    
    pip install -r requirements.txt

Test Simulation

    python -m demo.simulation.cyclic_pursuit

Test Evolution (Novelty Search) - Will take a long time to evolve.

    python -m demo.evolution.novelty_search

You're good to go!

[//]: # (## Use as Package)

[//]: # (You can now install this repository as a pip package under the name novel_swarms==0.0.1)

[//]: # ()
[//]: # (    pip install --upgrade git+ssh://git@github.com/Connor-Mattson/NovelSwarmBehavior.git@master)

## Demos

### Simulation

All 6 emergent behaviors defined in Brown et al. are available for simulation from the command line.

    python -m demo.simulation.cyclic_pursuit
    python -m demo.simulation.aggregation
    python -m demo.simulation.dispersal
    python -m demo.simulation.milling
    python -m demo.simulation.wall_following
    python -m demo.simulation.random

To alter world, agent, and sensor settings, modify the configurations in the [Simulation Playground](demo/simulation/playground.py)

    # Edit /demo/simulation/playground.py first
    python -m demo.simulation.playground

### Evolution

Use the following command to replicate the results shown in Brown et al.

    python -m demo.evolution.novelty_search

If you want to modify the parameters for evolution, use the [Evolution Playground](demo/evolution/playground.py)

    # Edit /demo/evolution/playground.py first
    python -m demo.evolution.playground

Evolving behaviors takes a long time, especially as the number of agents and lifespan increase. 
To save results in the [Output Folder](out/), set `save_archive = True` in the GeneticEvolutionConfig class instatiated in the evolution playground.

    GeneticEvolutionConfig(
        ...
        save_archive=True
    )

The resulting genotype (controller archive) and phenotype (behavior vector archive) files are 
saved to the output folder with the names `geno_g{genome_length}_gen{n_generations}_pop{population_size}_{timestamp}.csv` and `pheno_g{genome_length}_gen{n_generations}_pop{population_size}_{timestamp}.csv`.

### Results
If you have results saved to /out (see above section), modify [/demo/results/results_from_file.py](/demo/results/results_from_file.py) with the path to your files (relative to /out)

    archive = NoveltyArchive(
        pheno_file="PHENOTYPE_FILE",    # Replace with your file
        geno_file="GENOTYPE_FILE"       # Replace with your file
    )

Then run

    python -m demo.results.results_from_file

This will allow you to explore the reduced behavior space that you generated from an earlier evolution execution.
You can also use your pheno and geno files for plotting behaviors/controllers over time, as all entries are saved to the archives in order.

## Configuration
As part of our desire to make a framework that can easily be tweaked and expanded, much of the blackbox details are hidden behinds the scenes (in the /src folder).
Use the common configuration interfaces to modify common parameters that do not require a complex knowledge of the codebase.

    from src.novelty.GeneRule import GeneRule
    from src.novelty.evolve import main as evolve
    from src.results.results import main as report
    from src.config.WorldConfig import RectangularWorldConfig
    from src.config.defaults import ConfigurationDefaults
    from src.config.EvolutionaryConfig import GeneticEvolutionConfig

    # Use the default Differential Drive Agent, initialized with a single sensor and normal physics
    agent_config = ConfigurationDefaults.DIFF_DRIVE_AGENT

    # Create a Genotype Ruleset that matches the size and boundaries of your robot controller _max and _min represent
    # the maximum and minimum acceptable values for that index in the genome. mutation_step specifies the largest
    # possible step in any direction that the genome can experience during mutation.
    genotype = [
        GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=4),
        GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=4),
        GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=4),
        GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=4),
    ]

    # Use the default Behavior Vector (from Brown et al.) to measure the collective swarm behaviors
    phenotype = ConfigurationDefaults.BEHAVIOR_VECTOR

    # Define an empty Rectangular World with size (w, h) and n agents.
    world_config = RectangularWorldConfig(
        size=(500, 500),
        n_agents=30,
        behavior=phenotype,
        agentConfig=agent_config,
        padding=15
    )

    # Define the breath and depth of novelty search with n_generations and n_populations
    # Modify k_nn to change the number of nearest neighbors used in calculating novelty.
    # Increase simulation_lifespan to allow agents to interact with each other for longer.
    # Set save_archive to True to save the resulting archive to /out.
    novelty_config = GeneticEvolutionConfig(
        gene_rules=genotype,
        phenotype_config=phenotype,
        n_generations=100,
        n_population=100,
        crossover_rate=0.7,
        mutation_rate=0.15,
        world_config=world_config,
        k_nn=15,
        simulation_lifespan=300,
        display_novelty=False,
        save_archive=False,
    )

    # Novelty Search through Genetic Evolution
    archive = evolve(config=novelty_config)

    results_config = ConfigurationDefaults.RESULTS
    results_config.world = world_config
    results_config.archive = archive

    # Take Results from Evolution, reduce dimensionality, and present User with Clusters.
    report(config=results_config)

## Augmentation
We have explored the idea of augmenting this framework further to allow more complex world, sensor, controller, and actuator spaces. 
Much of the backbone to support these augmentations is present in this codebase, but lacks testing and robustness.

We invite you to augment cautiously and carefully test output validity.

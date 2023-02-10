"""
Feel free to copy this file and explore configurations that lead to interesting results.

If you do not plan to make commits to the GitHub repository or if you can ensure that changes to this file
are not included in your commits, you may directly edit and run this file.
"""
from novel_swarms.config.defaults import ConfigurationDefaults
from novel_swarms.novelty.GeneRule import GeneRule, GeneBuilder
from novel_swarms.novelty.evolve import main as evolve
from novel_swarms.results.results import main as report
from novel_swarms.behavior.AngularMomentum import AngularMomentumBehavior
from novel_swarms.behavior.AverageSpeed import AverageSpeedBehavior
from novel_swarms.behavior.GroupRotationBehavior import GroupRotationBehavior
from novel_swarms.behavior.RadialVariance import RadialVarianceBehavior
from novel_swarms.behavior.ScatterBehavior import ScatterBehavior
from novel_swarms.sensors.BinaryFOVSensor import BinaryFOVSensor
from novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from novel_swarms.sensors.GenomeDependentSensor import GenomeBinarySensor
from novel_swarms.sensors.SensorSet import SensorSet
from novel_swarms.config.AgentConfig import DiffDriveAgentConfig
from novel_swarms.config.WorldConfig import RectangularWorldConfig
from novel_swarms.config.EvolutionaryConfig import GeneticEvolutionConfig
import numpy as np
from novel_swarms.behavior.SensorOffset import GeneElementDifference

if __name__ == "__main__":

    SEED = None

    sensors = SensorSet([
        BinaryLOSSensor(angle=0),
        GenomeBinarySensor(genome_id=8)
        # BinaryFOVSensor(theta=14 / 2, distance=125, degrees=True)
    ])

    agent_config = DiffDriveAgentConfig(
        sensors=sensors,
        seed=SEED,
    )

    gene_specifications = GeneBuilder(
        heuristic_validation=True,
        round_to_digits=1,
        rules=[
            GeneRule(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
            GeneRule(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
            GeneRule(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
            GeneRule(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
            GeneRule(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
            GeneRule(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
            GeneRule(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
            GeneRule(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
            GeneRule(_max=((2 / 3) * np.pi), _min=-((2 / 3) * np.pi), mutation_step=(np.pi / 8), round_digits=4),
        ]
    )

    phenotype = [
        AverageSpeedBehavior(),
        AngularMomentumBehavior(),
        RadialVarianceBehavior(),
        ScatterBehavior(),
        GroupRotationBehavior(),
        # GeneElementDifference(8, 9)
    ]

    world_config = RectangularWorldConfig(
        size=(500, 500),
        n_agents=24,
        seed=SEED,
        behavior=phenotype,
        agentConfig=agent_config,
        padding=15
    )

    novelty_config = GeneticEvolutionConfig(
        gene_builder=gene_specifications,
        phenotype_config=phenotype,
        n_generations=15,
        n_population=100,
        crossover_rate=0.7,
        mutation_rate=0.15,
        world_config=world_config,
        k_nn=15,
        simulation_lifespan=1200,
        display_novelty=True,
        save_archive=True,
        show_gui=True,
        save_every=1,
    )

    # Novelty Search through Genetic Evolution
    archive = evolve(config=novelty_config)

    results_config = ConfigurationDefaults.RESULTS
    results_config.world = world_config
    results_config.archive = archive

    # Take Results from Evolution, reduce dimensionality, and present User with Clusters.
    report(config=results_config)

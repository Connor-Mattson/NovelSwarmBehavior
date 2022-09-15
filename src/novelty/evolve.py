import math

import pygame
import time
from src.gui.evolutionGUI import EvolutionGUI
from src.novelty.BehaviorDiscovery import BehaviorDiscovery
from src.novelty.GeneRule import GeneRule
from src.config.EvolutionaryConfig import GeneticEvolutionConfig

FRAMERATE = 60
GUI_WIDTH = 200


def main(config: GeneticEvolutionConfig):

    # initialize the pygame module
    pygame.init()
    pygame.display.set_caption("Evolutionary Novelty Search")

    # screen must be global so that other modules can access + draw to the window
    screen = pygame.display.set_mode((config.world_config.w + GUI_WIDTH, config.world_config.h))

    # define a variable to control the main loop
    running = True
    paused = False
    save_results = config.save_archive
    display_plots = config.display_novelty

    # Create the GUI
    gui = EvolutionGUI(x=config.world_config.w, y=0, h=config.world_config.h, w=GUI_WIDTH)
    gui.set_title("Novelty Evolution")

    # Initialize GA
    gene_rules = config.gene_rules
    evolution = BehaviorDiscovery(
        generations=config.generations,
        population_size=config.population,
        crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate,
        world_config=config.world_config,
        lifespan=config.lifespan,
        k_neighbors=config.k,
        genotype_rules=gene_rules,
        behavior_config=config.behavior_config
    )

    gui.set_discovery(evolution)
    last_gen_timestamp = time.time()

    # Generation Loop
    for generation in range(evolution.total_generations):

        if not running:
            break

        evolution.curr_generation = generation

        # Population loop
        for i, genome in enumerate(evolution.population):
            # Looped Event Handling
            for event in pygame.event.get():
                # Cancel the game loop if user quits the GUI
                if event.type == pygame.QUIT:
                    running = False

            if not running:
                break

            screen.fill((0, 0, 0))

            evolution.curr_genome = i
            evolution.runSingleGeneration(screen, i=i, seed=i)
            gui.draw(screen=screen)

            pygame.display.flip()

            # Limit the FPS of the simulation to FRAMERATE
            pygame.time.Clock().tick(FRAMERATE)

        screen.fill((0, 0, 0))
        evolution.evaluate(screen=screen)
        gui.draw(screen=screen)
        pygame.display.flip()

        screen.fill((0, 0, 0))
        evolution.evolve()
        gui.draw(screen=screen)
        pygame.display.flip()

        current_time = time.time()
        gui.set_elapsed_time(current_time - last_gen_timestamp)
        last_gen_timestamp = current_time

    if save_results:
        evolution.archive.saveArchive(
            f"pheno_g{len(gene_rules)}_gen{evolution.total_generations}_pop{len(evolution.population)}")
        evolution.archive.saveGenotypes(
            f"geno_g{len(gene_rules)}_gen{evolution.total_generations}_pop{len(evolution.population)}")

    if display_plots:
        evolution.results()

    return evolution.archive
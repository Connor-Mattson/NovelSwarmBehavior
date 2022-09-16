from typing import Tuple
import pygame
import random
import math
from copy import deepcopy
from .Agent import Agent
from ..config.AgentConfig import DiffDriveAgentConfig

class DifferentialDriveAgent(Agent):

    SEED = -1

    def __init__(self, config: DiffDriveAgentConfig = None) -> None:

        self.controller = config.controller

        if config.seed is not None:
            self.seed(config.seed)

        if config.x is None:
            self.x_pos = random.randint(0 + config.agent_radius, config.world.w - config.agent_radius)
        else:
            self.x_pos = config.x

        if config.y is None:
            self.y_pos = random.randint(0 + config.agent_radius, config.world.h - config.agent_radius)
        else:
            self.y_pos = config.y

        super().__init__(self.x_pos, self.y_pos, name="None")

        if config.angle is None:
            self.angle = random.random() * math.pi * 2
        else:
            self.angle = config.angle

        self.radius = config.agent_radius
        self.wheel_radius = config.wheel_radius
        self.dt = config.dt
        self.is_highlighted = False
        self.agent_in_sight = None

        self.sensors = deepcopy(config.sensors)
        self.attach_agent_to_sensors()

    def seed(self, seed):
        if DifferentialDriveAgent.SEED < 0:
            DifferentialDriveAgent.SEED = seed
        else:
            DifferentialDriveAgent.SEED += 1
        random.seed(DifferentialDriveAgent.SEED)

    def step(self, check_for_world_boundaries=None, population=None, check_for_agent_collisions=None) -> None:

        if population is None:
            raise Exception("Expected a Valid value for 'population' in step method call")

        super().step()
        vr, vl = self.interpretSensors()
        self.dx = (self.wheel_radius / 2) * (vl + vr) * math.cos(self.angle)
        self.dy = (self.wheel_radius / 2) * (vl + vr) * math.sin(self.angle)
        heading = (vr - vl) / (self.radius * 2)

        old_x_pos = self.x_pos
        old_y_pos = self.y_pos

        self.x_pos += self.dx * self.dt
        self.y_pos += self.dy * self.dt
        self.angle += heading * self.dt

        if check_for_world_boundaries is not None:
            check_for_world_boundaries(self)

        if check_for_agent_collisions is not None:
            check_for_agent_collisions(self)

        # Calculate the 'real' dx, dy after collisions have been calculated.
        # This is what we use for velocity in our equations
        self.dx = self.x_pos - old_x_pos
        self.dy = self.y_pos - old_y_pos

        for sensor in self.sensors:
            sensor.step(population=population)

    def draw(self, screen) -> None:
        super().draw(screen)
        for sensor in self.sensors:
            sensor.draw(screen)

        # Draw Cell Membrane
        filled = 0 if self.is_highlighted else 1
        pygame.draw.circle(screen, (255, 255, 255), (self.x_pos, self.y_pos), self.radius, width=filled)

        # "Front" direction vector
        head = self.getFrontalPoint()
        tail = self.getPosition()
        vec = [head[0] - tail[0], head[1] - tail[1]]
        mag = self.radius * 2
        vec_with_magnitude = ((vec[0] * mag) + tail[0], (vec[1] * mag) + tail[1])
        pygame.draw.line(screen, (255, 255, 255), tail, vec_with_magnitude)

    def interpretSensors(self) -> Tuple:
        sensor_state = self.sensors.getState()
        vr = self.controller[sensor_state * 2]
        vl = self.controller[(sensor_state * 2) + 1]
        return vr, vl

    def __str__(self) -> str:
        return "(x: {}, y: {}, r: {}, θ: {})".format(self.x_pos, self.y_pos, self.radius, self.angle)
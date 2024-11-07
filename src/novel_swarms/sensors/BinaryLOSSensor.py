import pygame
import numpy as np
import math
from .AbstractSensor import AbstractSensor
from typing import List
from datetime import datetime
import os
import csv


class BinaryLOSSensor(AbstractSensor):
    def __init__(self, parent=None, angle=None, draw=True, history_length=50, width=1):
        super().__init__(parent=parent, draw=draw)
        self.current_state = 0
        self.angle = angle
        self.history = []
        self.hist_len = history_length
        self.width = width
        self.show = draw
        # Set the CSV file path relative to the repo's root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        sensor_data_dir = os.path.join(repo_root, 'SensorData')
        os.makedirs(sensor_data_dir, exist_ok=True)  # Create the directory if it doesn't exist
        self.csv_file = os.path.join(sensor_data_dir, 'BinaryLOSsensorData.csv')
        self._initialize_csv()

    def _initialize_csv(self):
        """Initializes the CSV file with headers if it doesn't already exist."""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "sensor_value","BotName"])

    def log_to_csv(self, sensor_value):
        """Logs the timestamp and sensor value to a CSV file."""
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().isoformat(), sensor_value,self.parent.name])

    def checkForLOSCollisions(self, world) -> None:
        sensor_position = self.parent.getPosition()

        # Equations taken from Dunn's 3D Math Primer for Graphics, section A.12
        p_0 = np.array(sensor_position)
        d = np.array(self.getLOSVector())
        d_hat = d / np.linalg.norm(d)

        # Check seen agent from last frame first, to avoid expensive computation
        if self.parent.agent_in_sight is not None and not self.parent.agent_in_sight.deleted:
            agent = self.parent.agent_in_sight
            if self.agent_in_sight(agent, p_0, d_hat):
                self.parent.agent_in_sight = agent
                self.current_state = 1
                self.add_to_history(self.current_state)
                self.log_to_csv(self.current_state)
                return

        for agent in world.population:
            if self.agent_in_sight(agent, p_0, d_hat):
                self.parent.agent_in_sight = agent
                self.current_state = 1
                self.add_to_history(self.current_state)
                self.log_to_csv(self.current_state)
                return

        self.parent.agent_in_sight = None
        self.current_state = 0
        self.add_to_history(self.current_state)
        self.log_to_csv(self.current_state)

    def agent_in_sight(self, agent, p_0, d_hat):
        if agent == self.parent:
            return False

        c = np.array(agent.getPosition())
        e = c - p_0
        a = np.dot(e, d_hat)
        if a < 0:
            return False

        r_2 = agent.radius * agent.radius
        e_2 = np.dot(e, e)
        a_2 = a * a

        has_intersection = (r_2 - e_2 + a_2) >= 0
        if has_intersection:
            return True

        return False

    def step(self, world):
        super(BinaryLOSSensor, self).step(world=world)
        self.checkForLOSCollisions(world=world)

    def draw(self, screen):
        if not self.show:
            return
        super(BinaryLOSSensor, self).draw(screen)

        # Draw Sensory Vector (Vision Vector)
        sight_color = (255, 0, 0)
        if self.current_state == 1:
            sight_color = (0, 255, 0)

        magnitude = self.parent.radius * (20 if self.parent.is_highlighted else 1)
        magnitude *= 5 if self.angle is None else 3

        head = (self.parent.x_pos, self.parent.y_pos)

        if self.angle is None:
            tail = (self.parent.x_pos + (magnitude * math.cos(self.parent.angle)),
                    self.parent.y_pos + (magnitude * math.sin(self.parent.angle)))
        else:
            tail = (self.parent.x_pos + (magnitude * math.cos(self.angle + self.parent.angle)),
                    self.parent.y_pos + (magnitude * math.sin(self.angle + self.parent.angle)))

        pygame.draw.line(screen, sight_color, head, tail, width=self.width)

    def getLOSVector(self) -> List:
        head = self.parent.getPosition()
        tail = self.getFrontalPoint()
        return [tail[0] - head[0], tail[1] - head[1]]

    def getFrontalPoint(self):
        if self.angle is None:
            return self.parent.getFrontalPoint()
        return self.parent.x_pos + math.cos(self.angle + self.parent.angle), self.parent.y_pos + math.sin(self.angle + self.parent.angle)

    def add_to_history(self, value):
        if len(self.history) > self.hist_len:
            self.history = self.history[1:]
        self.history.append(value)

    def as_config_dict(self):
        return {
            "type": "BinaryLOSSensor",
            "angle": self.angle,
            "history_length": self.hist_len,
        }

    @staticmethod
    def from_dict(d):
        return BinaryLOSSensor(
            parent=None,
            angle=d["angle"],
            history_length=d["history_length"],
        )
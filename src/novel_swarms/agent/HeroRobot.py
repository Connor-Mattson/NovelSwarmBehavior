from typing import Tuple
import pygame
import random
import math
import time
# import pymunk
import numpy as np
from copy import deepcopy
from .Agent import Agent
from ..config.AgentConfig import HeroRobotConfig,HeroPlusRobotConfig
from ..sensors.GenomeDependentSensor import GenomeBinarySensor, GenomeFOVSensor
from ..util.collider.CircularCollider import CircularCollider,CircularFrictionCollider
from ..util.timer import Timer
from ..util.collider.AABB import AABB
from ..util.collider.AngleSensitiveCC import AngleSensitiveCC

class HeroRobot(Agent):
    SEED = -1
    DEBUG = False

    def __init__(self, config: HeroRobotConfig = None, name=None) -> None:

        self.controller = config.controller
        

        if config.seed is not None:
            self.seed(config.seed)

        if config.x is None:
            self.x_pos = random.randint(round(0 + config.agent_radius), round(config.world.w - config.agent_radius))
        else:
            self.x_pos = config.x

        if config.y is None:
            self.y_pos = random.randint(round(0 + config.agent_radius), round(config.world.h - config.agent_radius))
        else:
            self.y_pos = config.y

        super().__init__(self.x_pos, self.y_pos, name=name)

        if config.angle is None:
            self.angle = random.random() * math.pi * 2
        else:
            self.angle = config.angle

        self.radius = config.agent_radius
        self.distance_bw_wheels = config.distance_bw_wheels
        self.wheel_radius = config.wheel_radius
        self.dt = config.dt
        self.is_highlighted = False
        self.agent_in_sight = None
        '''self.idiosyncrasies = config.idiosyncrasies
        I1_MEAN, I1_SD = 0.93, 0.08
        I2_MEAN, I2_SD = 0.95, 0.06
        self.i_1 = np.random.normal(I1_MEAN, I1_SD) if self.idiosyncrasies else 1.0
        self.i_2 = np.random.normal(I2_MEAN, I2_SD) if self.idiosyncrasies else 1.0
        self.stop_on_collision = config.stop_on_collision
'''
        # mass=1
        # inertia=pymunk.moment_for_circle(mass,0,self.shield_radius)
        # self.munk_body=pymunk.Body(mass,inertia)
        # self.munk_body.position=self.x_pos,self.y_pos
        # self.munk_shape=pymunk.Circle(self.munk_body, self.shield_radius)
        
        self.sensors = deepcopy(config.sensors)
        for sensor in self.sensors:
            if isinstance(sensor, GenomeBinarySensor) or isinstance(sensor, GenomeFOVSensor):
                sensor.augment_from_genome(config.controller)

        self.aabb = None
        self.collider = None
        self.body_filled = config.body_filled
        self.body_color = config.body_color
        self.c_now = (0, 0)

        self.attach_agent_to_sensors()

        # Set Trace Settings if a trace was assigned to this object.
        self.trace_color = config.trace_color
        self.trace = config.trace_length is not None
        if self.trace:
            self.trace_path = []
            self.trace_length = config.trace_length

    def seed(self, seed):
        random.seed(HeroRobot.SEED)

    def step(self, check_for_world_boundaries=None, world=None, check_for_agent_collisions=None) -> None:

        if world is None:
            raise Exception("Expected a Valid value for 'World' in step method call - Unicycle Agent")

        # timer = Timer("Calculations")
        super().step()
        self.aabb = None
        self.collider = None

        if world.goals and world.goals[0].agent_achieved_goal(self):
            v, omega = 0, 0
        else:
            v, omega = self.interpretSensors()

        self.c_now = v, omega

        # Define Idiosyncrasies that may occur in actuation/sensing
        # idiosync_1 = self.i_1
        # idiosync_2 = self.i_2

        self.dx = v * math.cos(self.angle) # * idiosync_1
        self.dy = v * math.sin(self.angle) # * idiosync_1
        dw = omega # * idiosync_2

        old_x_pos = self.x_pos
        old_y_pos = self.y_pos
        old_heading = self.angle

        if self.stopped_duration > 0:
            self.stopped_duration -= 1

        else:
            self.x_pos += self.dx * self.dt
            self.y_pos += self.dy * self.dt
            self.angle += dw * self.dt

        if check_for_world_boundaries is not None:
            check_for_world_boundaries(self)

        # if check_for_agent_collisions is not None:
        #     check_for_agent_collisions(self, forward_freeze=True)

        self.handle_collisions(world)

        if self.stopped_duration > 0:
            self.x_pos = old_x_pos
            self.y_pos = old_y_pos
            self.angle = old_heading

        # Calculate the 'real' dx, dy after collisions have been calculated.
        # This is what we use for velocity in our equations
        self.dx = self.x_pos - old_x_pos
        self.dy = self.y_pos - old_y_pos
        # timer = timer.check_watch()

        # timer = Timer("Sensors")
        for sensor in self.sensors:
            sensor.step(world=world)
        # timer = timer.check_watch()

        self.add_to_trace(self.x_pos, self.y_pos)

    def handle_collisions(self, world):
        collisions = True
        limit = 10
        while collisions and limit > 0:
            limit -= 1
            collisions = False
            self.build_collider()
            agent_set = world.getAgentsMatchingYRange(self.get_aabb())
            for agent in agent_set:
                if agent.name == self.name:
                    continue
                if self.aabb.intersects(agent.get_aabb()):
                    self.get_aabb().toggle_intersection()
                    correction = self.collider.collision_then_correction(agent.build_collider())
                    if correction is not None:
                        collisions = True
                        if np.linalg.norm(correction) == 0:
                            self.stopped_duration = 15
                        else:
                            self.x_pos += correction[0]
                            self.y_pos += correction[1]
                        break
            if collisions:
                self.collider.flag_collision()

    def build_collider(self):
        if self.stop_on_collision:
            self.collider = AngleSensitiveCC(self.x_pos, self.y_pos, self.radius, self.angle, self.get_action(), sensitivity=45)
        else:
            self.collider = CircularCollider(self.x_pos, self.y_pos, self.radius)
        return self.collider

    def draw(self, screen) -> None:
        super().draw(screen)
        for sensor in self.sensors:
            sensor.draw(screen)

        # Draw Cell Membrane
        filled = 0 if self.is_highlighted or self.body_filled else 1
        color = self.body_color if not self.stopped_duration else (255, 255, 51)
        pygame.draw.circle(screen, color, (self.x_pos, self.y_pos), self.radius, width=filled)
        # pygame.draw.circle(screen, color, (self.x_pos, self.y_pos), self.shield_radius, width=1)

        # Draw Trace (if parameterized to do so)
        self.draw_trace(screen)

        # "Front" direction vector
        head = self.getFrontalPoint()
        tail = self.getPosition()
        vec = [head[0] - tail[0], head[1] - tail[1]]
        mag = self.radius * 1.7
        vec_with_magnitude = ((vec[0] * mag) + tail[0], (vec[1] * mag) + tail[1])
        pygame.draw.line(screen, (255, 255, 255), tail, vec_with_magnitude, width=3)


        if self.DEBUG:
            self.debug_draw(screen)

    def interpretSensors(self) -> Tuple:
        sensor_state = self.sensors.getState()
        v = self.controller[sensor_state * 2]
        omega = self.controller[(sensor_state * 2) + 1]
        return v, omega

    def debug_draw(self, screen):
        # self.aabb.draw(screen)
        self.collider.draw(screen)
        pygame.draw.circle(screen, (255, 0, 255), self.get_icc(), 3, width=0)

    def get_action(self):
        return np.array([self.dx * self.dt, self.dy, self.dt])

    def get_aabb(self):
        """
        Return the Bounding Box of the agent
        """
        if not self.aabb:
            top_left = (self.x_pos - self.radius, self.y_pos - self.radius)
            bottom_right = (self.x_pos + self.radius, self.y_pos + self.radius)
            self.aabb = AABB(top_left, bottom_right)
        return self.aabb

    def draw_trace(self, screen):
        if not self.trace:
            return
        for x, y in self.trace_path:
            pygame.draw.circle(screen, self.trace_color, (x, y), 2)

    def add_to_trace(self, x, y):
        if not self.trace:
            return
        self.trace_path.append((x, y))
        if len(self.trace_path) > self.trace_length:
            self.trace_path.pop(0)

    def get_icc(self):
        v, w = self.c_now
        r = v / w
        return self.x_pos - (r * np.sin(self.angle)), self.y_pos + (r * np.cos(self.angle))

    def __str__(self) -> str:
        return "(x: {}, y: {}, r: {}, θ: {})".format(self.x_pos, self.y_pos, self.radius, self.angle)
class HeroPlusRobot(Agent):
    SEED = -1
    DEBUG = False

    def __init__(self, config: HeroPlusRobotConfig = None, name=None) -> None:

        self.controller = config.controller
        

        if config.seed is not None:
            self.seed(config.seed)

        if config.x is None:
            self.x_pos = random.randint(round(0 + config.agent_radius), round(config.world.w - config.agent_radius))
        else:
            self.x_pos = config.x

        if config.y is None:
            self.y_pos = random.randint(round(0 + config.agent_radius), round(config.world.h - config.agent_radius))
        else:
            self.y_pos = config.y

        super().__init__(self.x_pos, self.y_pos, name=name)

        if config.angle is None:
            self.angle = random.random() * math.pi * 2
        else:
            self.angle = config.angle

        self.radius = config.agent_radius
        self.robot_friction= config.robot_friction

        self.distance_bw_wheels = config.distance_bw_wheels
        self.wheel_radius = config.wheel_radius
        self.shield_radius = config.shield_radius
        self.dt = config.dt
        self.is_highlighted = False
        self.agent_in_sight = None
        '''self.idiosyncrasies = config.idiosyncrasies
        I1_MEAN, I1_SD = 0.93, 0.08
        I2_MEAN, I2_SD = 0.95, 0.06
        self.i_1 = np.random.normal(I1_MEAN, I1_SD) if self.idiosyncrasies else 1.0
        self.i_2 = np.random.normal(I2_MEAN, I2_SD) if self.idiosyncrasies else 1.0
        self.stop_on_collision = config.stop_on_collision
'''
        # mass=1
        # inertia=pymunk.moment_for_circle(mass,0,self.shield_radius)
        # self.munk_body=pymunk.Body(mass,inertia)
        # self.munk_body.position=self.x_pos,self.y_pos
        # self.munk_shape=pymunk.Circle(self.munk_body, self.shield_radius)
        self.dw_f=0 # try using this to control angle - adds to dw
        self.da =0
        self.o_da=0

        self.sensors = deepcopy(config.sensors)
        for sensor in self.sensors:
            if isinstance(sensor, GenomeBinarySensor) or isinstance(sensor, GenomeFOVSensor):
                sensor.augment_from_genome(config.controller)

        self.aabb = None
        self.old_collide_agent=False
        self.collider = None
        self.old_collide = None

        self.body_filled = config.body_filled
        self.body_color = config.body_color
        self.c_now = (0, 0)

        self.attach_agent_to_sensors()

        # Set Trace Settings if a trace was assigned to this object.
        self.trace_color = config.trace_color
        self.trace = config.trace_length is not None
        if self.trace:
            self.trace_path = []
            self.trace_length = config.trace_length

    def seed(self, seed):
        random.seed(HeroPlusRobot.SEED)

    def step(self, check_for_world_boundaries=None, world=None, check_for_agent_collisions=None) -> None:

        if world is None:
            raise Exception("Expected a Valid value for 'World' in step method call - Unicycle Agent")

        # timer = Timer("Calculations")
        super().step()
        self.aabb = None
        self.collider = None

        if world.goals and world.goals[0].agent_achieved_goal(self):
            v, omega = 0, 0
        else:
            v, omega = self.interpretSensors()

        self.c_now = v, omega

        # Define Idiosyncrasies that may occur in actuation/sensing
        # idiosync_1 = self.i_1
        # idiosync_2 = self.i_2

        self.dx = v * math.cos(self.angle) # * idiosync_1
        self.dy = v * math.sin(self.angle) # * idiosync_1
        dw = omega # * idiosync_2

        old_x_pos = self.x_pos
        old_y_pos = self.y_pos
        old_heading = self.angle

        if self.stopped_duration > 0:
            self.stopped_duration -= 1

        else:
            self.x_pos += self.dx * self.dt
            self.y_pos += self.dy * self.dt
            self.angle += dw * self.dt
        
        if self.robot_friction:
            self.dx = self.x_pos - old_x_pos
            self.dy = self.y_pos - old_y_pos
            self.da = self.angle - old_heading

        if check_for_world_boundaries is not None:
            if self.robot_friction:
                self.old_collide=check_for_world_boundaries(self)
            else: 
                check_for_world_boundaries(self)

        # if check_for_agent_collisions is not None:
        #     check_for_agent_collisions(self, forward_freeze=True)

        if self.robot_friction:
            self.handle_friction_collisions(world)
            
        else:
            self.handle_collisions(world)

        if self.stopped_duration > 0:
            self.x_pos = old_x_pos
            self.y_pos = old_y_pos
            self.angle = old_heading

        # Calculate the 'real' dx, dy after collisions have been calculated.
        # This is what we use for velocity in our equations
        self.dx = self.x_pos - old_x_pos
        self.dy = self.y_pos - old_y_pos
        self.o_da = self.da
        if self.robot_friction:
            if not ((abs(self.angle - old_heading) <= abs(self.c_now[1]+0.04))):
                self.angle=old_heading
        # timer = timer.check_watch()

        # timer = Timer("Sensors")
        for sensor in self.sensors:
            sensor.step(world=world)
        # timer = timer.check_watch()

        self.add_to_trace(self.x_pos, self.y_pos)

    def handle_collisions(self, world):
        collisions = True
        limit = 10
        while collisions and limit > 0:
            limit -= 1
            collisions = False
            self.build_collider()
            agent_set = world.getAgentsMatchingYRange(self.get_aabb())
            for agent in agent_set:
                if agent.name == self.name:
                    continue
                if self.aabb.intersects(agent.get_aabb()):
                    self.get_aabb().toggle_intersection()
                    correction = self.collider.collision_then_correction(agent.build_collider())
                    if correction is not None:
                        collisions = True
                        if np.linalg.norm(correction) == 0:
                            self.stopped_duration = 15
                        else:
                            self.x_pos += correction[0]
                            self.y_pos += correction[1]
                        break
            if collisions:
                self.collider.flag_collision()

    def handle_friction_collisions(self, world):
        collisions = True
        limit = 10
        self.collision_flag = False
        while collisions and limit > 0:
            limit -= 1
            collisions = False
            self.build_collider()
            agent_set = world.getAgentsMatchingYRange(self.get_aabb())
            for agent in agent_set:
                if agent.name == self.name:
                    continue
                if self.aabb.intersects(agent.get_aabb()) or self.old_collide_agent:
                    dist_between_radii = np.linalg.norm(np.array([agent.x_pos, agent.y_pos, 0]) - np.array([self.x_pos, self.y_pos, 0]))
                    dist_difference = ((self.shield_radius+1) + (agent.shield_radius)) - dist_between_radii
                    #print(np.linalg.norm(np.array([self.dx,self.dy,0])))
                    if dist_difference > 0 and np.linalg.norm(np.array([self.dx,self.dy,0]))==0:
                        #print("here",self.da,np.linalg.norm(np.array([self.dx,self.dy,0])))
                        self.angle+=self.da*0.01 - self.da
                    self.get_aabb().toggle_intersection()
                    correction,angle_correction,flag  = self.collider.collision_then_correction(agent.build_collider())
                    self.old_collide_agent=flag
                    if correction is not None:
                        collisions = True
                        if np.linalg.norm(correction) == 0:
                            self.stopped_duration = 15
                        else:
                            self.x_pos -= correction[0]
                            self.y_pos -= correction[1]
                            self.angle -= angle_correction
                        break
            if collisions:
                self.collider.flag_collision()
                self.collision_flag = True

    def build_collider(self):
        if self.stop_on_collision:
            self.collider = AngleSensitiveCC(self.x_pos, self.y_pos, self.shield_radius, self.angle, self.get_action(), sensitivity=45)
        elif self.robot_friction:
            self.collider = CircularFrictionCollider(self.x_pos, self.y_pos, self.shield_radius, self.angle, self.c_now[0],self.dx,self.dy,self.da,self.robot_friction,self.old_collide_agent)
        else:
            self.collider = CircularCollider(self.x_pos, self.y_pos, self.shield_radius)

        return self.collider

    def draw(self, screen) -> None:
        super().draw(screen)
        for sensor in self.sensors:
            sensor.draw(screen)

        # Draw Cell Membrane
        filled = 0 if self.is_highlighted or self.body_filled else 1
        color = self.body_color if not self.stopped_duration else (255, 255, 51)
        pygame.draw.circle(screen, color, (self.x_pos, self.y_pos), self.radius, width=filled)
        pygame.draw.circle(screen, color, (self.x_pos, self.y_pos), self.shield_radius, width=1)

        # Draw Trace (if parameterized to do so)
        self.draw_trace(screen)

        # "Front" direction vector
        head = self.getFrontalPoint()
        tail = self.getPosition()
        vec = [head[0] - tail[0], head[1] - tail[1]]
        mag = self.radius * 1.7
        vec_with_magnitude = ((vec[0] * mag) + tail[0], (vec[1] * mag) + tail[1])
        pygame.draw.line(screen, (255, 255, 255), tail, vec_with_magnitude, width=3)


        if self.DEBUG:
            self.debug_draw(screen)

    def interpretSensors(self) -> Tuple:
        sensor_state = self.sensors.getState()
        v = self.controller[sensor_state * 2]
        omega = self.controller[(sensor_state * 2) + 1]
        return v, omega

    def debug_draw(self, screen):
        # self.aabb.draw(screen)
        self.collider.draw(screen)
        pygame.draw.circle(screen, (255, 0, 255), self.get_icc(), 3, width=0)

    def get_action(self):
        return np.array([self.dx * self.dt, self.dy, self.dt])

    def get_aabb(self):
        """
        Return the Bounding Box of the agent
        """
        if not self.aabb:
            top_left = (self.x_pos - self.shield_radius, self.y_pos - self.shield_radius)
            bottom_right = (self.x_pos + self.shield_radius, self.y_pos + self.shield_radius)
            self.aabb = AABB(top_left, bottom_right)
        return self.aabb

    def draw_trace(self, screen):
        if not self.trace:
            return
        for x, y in self.trace_path:
            pygame.draw.circle(screen, self.trace_color, (x, y), 2)

    def add_to_trace(self, x, y):
        if not self.trace:
            return
        self.trace_path.append((x, y))
        if len(self.trace_path) > self.trace_length:
            self.trace_path.pop(0)

    def get_icc(self):
        v, w = self.c_now
        r = v / w
        return self.x_pos - (r * np.sin(self.angle)), self.y_pos + (r * np.cos(self.angle))

    def __str__(self) -> str:
        return "(x: {}, y: {}, r: {}, θ: {})".format(self.x_pos, self.y_pos, self.radius, self.angle)

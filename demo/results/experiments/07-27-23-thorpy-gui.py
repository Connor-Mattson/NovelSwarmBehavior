from src.novel_swarms.world.simulate import main as sim
from src.novel_swarms.config.AgentConfig import LevyAgentConfig, MazeAgentConfig
from src.novel_swarms.sensors.SensorSet import SensorSet
from src.novel_swarms.sensors.BinaryFOVSensor import BinaryFOVSensor
from src.novel_swarms.agent.LevyAgent import LevyAgent
from src.novel_swarms.agent.UnicycleAgent import UnicycleAgent
from src.novel_swarms.config.WorldConfig import RectangularWorldConfig
from src.novel_swarms.world import RectangularWorld
from src.novel_swarms.world.initialization.RandomInit import RectRandomInitialization
from src.novel_swarms.gui.fullControlGUI import FullControlGUI

if __name__ == "__main__":

    # Initialize Conical Sensors
    sensors = SensorSet([
        BinaryFOVSensor(
            theta=18,
            distance=120
        )
    ])

    # Initialize Intelligent Agent
    agent_i = MazeAgentConfig(
        agent_radius=5,
        controller=[1, 0.5, 1, -0.5],
        sensors=sensors
    )
    
    # Initialize World
    N_AGENTS = 15
    GUI_WIDTH = 500
    WIDTH, HEIGHT = 500, 500
    env = RectangularWorldConfig(
        size=(WIDTH, HEIGHT),
        agentConfig=agent_i,
        n_agents=N_AGENTS,
        seed=10,
        init_type=RectRandomInitialization(
            num_agents=N_AGENTS,
            bb=((200, 200), (300, 300))
        )
    )

    gui = FullControlGUI(
        x = WIDTH, 
        y = 0,
        w = GUI_WIDTH,
        h = HEIGHT,
    )

    sim(world_config=env, gui=gui)



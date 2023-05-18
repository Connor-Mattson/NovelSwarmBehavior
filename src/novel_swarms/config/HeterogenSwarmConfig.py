from ..agent.AgentFactory import AgentFactory


class HeterogeneousSwarmConfig:
    def __init__(self):
        self.subpopulation_information = {}
        self.world = None

    def add_sub_populuation(self, config, count):
        self.subpopulation_information[config] = count

    def build_agent_population(self):
        population = []
        for key in self.subpopulation_information:
            for n in range(self.subpopulation_information[key]):
                agent = AgentFactory.create(key)
                population.append(agent)
        return population

    def attach_world_config(self, world_config):
        self.world = world_config
        for key in self.subpopulation_information:
            key.attach_world_config(world_config)

    def as_dict(self):
        return {
            "type": "HeterogeneousSwarmConfig",
            "sub_population_configs": [
                k.as_dict() for k in self.subpopulation_information.keys()
            ],
            "counts": [
                v for v in self.subpopulation_information.values()
            ]
        }

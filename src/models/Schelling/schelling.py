import agentpy as ap
import matplotlib.pyplot as plt
import numpy as np
import torch


class Person(ap.Agent):

    def setup(self):
        """Initiate agent attributes."""
        self.grid = self.model.grid
        self.random = self.model.random
        self.group = self.random.choice(range(self.p.n_groups))
        self.share_similar = 0
        self.happy = False

    def update_happiness(self):
        """Be happy if rate of similar neighbors is high enough."""
        neighbors = self.grid.neighbors(self)
        similar = len([n for n in neighbors if n.group == self.group])
        ln = len(neighbors)
        self.share_similar = similar / ln if ln > 0 else 0
        self.happy = self.share_similar >= self.p.want_similar

    def find_new_home(self):
        """Move to random free spot and update free spots."""
        new_spot = self.random.choice(self.model.grid.empty)
        self.grid.move_to(self, new_spot)


class SegregationModel(ap.Model):

    def setup(self):

        # Parameters
        s = self.p.size
        n = self.n = int(self.p.density * (s**2))

        # Create grid and agents
        self.grid = ap.Grid(self, (s, s), track_empty=True)
        self.agents = ap.AgentList(self, n, Person)
        self.grid.add_agents(self.agents, random=True, empty=True)

    def update(self):
        # Update list of unhappy people
        self.agents.update_happiness()
        self.unhappy = self.agents.select(self.agents.happy == False)

        # Stop simulation if all are happy
        if len(self.unhappy) == 0:
            self.stop()

    def step(self):
        # Move unhappy people to new location
        self.unhappy.find_new_home()

    def get_segregation(self):
        # Calculate average percentage of similar neighbors
        return round(sum(self.agents.share_similar) / self.n, 2)

    def get_happyness(self):
        # Calculate average percentage of similar neighbors
        return round(sum(self.agents.happy) / self.n, 2)

    def end(self):
        # Measure segregation at the end of the simulation
        self.report("segregation", self.get_segregation())
        self.report("happyness", self.get_happyness())


def run_schelling(parameters: dict) -> np.ndarray[np.uint8]:

    # Clip the want_similar to the valid range [0, 1]
    parameters["want_similar"] = max(0, min(1, parameters["want_similar"]))
    model = SegregationModel(parameters)
    run_summary = model.run(steps=parameters["max_run_duration"], display=0)

    final_state = model.grid.attr_grid("group")
    np.nan_to_num(final_state, copy=False, nan=-1)

    # Plot the final state
    # colors = {1: "red", 0: "blue", -1: "white"}
    # cmap = plt.matplotlib.colors.ListedColormap([colors[i] for i in sorted(colors)])
    # plt.imshow(final_state, cmap=cmap, vmin=-1, vmax=1)
    # plt.show()

    rgb_image = np.zeros((final_state.shape[0], final_state.shape[1], 3), dtype=np.uint8)
    rgb_image[final_state == 1] = [255, 0, 0]  # Red for 1
    rgb_image[final_state == 0] = [0, 0, 255]  # Blue for 0
    rgb_image[final_state == -1] = [255, 255, 255]  # White for -1

    # Convert to a PyTorch tensor and ensure uint8 dtype
    rgb_image = torch.from_numpy(rgb_image).to(torch.uint8)

    return rgb_image


if __name__ == "__main__":
    parameters = {
        "want_similar": 0.7,  # For agents to be happy
        "n_groups": 3,  # Number of groups
        "density": 0.9,  # Density of population
        "size": 100,  # Height and length of the grid
        "max_run_duration": 100,  # Simulation will stop before if hapyness converges
    }

    rgb_image = run_schelling(parameters)
    print(rgb_image.shape)

    # plot the rgb image
    plt.imshow(rgb_image)
    plt.axis("off")  # turn off axes for better visualization
    # save image as pdf without any edges, only the image
    plt.savefig(f"schelling_{parameters['want_similar']}.png", bbox_inches="tight", pad_inches=0)

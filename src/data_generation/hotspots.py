import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SyntheticHotspots:
    def __init__(self, rows, cols, time_steps, random_state=None):
        self.rows = rows
        self.cols = cols
        self.time_steps = time_steps

        if isinstance(random_state, (int, type(None))):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

        self.data = np.zeros((time_steps, rows, cols))

    def generate(self):
        """
        Placeholder for generation method. Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the generate() method.")

    def visualize(self, timestep=0, cmap='hot_r',label="Crime count"):
        """
        Quickly visualize data at a specific timestep.
        """
        fig, ax = plt.subplots(figsize=(6,5))
        cax = ax.imshow(self.data[timestep], cmap=cmap, origin='lower')
        ax.set_title(f"Hotspots Visualization at timestep {timestep}")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        fig.colorbar(cax, ax=ax, label=label)
        plt.show()

    def to_dataframe(self):
        """
        Transform data to a pandas DataFrame for easier analysis.
        """
        records = []
        for t in range(self.time_steps):
            for r in range(self.rows):
                for c in range(self.cols):
                    records.append({
                        "timestep": t,
                        "row": r,
                        "col": c,
                        "count": self.data[t, r, c]
                    })

        df = pd.DataFrame.from_records(records)
        return df

class PoissonHotspots(SyntheticHotspots):
    """
    Synthetic hotspots generated with a Poisson distribution.
    Allows multiple variable-size hotspots, varying intensity, and global noise.
    """
    def __init__(self, rows, cols, time_steps, base_lam=1, n_hotspots=5,
                 hotspot_strength_range=(5, 15), hotspot_size_range=(1, 3),
                 noise_level=0.2, random_state=None):
        super().__init__(rows, cols, time_steps, random_state)
        self.base_lam = base_lam
        self.n_hotspots = n_hotspots
        self.hotspot_strength_range = hotspot_strength_range
        self.hotspot_size_range = hotspot_size_range
        self.noise_level = noise_level

    def generate(self):
        
        self.hotspot_locations = []
        for _ in range(self.n_hotspots):
            size = self.random_state.randint(
                self.hotspot_size_range[0], self.hotspot_size_range[1] + 1
            )
            r = self.random_state.randint(0, self.rows - size + 1)
            c = self.random_state.randint(0, self.cols - size + 1)
            self.hotspot_locations.append((r, c, size))

        for t in range(self.time_steps):
            
            frame = self.random_state.poisson(self.base_lam, size=(self.rows, self.cols))

            
            for (r, c, size) in self.hotspot_locations:
                lam = self.random_state.randint(*self.hotspot_strength_range)
                frame[r:r+size, c:c+size] += self.random_state.poisson(lam=lam, size=(size, size))

            
            if self.noise_level > 0:
                frame += self.random_state.poisson(self.noise_level, size=(self.rows, self.cols))

            self.data[t] = frame

        return self.data.astype(int)



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SyntheticHotspots:
    """
    Base class for synthetic hotspot data generation.
    """
    def __init__(self, rows, cols, time_steps, random_state=None):
        self.rows = rows
        self.cols = cols
        self.time_steps = time_steps
        self.random_state = random_state
        np.random.seed(self.random_state)
        
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
    """
    def __init__(self, rows, cols, time_steps, lam=2, hotspots_num=2, hotspot_intensity=(10, 15), random_state=None):
        super().__init__(rows, cols, time_steps, random_state)
        self.lam = lam
        self.hotspots_num = hotspots_num
        self.hotspot_intensity = hotspot_intensity
        self.hotspot_coords = []

    def generate(self):
        self.data = np.random.poisson(self.lam, size=(self.time_steps, self.rows, self.cols))

        # Fixed hotspot locations
        self.hotspot_coords = [
            (np.random.randint(self.rows), np.random.randint(self.cols)) 
            for _ in range(self.hotspots_num)
        ]

        for (r, c) in self.hotspot_coords:
            r_min, r_max = max(0, r-1), min(self.rows, r+2)
            c_min, c_max = max(0, c-1), min(self.cols, c+2)
            increment = np.random.randint(self.hotspot_intensity[0], self.hotspot_intensity[1])
            self.data[:, r_min:r_max, c_min:c_max] += increment

        return self.data

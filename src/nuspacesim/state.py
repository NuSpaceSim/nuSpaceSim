from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Sequence

from astropy.io import fits
from astropy.table import Table as AstropyTable

from .config import NssConfig


@dataclass
class SimulationState:
    config: NssConfig = NssConfig()
    event_table: AstropyTable = AstropyTable()
    results: Dict[str, float] = {}
    plots: Sequence[Any] = []
    sim_time: str = f"{datetime.now():%Y%m%d%H%M%S}"
    output_file = f"nuspacesim_run_{sim_time}.fits"


class StagedWriter:
    """Optionally write intermediate values to file"""

    def __init__(self, sim: SimulationState):
        self.sim: SimulationState = sim

    def __call__(
        self,
        col_names: Sequence[str],
        columns,  # : Sequence[ArrayLike],
        *args,
        **kwargs,
    ):
        self.sim.event_table.add_columns(columns, names=col_names, *args, **kwargs)
        fits.write(self.sim.event_table, self.sim.output_file, overwrite=True)

    # def add_meta(self, name: str, value: Any, comment: str):
    # sim.meta[name] = (value, comment)
    # if write_stages:
    #     sim.write(output_file, format="fits", overwrite=True)

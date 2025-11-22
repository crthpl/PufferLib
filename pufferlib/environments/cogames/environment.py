"""CoGames wrapper for PufferLib."""

import functools
from cogames.cli.mission import get_mission
from mettagrid import PufferMettaGridEnv
from mettagrid.envs.stats_tracker import StatsTracker
from mettagrid.simulator import Simulator
from mettagrid.util.stats_writer import NoopStatsWriter


def env_creator(name="cogames.cogs-v-clips"):
    return functools.partial(make, name=name)


def make(name="cogames.cogs-v-clips.machina_1.open_world", variants=None, cogs=None, render_mode="auto", seed=None, buf=None):
    # Strip package prefixes
    parts = name.split(".")
    while parts and parts[0].replace("-", "_") in {"cogames", "cogs_v_clips"}:
        parts.pop(0)
    mission_name = ".".join(parts) if parts else "training_facility.harvest"

    _, env_cfg, _ = get_mission(mission_name, variants_arg=variants, cogs=cogs)

    render = "none" if render_mode == "auto" else "unicode" if render_mode in {"human", "ansi"} else render_mode
    simulator = Simulator()
    simulator.add_event_handler(StatsTracker(NoopStatsWriter()))
    env = PufferMettaGridEnv(simulator=simulator, cfg=env_cfg, buf=buf, seed=seed or 0)
    env.render_mode = render
    if seed:
        env.reset(seed)
    return env

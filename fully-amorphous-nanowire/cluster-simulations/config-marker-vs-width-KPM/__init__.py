import pathlib
import tomli

path = pathlib.Path(__file__).parent / "variables_marker_vs_width_KPM.toml"
with path.open(mode="rb") as fp:
    variables_cond_vs_N = tomli.load(fp)
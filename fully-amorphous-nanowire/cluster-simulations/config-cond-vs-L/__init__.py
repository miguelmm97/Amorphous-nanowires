import pathlib
import tomli

path = pathlib.Path(__file__).parent / "variables_cond_vs_L.toml"
with path.open(mode="rb") as fp:
    variables_cond_vs_L = tomli.load(fp)


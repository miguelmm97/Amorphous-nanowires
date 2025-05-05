import pathlib
import tomli

path = pathlib.Path(__file__).parent / "variables_cond_vs_N.toml"
with path.open(mode="rb") as fp:
    variables_cond_vs_N = tomli.load(fp)
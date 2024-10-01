import pathlib
import tomli

path = pathlib.Path(__file__).parent / "variables_cond_vs_L.toml"
with path.open(mode="rb") as fp:
    variables_cond_vs_L = tomli.load(fp)


path = pathlib.Path(__file__).parent / "variables_cond_vs_flux.toml"
with path.open(mode="rb") as fp:
    variables_cond_vs_flux = tomli.load(fp)


path = pathlib.Path(__file__).parent / "variables_cond_vs_Ef.toml"
with path.open(mode="rb") as fp:
    variables_cond_vs_Ef = tomli.load(fp)
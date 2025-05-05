import config
import numpy as np
variables = config.variables_cond_vs_L

Nz_max      = variables['Nz_max']
Nz_min      = variables['Nz_min']
Nz_L        = variables['Nz_L']
Nz = np.linspace(Nz_min, Nz_max, Nz_L)


with open('params-cond-vs-L.txt', 'w') as f:
    for w in variables['width']:
        for L in Nz:
            for i in range(variables['Nsamples']):
                f.write(f'{w} {L} {i}\n')

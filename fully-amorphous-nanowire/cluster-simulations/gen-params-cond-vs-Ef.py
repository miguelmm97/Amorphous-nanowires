import config
variables = config.variables_cond_vs_Ef

with open('params-cond-vs-Ef.txt', 'w') as f:
    for w in variables['width']:
        for i in range(variables['Nsamples']):
            f.write(f'{w} {i}\n')

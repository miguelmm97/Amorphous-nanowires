import config
variables = config.variables_cond_vs_L

with open('params-cond-vs-L.txt', 'w') as f:
    for w in variables['width']:
        for i in range(variables['Nsamples']):
            f.write(f'{w} {i}\n')

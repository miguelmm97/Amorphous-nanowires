import config
variables = config.variables_cond_vs_N

with open('params-cond-vs-N.txt', 'w') as f:
    for n in variables['N']:
        for i in range(variables['Nsamples']):
            f.write(f'{n} {i}\n')

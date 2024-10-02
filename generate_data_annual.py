import pandas as pd
import starsim as ss
import sciris as sc
import numpy as np
import zombie as z

people = ss.People(n_agents=5_000) # People, as before

# Configure and create an instance of the Zombie class
zombie_pars = dict(
    init_prev = 0.001,
    beta = {'random': ss.beta(0.06), 'maternal': ss.beta(1)},
    p_fast = ss.bernoulli(p=0.3),
    p_death_on_zombie_infection = ss.bernoulli(p=0.15),
    p_symptomatic = ss.bernoulli(p=1.0),
)
zombie = z.Zombie(zombie_pars)

# This function allows the lambda parameter of the poisson distribution used to determine
# n_contacts to vary based on agent characteristics, a key feature of Starsim.
def choose_degree(self, sim, uids):
    mean_degree = np.full(fill_value=4, shape=len(uids)) # Default value is 4
    zombie = sim.diseases['zombie']
    is_fast = zombie.infected[uids] & zombie.fast[uids]
    mean_degree[is_fast] = 50 # Fast zombies get 50
    return mean_degree

# We create two network layers, random and maternal
networks = [
    ss.RandomNet(n_contacts=ss.poisson(lam=choose_degree)),
    ss.MaternalNet()
]

# Configure and create demographic modules
death_pars = dict(
    death_rate = 15, # per 1,000
    p_zombie_on_natural_death = ss.bernoulli(p=0.2),
)
deaths = z.DeathZombies(**death_pars)
births = ss.Pregnancy(fertility_rate=175) # per 1,000 women 15-49 annually
demog = [births, deaths]

# Create an intervention that kills symptomatic zombies
interventions = z.KillZombies(year=2024, rate=0.1)

# And finally bring everything together in a sim
sim_pars = dict(start=2024, stop=2040, dt=.5, verbose=0)
sim = ss.Sim(sim_pars, people=people, diseases=zombie, networks=networks, demographics=demog, interventions=interventions)

# Run the sim and plot results
sim.run()
sim.plot('zombie')

sim_result_list = ['n_infected']
df_res = sim.results['zombie'].to_df()
df_res['time'] = np.floor(np.round(df_res.timevec, 1)).astype(int)

model_output = pd.DataFrame()
for skey in sim_result_list:
    model_output[skey] = df_res.groupby(by='time')[skey].sum()

with open('zombie_outbreak_data_annual.csv', 'w') as f:
    f.write("time,zombie.n_infected\n")

    for i in range(len(model_output)):
        #f.write(f"{sim.yearvec[i]},{int(sim.results.zombie.n_infected[i])}\n")
        f.write(f"{model_output.index[i]},{int(model_output.values[i])}\n")
print("Complete")
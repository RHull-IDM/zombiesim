import sciris as sc
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import starsim as ss
from zombie import *
import seaborn as sns

# Implement custom calibration class
class ZombieCalibration(ss.Calibration):
    @staticmethod
    def translate_pars(sim=None, calib_pars=None):
        sim.pars['verbose'] = 0

        spec = calib_pars.pop('zombie_beta', None)
        if spec is not None:
            sim.diseases['zombie'].pars['beta']['random'].set(spec['value'])

        sim = ss.Calibration.translate_pars(sim, calib_pars)
        return sim

    def plot_calib_results(self, results_path=None, show=False):
        ''' Plot the observed and simulated hospitalization data for the best calibration'''

        # Summarize observed data
        df_actual = self.data.copy()

        df_pred = pd.DataFrame({'predicted.n_infected': self.after_sim.results.zombie.n_infected})
        df_pred['year'] = np.floor(self.after_sim.timevec).astype(int)
        df_init = pd.DataFrame({'year': self.before_sim.timevec, 'predicted.n_infected': self.before_sim.results.zombie.n_infected})
        df_init['year'] = np.floor(self.before_sim.timevec).astype(int)

        df_pred = df_pred.groupby(by='year').sum()
        df_init = df_init.groupby(by='year').sum()



        # Plot the observed & simulated timeseries
        plt.figure(figsize=(10, 6))
        plt.plot(df_actual.index, df_actual['zombie.n_infected'], marker='o', linestyle='-', label='Observed')
        plt.plot(df_pred.index, df_pred['predicted.n_infected'], marker='o', linestyle='-', label='Predicted', color='orange')
        plt.plot(df_init.index, df_init['predicted.n_infected'], marker='o', linestyle='-', label='Initial guess', color='green')
        plt.xlabel('Time')
        plt.ylabel('Total infections by day')
        #plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()  # This line adds the legend to the plot
        diff = np.round(np.sum(abs(df_actual['zombie.n_infected'] - df_pred['predicted.n_infected'])), 1)
        plt.text(0.99, 0.85, f'Obs - pred: {diff}', horizontalalignment='right', verticalalignment='center',
                 transform=plt.gca().transAxes)
        plt.savefig(f"{results_path}/zombie_timeseries.png")
        if show:
            plt.show()

# This function allows the lambda parameter of the poisson distribution used to determine
# n_contacts to vary based on zombie type



def basic_zombie():
    people = ss.People(n_agents=5_000)  # People, as before

    # Configure and create an instance of the Zombie class
    zombie_pars = dict(
        init_prev=0.03,
        beta = ss.beta(0.05),
        p_fast=ss.bernoulli(p=0.1),
        p_death_on_zombie_infection=ss.bernoulli(p=0.25),
        p_symptomatic=ss.bernoulli(p=1.0),
    )
    zombie = Zombie(zombie_pars)

    def choose_degree(self, sim, uids):
        mean_degree = np.full(fill_value=4, shape=len(uids))  # Default value is 4
        zombie = sim.diseases['zombie']
        is_fast = zombie.infected[uids] & zombie.fast[uids]
        mean_degree[is_fast] = 50  # Fast zombies get 50
        return mean_degree

    # We create two network layers, random and maternal
    networks = [
        ss.RandomNet(n_contacts=ss.poisson(lam=choose_degree)),
        ss.MaternalNet()
    ]

    # Configure and create demographic modules
    death_pars = dict(
        death_rate=15,  # per 1,000
        p_zombie_on_natural_death=ss.bernoulli(p=0.2),
    )
    deaths = DeathZombies(**death_pars)
    births = ss.Pregnancy(fertility_rate=175)  # per 1,000 women 15-49 annually
    demog = [births, deaths]

    # Create an intervention that kills symptomatic zombies
    interventions = KillZombies(year=2024, rate=0.1)

    # And finally bring everything together in a sim
    sim_pars = dict(start=2024, stop=2040, dt=0.5, verbose=0)
    sim = ss.Sim(sim_pars, people=people, diseases=zombie, networks=networks, demographics=demog,
                 interventions=interventions)

    # Run the sim and plot results
    sim.run()
    sim.plot('zombie');


scens = {
    'Default assumptions': {},
    'More fast zombies': {'zombie_pars': dict(p_fast=ss.bernoulli(p=0.75))},
    'Fast-->Slow zombies': {'zombie_pars': dict(p_fast=ss.bernoulli(p=0.75), dur_fast=ss.weibull(c=2, scale=2))},
    'Finite infectious period': {'zombie_pars': dict(dur_inf=ss.normal(loc=5, scale=2))},
    'All zombies asymptomatic': {'zombie_pars': dict(p_symptomatic=ss.bernoulli(p=0))},
    'Less death on zombie infection': {'zombie_pars': dict(p_death_on_zombie_infection=ss.bernoulli(p=0.10))},
    'More zombies on natural death': {'death_pars': dict(p_zombie_on_natural_death=ss.bernoulli(p=0.5))},
    'REALLY BAD': {'zombie_pars': dict(p_fast=ss.bernoulli(p=1.0), p_symptomatic=ss.bernoulli(p=0), p_death_on_zombie_infection=ss.bernoulli(p=0.50)),
                   'death_pars': dict(p_zombie_on_natural_death=ss.bernoulli(p=1.0))},
}


def run_zombies(scen, rand_seed, zombie_pars=None, death_pars=None, intvs=[], **kwargs):
    people = ss.People(n_agents=5_000) # People

    # Zombies
    zombie_defaults = dict(
        init_prev = 0.03,
        beta = ss.beta(0.05), #{'random': 0.05, 'maternal': 0.5},
        p_fast = ss.bernoulli(p=0.1),
        p_death_on_zombie_infection = ss.bernoulli(p=0.25),
        p_symptomatic = ss.bernoulli(p=1.0),
    )
    zombie_pars = sc.mergedicts(zombie_defaults, zombie_pars) # Override defaults with user-specified parameters
    zombie = Zombie(zombie_pars)

    def choose_degree(self, sim, uids):
        mean_degree = np.full(fill_value=4, shape=len(uids))  # Default value is 4
        zombie = sim.diseases['zombie']
        is_fast = zombie.infected[uids] & zombie.fast[uids]
        mean_degree[is_fast] = 50  # Fast zombies get 50
        return mean_degree


    # Networks
    networks = [
        ss.RandomNet(n_contacts=ss.poisson(lam=choose_degree)),
        ss.MaternalNet()
    ]

    # Deaths
    death_defaults = dict(
        death_rate = 15, # per 1,000 per year
        p_zombie_on_natural_death = ss.bernoulli(p=0.2),
    )
    death_pars = sc.mergedicts(death_defaults, death_pars)
    deaths = DeathZombies(**death_pars)

    # Births
    births = ss.Pregnancy(fertility_rate=175) # per 1,000 women 15-49 per year
    demog = [births, deaths]

    # Interventions
    interventions = KillZombies(year=2024, rate=0.1)
    interventions = [interventions] + sc.promotetolist(intvs) # Add interventions passed in

    # Create and run the simulation
    sim_pars = dict(start=2024, stop=2040, dt=0.5, rand_seed=rand_seed, label=scen, verbose=0)
    sim = ss.Sim(sim_pars, people=people, diseases=zombie, networks=networks, demographics=demog, interventions=interventions)
    sim.run()

    # Package results
    df = pd.DataFrame( {
        'Year': sim.timevec,
        'Population': sim.results.n_alive,
        'Humans': sim.results.n_alive - sim.results.zombie.n_infected,
        'Zombies': sim.results.zombie.n_infected,
        'Zombie Prevalence': sim.results.zombie.prevalence,
        'Congential Zombies (cum)': sim.results.zombie.cum_congenital,
        'Zombie-Cause Mortality': sim.results.zombie.cum_deaths,
    })
    df['rand_seed'] = rand_seed
    df['Scen'] = scen
    for key, val in kwargs.items():
        df[key] = val

    return df
def run_calibs():
    people = ss.People(n_agents=5_000)  # 5000 people live in this town

    # Configure and create an instance of the Zombie class
    zombie_pars = dict(
        init_prev = 0.001,  # 5 / 5_000, 5 people were initially infected
        beta = {'random': ss.beta(0.06), 'maternal': ss.beta(1)},   # Guess. To be calibrated
        p_fast=ss.bernoulli(p=0.1),  # Guess. To be calibrated
        p_death_on_zombie_infection=ss.bernoulli(p=0.15),
        p_symptomatic=ss.bernoulli(p=1.0),
    )
    zombie = Zombie(zombie_pars)

    def choose_degree(self, sim, uids):
        mean_degree = np.full(fill_value=4, shape=len(uids))  # Default value is 4
        zombie = sim.diseases['zombie']
        is_fast = zombie.infected[uids] & zombie.fast[uids]
        mean_degree[is_fast] = 50  # Fast zombies get 50
        return mean_degree

    # We create two network layers, random and maternal
    networks = [
        ss.RandomNet(n_contacts=ss.poisson(lam=choose_degree)),
        ss.MaternalNet()
    ]

    # Configure and create demographic modules
    death_pars = dict(
        death_rate=15,  # per 1,000 (during normal times!)
        p_zombie_on_natural_death=ss.bernoulli(p=0.2),  # Estimate based on observed data
    )
    deaths = DeathZombies(**death_pars)
    births = ss.Pregnancy(fertility_rate=175)  # per 1,000 women 15-49 annually (during normal times!)
    demog = [births, deaths]

    kill_int = KillZombies(year=2024, rate=0.1)  # no zombies are killed at the start
    #vx_int = ss.campaign_vx(product=zombie_vaccine(), years=2024, prob=0.0)  # no vx happening now

    intvs = [kill_int,
             #vx_int
             ]

    # And finally bring everything together in a sim
    sim_pars = dict(start=2024, stop=2040, dt=0.5, verbose=0)
    sim = ss.Sim(sim_pars, people=people, diseases=zombie, networks=networks, demographics=demog, interventions=intvs)

    # Define the calibration parameters
    # calib_pars = dict(
    #     diseases = dict(
    #         zombie = dict(
    #             beta = [0.3, 0.01, 0.5],
    #             p_fast = [0.1, 0.1, 0.5],
    #         ),
    #     ),
    # )
    calib_pars = dict(
        zombie_beta = dict(guess=0.05, low=0.01, high=0.15, path=('diseases', 'zombie', 'beta')),
        zombie_p_fast = dict(guess=0.31, low=0.2, high=0.4, path=('diseases', 'zombie', 'p_fast')),
    )


    # Load the calibration data
    data = pd.read_csv('./zombie_outbreak_data_annual.csv')


    # Create the calibration object
    calib = ZombieCalibration(calib_pars=calib_pars,
                           sim=sim,
                           data=data,
                           total_trials=10,
                           name="test4",
                           keep_db=False,
                           die=True,
                           verbose=True,
                           #debug=True
                              )

    calib.calibrate(confirm_fit=False, load=False, verbose=True)

    # Confirm
    sc.printcyan('\nConfirming fit...')
    calib.confirm_fit()
    print(f'Fit with original pars: {calib.before_fit:n}')
    print(f'Fit with best-fit pars: {calib.after_fit:n}')

    calib.plot_calib_results(results_path='./', show=True)

    # calibrated_sim = sc.dcp(calib.after_sim)

    # Package results
    # df = pd.DataFrame( {
    #     'Day': calibrated_sim.tivec,
    #     'Population': calibrated_sim.results.n_alive,
    #     'Humans': (calibrated_sim.results.n_alive - calibrated_sim.results.zombie.n_infected),
    #     'Zombies': calibrated_sim.results.zombie.n_infected,
    #     'Zombie Prevalence': calibrated_sim.results.zombie.prevalence,
    #     'Congential Zombies (cum)': calibrated_sim.results.zombie.cum_congenital,
    #     'Zombie-Cause Mortality': calibrated_sim.results.zombie.cum_deaths,
    # })



def run_scens():
    # Now run all the scenarios in parallel, repeating each configuration 10 times
    n_repeats = 3

    results = []
    cfgs = []

    for skey, scen in scens.items():
        for rand_seed in range(n_repeats):
            cfgs.append({'scen': skey, 'rand_seed': rand_seed} | scen)

    print(f'Running {len(cfgs)} zombie simulations...')
    T = sc.tic()
    results += sc.parallelize(run_zombies, iterkwargs=cfgs)
    print(f'Completed in {sc.toc(T, output=True):.1f}s')
    df = pd.concat(results).replace(np.inf, np.nan)

    # Display the first few reows of the results data frame
    # display(df.head())

    # Manipulate the data and create a plot using the Seaborn library
    dfm = df.melt(id_vars=['Scen', 'Year', 'rand_seed'],
                  value_vars=['Humans', 'Zombies', 'Zombie Prevalence', 'Zombie-Cause Mortality'], var_name='Channel',
                  value_name='Value')
    g = sns.relplot(kind='line', data=dfm, col='Channel', x='Year', y='Value', hue='Scen', hue_order=scens.keys(),
                    facet_kws=dict(sharey=False), col_wrap=2, height=4)
    g.set(ylim=(0, None))
    g.axes[2].yaxis.set_major_formatter(mtick.PercentFormatter(1));

if __name__ == '__main__':
    #basic_zombie()
    #run_scens()
    run_calibs()
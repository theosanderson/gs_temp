#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import os
import pickle

import numpyro as npy


import genomicsurveillance as gs
from genomicsurveillance.utils.knots import NowCastKnots

# In[2]:


def time_to_str(date):
    return str(date)[:10]

analysis_date = pd.to_datetime('today')
start_date = pd.to_datetime('2020-09-01')
end_date = analysis_date - pd.Timedelta("5 days")


# In[3]:


eng = gs.get_england()


# In[4]:


aliases = gs.get_aliases()


# In[5]:


# add this alias as it does not get merged properly
aliases['A.2.5.1'] = 'A.2'


# In[6]:


vocs = [
    'A.23.1',
    'A.23.1.484',
    'B.1.1.7.484',
    'B.1.351',
    'B.1.525',
    'B.1.1.318',
    'P.1',
    'P.2',
    'B.1.617',
    'B.1.617.1',
    'B.1.617.2',
    'B.1.617.3'
]


# In[7]:


specimen = gs.get_specimen()


# In[8]:


#@markdown Perform a inner join to LTLA codes.
cases = eng.merge(specimen.T, left_on='lad19cd', right_index=True, how='left').loc[:,time_to_str(start_date):time_to_str(end_date)].values


# In[9]:


cog_uk = "https://covid-surveillance-data.cog.sanger.ac.uk/download/lineages_by_ltla_and_week.tsv"


# In[10]:


genomes = (pd.read_csv(cog_uk, sep='\t')
  .assign(Lineage=lambda df: df['Lineage'].apply(lambda x: aliases[x] if ((x in aliases.keys()) and (x not in vocs)) else x))
  )


# In[11]:


selected_lineages = [
        "A", "B", "B.1.177", "B.1.1.7", "B.1.351", "P.1", "B.1.617.1",
        "B.1.617.2", "B.1.525"
    ] 
def filter_and_merge_lineages(lineages, selected_lineages):
    """Filters out 'None' and merges lineages into parents."""
    # TODO: use PANGO aliases to merge non string matches

    lineages = lineages[lineages.Lineage != "None"].copy()
    lineages = lineages[lineages.Lineage != "Lineage data suppressed"].copy()
    lineages['new_lineage'] = 'Other'
    for lin in selected_lineages:
        lineages.loc[lineages['Lineage'].str.startswith(f"{lin}."),
                     'new_lineage'] = lin
        lineages.loc[lineages['Lineage'] == lin, 'new_lineage'] = lin
    lineages = lineages.drop(columns="Lineage").rename(
        columns={'new_lineage': 'Lineage'})
    return (lineages)


# In[12]:


genomes = filter_and_merge_lineages(genomes,selected_lineages)


# In[13]:


#@markdown * Extract dates and sort lineages by name
dates = genomes.WeekEndDate.unique().tolist()
ordered_lineages, other_lineages = gs.sort_lineages(genomes.Lineage.unique().tolist())
all_lineages = ordered_lineages + other_lineages

#@markdown * Code to create the lineage tensor `(num_ltla, date, num_lineages)`
all_tensor = np.stack([(genomes[genomes.WeekEndDate==d]
 .pivot_table(index='LTLA', columns='Lineage', values='Count')
 .merge(eng, left_index=True, right_on='lad19cd', how='right')
 .reindex(columns = all_lineages)
 .fillna(0)
 .values
) for d in dates], 1) 


# In[14]:


merged_names, merged_tensor, merged_cluster =  gs.preprocess_lineage_tensor(all_lineages, all_tensor, vocs=selected_lineages,)


# In[15]:


#@markdown * Before we perform the analysis we set the baseline lineage to B.1.177. This requires to move B.1.177 index to the end in the `merged_tensor` of the final input tensor `lin_tensor`.
baseline_lineage = 'B.1.177'
lin_tensor = np.concatenate([merged_tensor[..., [ i for i in range(merged_tensor.shape[-1]) if merged_names[i] != baseline_lineage]], 
                             merged_tensor[..., [merged_names.index(baseline_lineage)]]], -1)
lin_names = [ name for name in merged_names if name != baseline_lineage] + [baseline_lineage]
lin_dates = np.array([gs.create_date_list(cases.shape[-1], time_to_str(start_date)).index(d) for d in dates])
date_list = gs.create_date_list(cases.shape[-1], time_to_str(start_date))


# In[ ]:





# In[16]:


knots = NowCastKnots(cases.shape[-1], cutoff=3)
m1 = gs.MultiLineage(cases, lin_tensor, lin_dates, eng.pop18.values, tau=5.1, 
                     basis=knots.basis,
                     beta_scale=np.concatenate([np.repeat(1., knots.num_long_basis), np.repeat(.2, knots.num_short_basis)]),
                     alpha1=(cases[..., lin_dates].reshape(317, -1, 1)/2), auto_correlation=None,
                     model_kwargs={'handler':'SVI', 'num_epochs':30000, 'lr':0.001, 'num_samples': 500})
# In[17]:


m1.fit()


# In[18]:


m1


# In[ ]:





# In[19]:


plt.semilogy(m1.loss)
plt.xlabel('Epoch')
plt.ylabel('ELBO')


# In[20]:


from genomicsurveillance.plots import dot_plot
from genomicsurveillance.plots.england import plot_lad, plot_median_and_ci


# In[21]:


plt.figure(figsize=(16, 3))
dot_plot(m1.get_transmissibility(rebase=lin_names.index( 'B.1.177')), baseline= 'B.1.177', xticklabels=lin_names)


# In[22]:


b117_idx = lin_names.index('B.1.1.7')
b16172_idx = lin_names.index('B.1.617.2')
plot_lad(m1, 316, cases, lin_tensor, lin_dates, lin=[b117_idx, b16172_idx, -1], colors=["C3", "C5", "C2"], labels=["B.1.1.7", 'B.1.617.2', "B.1.177"])


# In[ ]:





# In[ ]:





# In[23]:


lambdas = m1.get_lambda_lineage()
log_Rs = m1.get_log_R_lineage()
probabilities = m1.get_probabilities()


# In[24]:


country_lambda = m1.aggregate_lambda_lineage(eng.ctry19id.values)
country_probabilities = m1.aggregate_probabilities(eng.ctry19id.values)


# In[25]:


country_logRs = m1.aggregate_log_R_lineage(eng.ctry19id.values)


# In[26]:


log_Rs_both=  np.concatenate([log_Rs,country_logRs],axis=1)


# In[27]:


lambdas_both = np.concatenate([lambdas,country_lambda],axis=1)
log_Rs_both=  np.concatenate([log_Rs,country_logRs],axis=1)
probabilities_both = np.concatenate([probabilities,country_probabilities],axis=1)


# In[ ]:





# In[28]:


from numpyro.diagnostics import hpdi
def process_parameter(tensor, parameter_name, lin_names, date_list,ltla_codes):
    med = np.median(tensor,axis=0)
    hp = hpdi(tensor)
    if parameter_name=="p":
        med = med / np.sum(med,axis=-1,keepdims=True)
        hp = hp / np.sum(med,axis=-1,keepdims=True)
    data = []
    for ltla_idx, ltla in enumerate(tqdm.tqdm(ltla_codes)):
        for lin_idx,lineage in enumerate(tqdm.tqdm(lin_names)):
                for date_idx,date in enumerate(date_list):
                    data.append({'parameter': parameter_name, 'lad19cd':ltla_codes[ltla_idx], 'date':date,'lineage':lin_names[lin_idx], 'mean':med[ltla_idx,date_idx,lin_idx], 'upper':hp[1,ltla_idx,date_idx,lin_idx], 'lower':hp[0,ltla_idx,date_idx,lin_idx]})
    return(data)


# In[29]:



Rs_both = np.exp(log_Rs_both)
import tqdm
results = [process_parameter(x,i,lin_names,date_list,eng.lad19cd.to_list() + ["overview",]) for i,x in {"lambda":lambdas_both,"R":Rs_both,"p":probabilities_both}.items()]


# In[32]:


data = pd.DataFrame(sum(results, []))


# In[33]:


low_incidence = data.query('parameter=="p" and mean<0.001')


# In[34]:


low_incidence


# In[35]:


data = data[np.logical_or(data['parameter']!="R",np.logical_not((data['lad19cd']+data['date']+data['lineage']).isin(low_incidence['lad19cd']+low_incidence['date']+data['lineage'])))]


# In[36]:




# In[37]:


data.to_csv("full_data_table.csv")







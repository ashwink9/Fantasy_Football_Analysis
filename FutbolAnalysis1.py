#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sbn
import altair as alt
import scipy.stats as stats
import matplotlib.pyplot as plt


# In[2]:


file = "/Users/anair/Downloads/cleaned_players.csv"
df = pd.read_csv(file)


# In[3]:


df.columns


# In[4]:


df.sample(20)


# In[5]:


df["Full_Name"] = df.apply(lambda row: row['first_name'] + ' ' + row['second_name'], axis = 1)
df["Full_Name"]


# In[6]:


df_topgoalscorers = df.loc[df["goals_scored"] > 15, "Full_Name"]
print(f" All top goalscorers :\n {df_topgoalscorers}")


# In[7]:


df.head(10)


# In[8]:


df_topbonus= df.nlargest(10, 'bonus')
df_topbonus


# In[9]:


df_highest_selected = df[df["selected_by_percent"] > 20]
df_highest_selected


# In[10]:


[
    method_name
    for method_name in dir(df.plot)
    if not method_name.startswith("_")
]


# In[11]:


df_2 = df[['total_points', 'minutes', 'Full_Name', 'threat', 'element_type']]


# In[12]:


df.columns


# In[13]:


df_2['pscore'] = df_2['minutes']/df_2['total_points']


# ## fig, axs = plt.subplots(figsize=(12, 4))        Create an empty Matplotlib Figure and Axes
# ## air_quality.plot.area(ax=axs)                   Use pandas to put the area plot on the prepared Figure/Axes
# ## axs.set_ylabel("NO$_2$ concentration")          Do any Matplotlib customization you like
# ## fig.savefig("no2_concentrations.png")           Save the Figure/Axes using the existing Matplotlib method.
# ## plt.show()                                      Display the plot

# In[14]:


df_2


# In[15]:


df_p = df_2[df_2['pscore'].notna()]


# In[16]:


df_p


# In[17]:


sbn.scatterplot(data = df_p, x = 'minutes', y ='pscore', hue = 'element_type', size ='threat')
plt.title("Scatter Plot of minutes played and pscore for players")
plt.show()


# In[18]:


plt.figure(figsize=(14,8))
df_p['element_type'] = df_p['element_type'].astype('category').cat.codes

plt.scatter(df_p['minutes'], df_p['total_points'], s = df_p['pscore'], c = df_p['element_type'], cmap = 'viridis', alpha =0.7)
plt.colorbar(label = 'Element Type')
plt.ylabel("Total points earned")
plt.xlabel("Minutes Played")
plt.title("Position Based Threat & Score")
plt.show()


# In[19]:


fig, axs = plt.subplots(1,2, figsize =(16,6))

## First Plot
axs[0].scatter(df_p['minutes'], df_p['total_points'], s = df_p['pscore'], c = df_p['element_type'], cmap = 'viridis', alpha =0.7)
axs[0].set_ylabel("Total points earned")
axs[0].set_xlabel("Minutes Played")
axs[0].set_title("Position Based Threat & Score")
fig.colorbar(axs[0].collections[0], ax = axs[0], label ='element Type')

## Second Plot
df_top = df_p.sort_values(by='threat', ascending=False).head(20)
axs[1].barh(df_top['Full_Name'], df_top['threat'], color='skyblue')
axs[1].set_title("Players and threat score")
axs[1].set_xlabel("Top Players")
axs[1].set_label("Threat Score")
axs[1].invert_yaxis()

plt.tight_layout()
plt.show()


# #### List comprehensions
# #### Simplify traditional approach to one line
# 
# #### 1.) squares = []
# #### for number in range(11):
# #### if number %2 == 0
# #### squares.append(number**2)
# 
# ### squares = [number**2 for number in range(11) if number % 2 == 0]
# ### Condition - For loop - if
# 

# In[20]:


df_p


# In[21]:


df


# ### Top Performers by Position

# In[22]:


df_p.sample(5)


# In[23]:


high_perf = df_p.groupby('element_type').apply(lambda x: x.nlargest(3, 'total_points'))[['Full_Name', 'total_points', 'minutes']]
high_perf


# ### Efficiency Metric

# In[24]:


df_p


# In[25]:


eff = df_p.query('minutes>=750').nsmallest(10,'pscore')[['Full_Name', 'threat']]
eff


# In[26]:


df.sample(5)


# In[27]:


df['gc'] = df['goals_scored'] + df['assists']
df['gc_per_game'] = (df['gc'] / df['minutes']) * 90
df.query('minutes > 0').sort_values('gc_per_game', ascending=False).head(10)


# In[28]:


no_rules = df.query('yellow_cards > 5 or red_cards > 1').sort_values('yellow_cards', ascending=False)[['Full_Name', 'yellow_cards','selected_by_percent', 'influence', 'gc_per_game']]
no_rules.head(15)


# ## ICT Index Breakdown - influence, threat and creativity

# In[29]:


# Ensure these columns are numeric
cols = ['influence', 'threat', 'creativity']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

# Now you can safely scale them
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['norm_influence', 'norm_threat', 'norm_creativity']] = scaler.fit_transform(df[cols])

# Compute the custom ICT score
df['ict_custom'] = (
    0.5 * df['norm_influence'] +
    0.3 * df['norm_threat'] +
    0.2 * df['norm_creativity']
)

# View the top 10
print(df[['first_name', 'second_name', 'ict_custom']].sort_values('ict_custom', ascending=False).head(10))


# In[30]:


cor_matrix = df.corr(numeric_only=True)
cor_matrix['total_points'].sort_values(ascending=False)


# ### Popularity & Performance

# In[31]:


high_perf = df['total_points'] > df['total_points'].quantile(0.6)
low_select = df['selected_by_percent'] < df['selected_by_percent'].quantile(0.4)
outliers = df[high_perf & low_select]


# In[32]:


import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'notebook'  # or 'browser', 'iframe', etc.

fig = px.scatter(
    outliers,
    x = 'selected_by_percent',
    y = 'total_points',
    hover_data=['Full_Name', 'element_type'],
    labels = {'selected_by_percent': '% Selected by Managers',
              'total_points': 'Total Fantasy Points'
             }
)
fig.update_traces(marker = dict(size=10, color='red'), textposition='top center')
fig.update_layout(height=500)

fig.show()


# In[33]:


pip install plotly


# In[34]:


print(outliers.shape)


# In[35]:


outliers


# In[ ]:





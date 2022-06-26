
# importing the required library
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import matplotlib
from copy import deepcopy

# read a titanic.csv file
# from seaborn library
# df = sns.load_dataset('titanic')
font_size = 22
font = {'size' : font_size - 4, 'family': 'serif'}
df = pd.read_csv('xlm.csv')

xlabels = ["XNLI", "NER", "POS"]
labels = ["MLM", "XLM", "DICT-MLM", "Align-MLM"]
# print(type(df))

# df = pd.melt(df, id_vars=['Pre-training '], value_vars=['B'],
#         var_name='myVarname', value_name='myValname')
df['XNLI BZ-BS'] = -df['XNLI BZ-BS'].values
df.rename(columns = {'XNLI BZ-BS':'XNLI BS-BZ'}, inplace = True)
df['NER BZ-BS'] = -df['NER BZ-BS'].values
df.rename(columns = {'NER BZ-BS':'NER BS-BZ'}, inplace = True)
df['POS BZ-BS'] = -df['POS BZ-BS'].values
df.rename(columns = {'POS BZ-BS':'POS BS-BZ'}, inplace = True)

df = df.round(0)

df = df.melt(id_vars=['Pre-training method', 'Transformation'], value_vars=['XNLI BS-BZ', 'NER BS-BZ', 'POS BS-BZ'], var_name='Metric', value_name='Score')
# tidy = df.pivot_table(values='value', index=['Transformation, variable'], columns='Pre-training method')
# tidy =df.pivot(index=['Transformation, variable'], columns='Pre-training method')['value']
df = df.set_index(['Pre-training method','Metric','Transformation'])['Score'].unstack().rename_axis(columns = None).reset_index()

pretrain_order = CategoricalDtype(
    ['MLM', 'XLM', 'Dict-MLM', 'Aligned-MLM'], 
    ordered=True
)
metric_order = CategoricalDtype(
    ['XNLI BS-BZ', 'NER BS-BZ', 'POS BS-BZ'], 
    ordered=True
)
df['Pre-training method'] = df['Pre-training method'].astype(pretrain_order)
df = df.sort_values('Pre-training method')
df['Metric'] = df['Metric'].astype(metric_order)
df = df.sort_values('Metric')

num_columns = ['Trans', 'Trans + Inv', 'Trans + Syn']
offset = -2
df_shift = deepcopy(df)
for c in num_columns:
    df_shift[c] -= offset

# print(df_shift)
print(df)

sns.set(style="dark")
width = 0.20  # the width of the bars

plt.figure()
matplotlib.rc('font', **font)
fig, ax = plt.subplots(figsize=(7.17, 6))

sns.barplot(x = 'Metric',
            y = 'Trans',
            hue = 'Pre-training method',
            data = df_shift,
            bottom = offset).set_title('$\mathcal{T}_{trans}$', fontsize=font_size+6)
# sns.barplot(x='Factor', y='Value', hue='Variable', data=tidy, ax=ax1)

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, fontsize=font_size-2, loc='upper left')

for container in ax.containers:
    ax.bar_label(container)

ax.set_ylabel('$\Delta$', fontsize=font_size+6, labelpad=14, rotation=0)
plt.yticks(fontsize=font_size+2)
ax.set_ylim(top=50, bottom=-3)
ax.set(xlabel=None)
ax.set_xticklabels(xlabels, fontsize=font_size)

fig.tight_layout()

# plt.show()
plt.savefig('Trans.pdf')

plt.figure()
matplotlib.rc('font', **font)
fig, ax = plt.subplots(figsize=(7, 6))
sns.barplot(x = 'Metric',
            y = 'Trans + Inv',
            hue = 'Pre-training method',
            data = df_shift,
            bottom = offset).set_title('$\mathcal{T}_{trans} \circ \mathcal{T}_{inv}$', fontsize=font_size+6)

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, fontsize=font_size-9, loc='upper left')
# plt.legend([],[], frameon=False)

for container in ax.containers:
    ax.bar_label(container)

# ax.set(ylabel='$\Delta_{(BS-BZ)}$', fontsize=font_size+2)
# ax.set_ylabel('$\Delta_{(BS-BZ)}$', fontsize=font_size+6, labelpad=14)
plt.yticks(fontsize=font_size+2)
ax.set_ylim(top=50, bottom=-3)
ax.set(ylabel=None)
ax.set(xlabel=None)
ax.set_xticklabels(xlabels, fontsize=font_size)

fig.tight_layout()
 
# plt.show()
plt.savefig('Trans + Inv.pdf')

plt.figure()
matplotlib.rc('font', **font)
fig, ax = plt.subplots(figsize=(7, 6))
sns.barplot(x = 'Metric',
            y = 'Trans + Syn',
            hue = 'Pre-training method',
            data = df_shift,
            bottom = offset).set_title('$\mathcal{T}_{trans} \circ \mathcal{T}_{syn}$', fontsize=font_size+6)

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, fontsize=font_size-2, loc='upper left')
# plt.legend([],[], frameon=False)

for container in ax.containers:
    ax.bar_label(container)

# ax.set(ylabel='$\Delta_{(BS-BZ)}$', fontsize=font_size+2)
# ax.set_ylabel('$\Delta_{(BS-BZ)}$', fontsize=font_size+6, labelpad=14)
plt.yticks(fontsize=font_size+2)
ax.set_ylim(top=50, bottom=-3)
ax.set(ylabel=None)
ax.set(xlabel=None)
ax.set_xticklabels(xlabels, fontsize=font_size)

fig.tight_layout()

plt.savefig('Trans + Syn.pdf')

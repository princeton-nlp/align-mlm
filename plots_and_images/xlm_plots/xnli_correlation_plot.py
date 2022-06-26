import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
sns.set()

font_size=18
font = {'size' : font_size}
matplotlib.rc('font', **font)
matplotlib.rcParams['lines.markersize'] = matplotlib.rcParams['lines.markersize'] + 8

# Define the means and variance of the athletes plot
alignment_trans = np.array([90.0, 0.3, 57.3, 97.7, 85.6, 98.6, 95.7, 64.2, 93.7, 98.4, 95.2, 98.2])[0::3]
alignment_trans_inv = np.array([90.0, 0.3, 57.3, 97.7, 85.6, 98.6, 95.7, 64.2, 93.7, 98.4, 95.2, 98.2])[1::3]
alignment_trans_syn = np.array([90.0, 0.3, 57.3, 97.7, 85.6, 98.6, 95.7, 64.2, 93.7, 98.4, 95.2, 98.2])[2::3]
# alignment = np.array([7, 5, 4, 2, 1, 6, 3])

alpha = 0.0

xnli_trans = np.array([1.5, 17.2, 4.5, -0.2, 8.2, 1.9, 1.1, 9.2, 2.2, -0.5, 6.7, 3.6])[0::3]
xnli_trans_inv = np.array([1.5, 17.2, 4.5, -0.2, 8.2, 1.9, 1.1, 9.2, 2.2, -0.5, 6.7, 3.6])[1::3]
xnli_trans_syn = np.array([1.5, 17.2, 4.5, -0.2, 8.2, 1.9, 1.1, 9.2, 2.2, -0.5, 6.7, 3.6])[2::3]

ner_trans = np.array([2.6, 43.3, 19.0, 1.1, 19.4, 11.9, 1.1, 19.3, 13.8, 0.8, 15.5, 13.1])[0::3]
ner_trans_inv = np.array([2.6, 43.3, 19.0, 1.1, 19.4, 11.9, 1.1, 19.3, 13.8, 0.8, 15.5, 13.1])[1::3]
ner_trans_syn = np.array([2.6, 43.3, 19.0, 1.1, 19.4, 11.9, 1.1, 19.3, 13.8, 0.8, 15.5, 13.1])[2::3]

pos_trans = np.array([0.4, 38.5, 2.0, 0.2, 33.2, 0.9, 0.3, 19.0, 1.0, 0.1, 3.0, 0.8])[0::3]
pos_trans_inv = np.array([0.4, 38.5, 2.0, 0.2, 33.2, 0.9, 0.3, 19.0, 1.0, 0.1, 3.0, 0.8])[1::3]
pos_trans_syn = np.array([0.4, 38.5, 2.0, 0.2, 33.2, 0.9, 0.3, 19.0, 1.0, 0.1, 3.0, 0.8])[2::3]

# Set plot limits
# xmin = 5.5
# xmax=7.5
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.set_xlim(xmin,xmax)
# ax.set_ylim(y-0.1,y+1)

# Plot the data
plt.scatter(xnli_trans - alpha, alignment_trans - alpha, s = 130, color='g', marker='^', label='$\mathcal{T}_{trans}$')
plt.scatter(xnli_trans_inv - alpha, alignment_trans_inv - alpha, s = 130, color='r', marker='^', label='$\mathcal{T}_{trans} \circ \mathcal{T}_{inv}$')
plt.scatter(xnli_trans_syn - alpha, alignment_trans_syn - alpha, s = 130, color='b', marker='^', label='$\mathcal{T}_{trans} \circ \mathcal{T}_{syn}$')

# plt.scatter(ner, alignment, color='g', marker='*', label='NER')
# plt.scatter(pos + alpha, alignment + alpha, color='b', marker='*', label='POS')


# Show the plot
plt.xlabel(r'$\Delta$', fontsize=font_size)
plt.ylabel("Alignment", fontsize=font_size, labelpad=8)
plt.tight_layout()
plt.xticks(fontsize=font_size-2)
plt.yticks(fontsize=font_size-2)

plt.text(-0.5, 3, 'Spearman\'s\n' + r'$\rho = 0.727$, $p < .01$', fontsize = font_size-3, color='darkblue')
# plt.text(-50, 65, 'Spearman\'s correlation\n' + r'$\rho = 0.93$, $p < .005$', fontsize = font_size-2)
# plt.text(-60, 65, 'Spearman\'s correlation\n' + r'$\rho = 0.89$, $p < .01$', fontsize = font_size-2)

plt.legend(prop={'size': font_size-2})

# Add annotations for different points
delta=-0.5
alpha=0.7
dict_design = dict(facecolor='black', shrink=0.05, width=2, headwidth=8, alpha=0.15)
ax.annotate('Align-MLM', xy=(6.3, 93), xytext=(2.1, 83), xycoords='data', textcoords='data', fontsize=font_size-6, arrowprops=dict_design, alpha=alpha)
ax.annotate('XLM', xy=(8.0, 83), xytext=(5.5, 71), xycoords='data', textcoords='data', fontsize=font_size-6, arrowprops=dict_design, alpha=alpha)
ax.annotate('DICT-MLM', xy=(9.1, 61), xytext=(5.6, 47), xycoords='data', textcoords='data', fontsize=font_size-6, arrowprops=dict_design, alpha=alpha)
ax.annotate('MLM', xy=(16.9, 2), xytext=(14.3, 9), xycoords='data', textcoords='data', fontsize=font_size-6, arrowprops=dict_design, alpha=alpha)

# ax.annotate('Parallel', xy=(6.3, 93), xytext=(2.5, 80), xycoords='data', textcoords='data', fontsize=font_size-6, arrowprops=dict_design, alpha=alpha)

# ax.annotate('Non-parallel (Same)', xy=(-3.8 + delta, 43), xytext=(-15, 42), xycoords='data', textcoords='data', fontsize=font_size-6, arrowprops=dict_design, alpha=alpha)
# ax.annotate('Non-parallel (Diff)', xy=(-5.7 + delta, 11.8), xytext=(-15, 25), xycoords='data', textcoords='data', fontsize=font_size-6, arrowprops=dict_design, alpha=alpha)
# ax.annotate(r'$\mathbf{\mathcal{T}}_{trans}}$' + r'$\circ$' + r'$\mathbf{\mathcal{T}}_{inv}}$', xy=(-19.2 + delta, 0.16), xytext=(-25, 10), xycoords='data', textcoords='data', fontsize=font_size-4, arrowprops=dict_design, alpha=alpha)
# ax.annotate(r'$\mathbf{\mathcal{T}}_{trans}}$' + r'$\circ$' + r'$\mathbf{\mathcal{T}}_{perm}}$', xy=(-27.7 + delta, 0.01), xytext=(-28, 25), xycoords='data', textcoords='data', fontsize=font_size-4, arrowprops=dict_design, alpha=alpha)
# ax.annotate(r'$\mathbf{\mathcal{T}}_{trans}}$' + r'$\circ$' + r'$\mathbf{\mathcal{T}}_{syn}}$', xy=(-5.7 + delta, 57.3), xytext=(-15, 55), xycoords='data', textcoords='data', fontsize=font_size-4, arrowprops=dict_design, alpha=alpha)
# ax.annotate('Non-parallel (50%)', xy=(-9.2 + delta, 4.9), xytext=(-19, 13), xycoords='data', textcoords='data', fontsize=font_size-6, arrowprops=dict_design, alpha=alpha)

# plt.savefig('../images/correlation.png')
plt.savefig('XNLI_correlation_annotated.pdf')
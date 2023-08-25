import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcdefaults()

sns.set_style('ticks',  {"axes.grid" : "True", "grid.color": ".4", "grid.linestyle": ":"})
sns.set_context('paper')
plt.rcParams.update({'mathtext.default':  'regular',
#                       'mathtext.fontset' : 'cm',
#                       "font.family": "Helvetica",
#                       "text.usetex": True
                     })
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.labelsize']  =15
plt.rcParams['xtick.labelsize']  =15
plt.rcParams['legend.fontsize']  = 16
plt.rcParams['axes.labelsize' ] = 16
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.markersize'] = 15


factor = 6
figsize = ((1 + 5**.5)/2 * factor, 1 * factor)
half_figsize = ((1 + 5**.5)/4 * factor, 1 * factor)
square_figsize = ((1 + 5**.5)/2 * factor, (1 + 5**.5)/2 * factor)

style_scatter = {
    'facecolor' : 'black',
    'edgecolors' : 'black',
    'lw' : 1,
}


ne_style = {
    'alpha' : 0.9
    }


so_style = {
    'c' : 'white',
    'alpha' : 0.9
    }

so_style_curve = {
    'markerfacecolor' : 'white',
    'alpha' : 0.9,
}

one_shot_style = {
    'marker' : '^' ,
    # 'markevery' : 3, 
    # 'ms' : 10
    }

one_shot_style_curve = {
    'markevery' : 3,
} | one_shot_style


planning_style = {
    'marker' : 'p' ,
    # 'markevery' : (0,2),
    # 'ms' : 10,
    # 'lw' : 1.5
    }

planning_style_curve = {
    'markevery' : (0,2), 
} | planning_style

receding_style = { 
    'marker' : 'd', 
    # 'markevery' : (1,2),  
    # 'ms' : 10, 
    # 'ls' : 'dotted',
    # 'lw' : 1.5
    }

receding_style_curve = {
    'markevery' : (1,2), 
} | receding_style

ne_style_legend = {
    'color' : 'black',
            }

so_style_legend = {
    'c' : 'white',
    'markeredgecolor' : 'black'
            }


repeated_handle = (Line2D([], [], linewidth= 0, ls= '', label='Repeated one-shot', **ne_style_legend, **one_shot_style), plt.Line2D([], [], ls= '', **one_shot_style, label='Repeated one-shot', **so_style_legend))
planning_handle = (Line2D([], [], linewidth= 0, ls= '', label='Planning', **ne_style_legend, **planning_style), plt.Line2D([], [], ls= '', **planning_style, label='Planning', **so_style_legend))
receding_handle = (Line2D([], [], linewidth= 0, ls= '', label='Receding', **ne_style_legend, **receding_style), plt.Line2D([], [], ls= '', **receding_style, label='Receding', **so_style_legend))
# label_cbar_CO2 = 'Total CO2 Emissions (in thousands of Gt)'
# label_axis_temperature = 'Projected Temperature Change by {FINAL_YEAR} (in °C)'


# ##### title

# title_temperature = f'Temperature Trajectories under Various Solutions ({FIRST_YEAR}-{FINAL_YEAR})'
# title_emission = f'CO2 Emission Trajectories under Various Solutions ({FIRST_YEAR}-{FINAL_YEAR})'

# ##### x-label

# x_label_time = f'Years ({FIRST_YEAR}-{FINAL_YEAR})'


# ##### y-label

# y_label_temperature = 'Projected Temperature Variation (°C)'
# y_label_emission = 'Annual Global CO2 Emissions (Gt)'
# y_label_utility = r"Player's Utility"
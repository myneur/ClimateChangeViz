# Viz
import matplotlib
import matplotlib.path as mpath
import matplotlib.pyplot as plt

import numpy as np

import datetime
import bisect

import traceback

colors = ['black','#3DB5AF','#61A3D2','#EE7F00', '#E34D21']

class Charter:        
  def __init__(self, data, variable=None, range=None, format='png'):
    self.data = data
    self.variable = variable
    self.range = range
    self.format = format

  def chartstacked(self, what='series'):
    try:
      fig, ax = plt.subplots(1, 1)

      models = set(self.data.model.values.flat)
      ax.set(title=f'Local tropic days projection ({len(models)} CMIP6 models)', ylabel='Tropic days annualy')  

      self._xaxis_climatic(ax)

      years = self.data.year.values #years = self.data.coords['year'].values

      # one value, no buckets
      #tasmax_max = self.data.tasmax.max(dim='experiment').mean(dim='model')
      #plt.bar(years, tasmax_max.squeeze().values)

      bucket_values = self.data.bucket.squeeze()  
      bins = self.data.bins.values
      #bins = sorted(bins, reverse=True)
      bottom = np.zeros(len(years)) 
      bucket_sums = bucket_values.mean(dim='model').max(dim='experiment')
      palette = ["#FCED8D", "#FF9F47", "#E04B25", "#7A0B0A", "#330024"][2:]
      colors = plt.cm.hot(range(len(bins)))
      for i, bin_label in enumerate(bins):
          bin_values = bucket_sums.sel(bins=bin_label).values
          ax.bar(years, bin_values, label=f'{bin_label} °C', bottom=bottom, color=palette[i], width=1)
          bottom += bin_values

      handles, labels = ax.get_legend_handles_labels()
      ax.legend(handles, labels, loc='upper left', frameon=False)

      # CONTEXT
      context = "models: " + ' '.join(map(str, models)) # +'CMIP6 projections. Averages by 50th quantile. Ranges by 10-90th quantile.'
      plt.text(0.5, 0.005, context, horizontalalignment='center', color='#cccccc', fontsize=6, transform=plt.gcf().transFigure)
      print(context)

      # OUTPUT
      fig.savefig(f'charts/chart_{self.variable}_{len(models)}m.'+self.format)
      plt.show()
    
    except Exception as e: print(f"\nError in Viz: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)

  def chart(self, what='mean', zero=None, reference_lines=None, labels=None):
    try:
      fig, ax = plt.subplots(1, 1)

      models = set(self.data.model.values.flat)
      if self.variable == 'temperature':
        ax.set(title=f'Global temperature projections ({len(models)} CMIP6 models)', ylabel='Temperature')  
      else:
        ax.set(title=f'Maximal temperature (in Czechia) projections ({len(models)} CMIP6 models)', ylabel='Max Temperature (°C)')  

      # SCALE
      if zero and not (np.isnan(zero) and not np.isinf(zero)):
        if zero:
          yticks = [0, 1.5, 2, 3, 4]
          plt.gca().set_yticks([val + zero for val in yticks])
          ax.set_ylim([-1 +zero, 4 + zero])
          plt.gca().set_yticklabels([f'{"+" if val > 0 else ""}{val:.1f} °C' for val in yticks])
      else:
        if False and self.variable == 'max_temperature':
          ax.set_ylim([34, 40])

      self._xaxis_climatic(ax)

      if reference_lines: 
        if not zero:
          zero = 0
        ax.axhline(y=zero+reference_lines[0], color='#717174') # base
        for ref in reference_lines[1:]:
          ax.axhline(y=zero+ref, color='#E34D21', linewidth=.5)
        plt.grid(axis='y')

      current_year = datetime.datetime.now().year
      ax.axvline(x=current_year, color='lightgray', linewidth=.5)

      
      # DATA
      if what == 'mean':
        series = self.data.experiment.values
        legend = [labels[s] for s in series]
        for i in np.arange(len(series)):
          try:
            ax.plot(self.data.year, self.range['mean'][i,:], color=f'{colors[i%len(colors)]}', label=f'{legend[i]}', linewidth=1.3)
            ax.fill_between(self.data.year, self.range['top'][i,:], self.range['bottom'][i,:], alpha=0.05, color=f'{colors[i]}')
          except Exception as e: print(f"Error: {type(e).__name__}: {e}")
      else:
        years = self.data.coords['year'].values
        legend = self.data.model.values
        for i, model in enumerate(self.data.coords['model'].values):
          try:
            ax.plot(years, self.data.sel(model=model).values.squeeze(), color=f'{colors[i%len(colors)]}', label=model, linewidth=1.3)
          except Exception as e: print(f"Error: {type(e).__name__}: {e}")

      handles, labels = ax.get_legend_handles_labels()
      ax.legend(handles, labels, loc='upper left', frameon=False)


      # CONTEXT
      context = "models: " + ' '.join(map(str, models)) # +'CMIP6 projections. Averages by 50th quantile. Ranges by 10-90th quantile.'
      plt.text(0.5, 0.005, context, horizontalalignment='center', color='#cccccc', fontsize=6, transform=plt.gcf().transFigure)
      print(context)

      # OUTPUT
      fig.savefig(f'charts/chart_{self.variable}_{len(models)}m.'+ self.format)
      plt.show()
    
    except Exception as e: print(f"Visualization\nError: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)

  def _xaxis_climatic(self, ax):
    current_year = datetime.datetime.now().year
    ax.set(xlim=(1850, 2100))
    plt.subplots_adjust(left=.08, right=.97, top=0.95, bottom=0.15)
    ax.yaxis.label.set_size(14)

    xticks_major = [1850, 2000, 2015, 2050, 2075, 2100]
    bisect.insort(xticks_major, current_year)
    xticks_minor = [1900, 1945, 1970, 1995, 2020, 2045, 2070, 2095]
    xtickvals_minor = ['Industrial Era', 'Baby Boomers', '+1 gen', '+2 gen', '+3 gen', '+4 gen', '+5 gen', '+6 gen']

    ax.set_xticks(xticks_major) 
    ax.set_xticklabels(xticks_major)
    ax.set_xticks(xticks_minor, minor=True)  
    ax.set_xticklabels(xtickvals_minor, minor=True, rotation=45, va='bottom', ha='right',  fontstyle='italic', color='#b2b2b2', fontsize=9)
    ax.xaxis.set_tick_params(which='minor', pad=70, color="white")

    ax.axvline(x=current_year, color='lightgray', linewidth=.5)


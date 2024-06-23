# Viz
import matplotlib
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import numpy as np

import datetime
import bisect

import traceback

colors = ['black','#3DB5AF','#61A3D2','#EE7F00', '#E34D21']

class Charter:        
  def __init__(self, data, variable=None, range=None, format='png', size=None):
    self.data = data
    self.variable = variable
    self.range = range
    self.format = format
    self.size = size

  def stack(self, title=None, ylabel=None, what='series', marker=None):
    try:
      fig, ax = plt.subplots(1, 1)
      if self.size: fig.set_size_inches(self.size[0], self.size[1])
      if title: ax.set(title=title)  
      if ylabel: ax.set(title=title)  

      models = set(self.data.model.values.flat)

      self._xaxis_climatic(ax, marker=marker)        

      years = self.data.year.values #years = self.data.coords['year'].values

      # one value, no buckets
      #tasmax_max = self.data.tasmax.max(dim='experiment').mean(dim='model')
      #plt.bar(years, tasmax_max.squeeze().values)

      bucket_values = self.data.bucket.squeeze()  
      bins = self.data.bins.values
      #bins = sorted(bins, reverse=True)
      bottom = np.zeros(len(years)) 
      bucket_sums = bucket_values.mean(dim='model').max(dim='experiment')      
      palette = ["#E0C030", "#E0AB2F", "#E0952F", "#E0762F", "#E0572F", "#E0382F", "#BA2D25", "#911B14", "#690500"] #palette = ["#FCED8D", "#FF9F47", "#E04B25", "#7A0B0A", "#330024"][2:] #palette = ["#D9BC2B", "#D99518", "#D98218", "#D9601A", "#BF3111"][3:] # palette = ["#F0E500", "#F5BC00", "#DB8000", "#F54E00", "#EB0C00"][3:]
      palette = [palette[5], palette[8]]

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

  def plot(self, what='mean', title=None, ylabel=None, zero=None, reference_lines=None, labels=None, marker=None):
    try:
      fig, ax = plt.subplots(1, 1)
      if self.size: fig.set_size_inches(self.size[0], self.size[1])
      if title: ax.set(title=title)  
      if ylabel: ax.set(title=title)  

      models = set(self.data.model.values.flat)
      
      # SCALE
      if zero and not (np.isnan(zero) and not np.isinf(zero)):
        if zero:
          yticks = [0, 1.5, 2, 3, 4]
          plt.gca().set_yticks([val + zero for val in yticks])
          ax.set_ylim([-1 +zero, 4 + zero])
          plt.gca().set_yticklabels([f'{"+" if val > 0 else ""}{val:.1f} °C' for val in yticks])
      else:
        if self.variable == 'max_temperature':
          #ax.set_ylim([34, 40])
          ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f} °C'))

      self._xaxis_climatic(ax, marker=marker)

      if reference_lines: 
        if not zero:
          zero = 0
        ax.axhline(y=zero+reference_lines[0], color='#717174') # base
        for ref in reference_lines[1:]:
          ax.axhline(y=zero+ref, color='#E34D21', linewidth=.5)
        plt.grid(axis='y')

      
      # DATA
      if what == 'mean':
        series = self.data.experiment.values
        legend = [labels[s] for s in series]
        for i in np.arange(len(series)):
          try:
            ax.plot(self.data.year, self.range['mean'][i,:], color=f'{colors[i%len(colors)]}', label=f'{legend[i]}', linewidth=1.8)
            
            alpha=0.03 if legend[i] == 'hindcast' else 0.07
            ax.fill_between(self.data.year, self.range['top'][i,:], self.range['bottom'][i,:], alpha=alpha, color=f'{colors[i]}')
          except Exception as e: print(f"Error: {type(e).__name__}: {e}")
      else:
        years = self.data.coords['year'].values
        legend = self.data.model.values
        for i, model in enumerate(self.data.coords['model'].values):
          try:
            ax.plot(years, self.data.sel(model=model).values.squeeze(), color=f'{colors[i%len(colors)]}', label=model, linewidth=1.8)
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

  def _xaxis_climatic(self, ax, marker=None):
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
    ax.set_xticklabels(xtickvals_minor, minor=True, rotation=35, va='bottom', ha='right',  fontstyle='italic', color='#b2b2b2', fontsize=11)
    ax.xaxis.set_tick_params(which='minor', pad=70, color="white")

    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
      if tick == xticks_major[-1]: label.set_ha('right')
      elif tick == current_year: label.set_ha('left')
    
    if marker:
      ax.axvline(x=marker, color='white', linewidth=.5)
    else:
      ax.axvline(x=current_year, color='lightgray', linewidth=.5)

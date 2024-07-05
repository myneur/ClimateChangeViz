# Viz
import matplotlib
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import numpy as np

import datetime
import bisect

import traceback

class Charter:        
  def __init__(self, variable=None, models=[], title=None, subtitle=None, ylabel=None, format='png', size=None, zero=None, reference_lines=None, ylimit=None, marker=None):
    self.variable = variable
    self.models = set(models)
    self.format = format
    self.size = size
    self.marker = marker

    fig, ax = plt.subplots(1, 1)
    if self.size: fig.set_size_inches(self.size[0], self.size[1])
    if title: ax.set(title=title)  
    if subtitle: ax.text(0.5, .98, subtitle, ha='center', va='center', transform=ax.transAxes, fontsize=12, color='lightgray')
    if ylabel: ax.set(ylabel=ylabel)  
    self.fig = fig
    self.ax = ax

    self.palette = {
      'heat': ["#E0C030", "#E0AB2F", "#E0952F", "#E0762F", "#E0572F", "#E0382F", "#BA2D25", "#911B14", "#690500"],
      'series': ['black','#3DB5AF','#61A3D2','#EE7F00', '#E34D21']}

    if zero: 
      self.zero(zero)
    else:
      self._zero = 0
      if self.variable == 'max_temperature':
        #ax.set_ylim([34, 40])
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f} °C'))

    if ylimit: self.ylimit(ylimit)

    if reference_lines: self.reference_lines(reference_lines)
    self._xaxis_climatic(self.ax)


  def zero(self, zero):
    self._zero = zero
    if not (np.isnan(zero) and not np.isinf(zero)):
      yticks = [0, 1.5, 2, 3]
      plt.gca().set_yticks([val + zero for val in yticks])
      plt.gca().set_yticklabels([f'{"+" if val > 0 else ""}{val:.1f} °C' for val in yticks])

  def ylimit(self, y):
    self._ylimit = y
    zero = self._zero if self._zero else 0
    self.ax.set_ylim([y[0]+zero, y[1]+zero])

  def reference_lines(self, reference_lines):
    zero = self._zero if self._zero else 0
    self.ax.axhline(y=zero+reference_lines[0], color='#717174') # base
    for ref in reference_lines[1:]:
      self.ax.axhline(y=zero+ref, color='#E34D21', linewidth=.5)
    plt.grid(axis='y')

  def show(self):
    # CONTEXT
    context = "models: " + ' '.join(map(str, self.models)) # +'CMIP6 projections. Averages by 50th quantile. Ranges by 10-90th quantile.'
    plt.text(0.5, 0.005, context, horizontalalignment='center', color='#cccccc', fontsize=6, transform=plt.gcf().transFigure)
    print(context)
    plt.show()

  def save(self):
    self.fig.savefig(f'charts/chart_{self.variable}_{len(self.models)}m.'+self.format)
    
  def stack(self, data):
    ax = self.ax
    try:

      years = data.year.values #years = data.coords['year'].values

      # one value, no buckets
      #tasmax_max = data.tasmax.max(dim='experiment').mean(dim='model')
      #plt.bar(years, tasmax_max.squeeze().values)


      data = data.bucket.squeeze()  
      bins = data.bins.values
      #bins = sorted(bins, reverse=True)
      bottom = np.zeros(len(years)) 

      palette = self.palette['heat'] 
      palette = [palette[5], palette[8]]

      colors = plt.cm.hot(range(len(bins)))
      for i, bin_label in enumerate(bins):
          bin_values = data.sel(bins=bin_label).values
          ax.bar(years, bin_values, label=f'{bin_label} °C', bottom=bottom, color=palette[i], width=1)
          bottom += bin_values

      handles, labels = ax.get_legend_handles_labels()
      ax.legend(handles, labels, loc='upper left', frameon=False)
    
    except Exception as e: print(f"\nError in Viz: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)

  def rightContext(self, data):
    ax = self.ax
    for d in data:
      ax.plot([ax.get_xlim()[1]], [d], 'ro')
      ax.annotate('Point', 
        xy=(1, d), 
        xycoords='axes fraction', 
        xytext=(10, 0), textcoords='offset points',
        arrowprops=dict(facecolor='black', shrink=0.05),
        horizontalalignment='right', verticalalignment='bottom')

  def scatter(self, data, label=None):
    ax = self.ax
    ax.scatter(data.index.tolist(), data.iloc[:, 0], marker='o', color='black', s=7, label=label)
    ax.legend(frameon=False)

  def plot(self, data, labels=None, models=[], ranges=False, alpha=None, color=None, series='experiment', dimensions=None, linewidth=1.8):
    ax = self.ax
    colors = self.palette['series']
    try:    

      self.models = self.models | set(models)

      # TODO refactor so it's all in one xarray
      years = data[0].year
      series = data[0][series].values
      
      if labels:
        legend = [labels[s] for s in series] if labels else series

      for i in np.arange(len(series)):
        try:
          if ranges: 
            alpha=0.03 if legend[i] == 'hindcast' else 0.07
          else:
            if not alpha: alpha = 1
          if ranges:
            #for quantile in data[ranges].values.flat:
            ax.fill_between(years, data[0][i,:], data[-1][i,:], alpha=alpha, color=f'{colors[i]}')
          else:
            #for model in data[dimension].values.flat:
            for serie in data:
              if labels:
                ax.plot(years, serie[i,:], color=f'{colors[i%len(colors)] if not color else color}', label=f'{legend[i]}', linewidth=linewidth, alpha=alpha)
              else:
                ax.plot(years, serie[i,:], color=f'{colors[i%len(colors)] if not color else color}', linewidth=linewidth, alpha=alpha)
          
        except Exception as e: print(f"Error in {legend[i] if labels else ''}: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)
      

      handles, labels = ax.get_legend_handles_labels()
      ax.legend(handles, labels, loc='upper left', frameon=False)
    
    except Exception as e: print(f"Visualization\nError: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)



  def plotDiscovery(self, data, ranges=None, what='mean', labels=None):
    ax = self.ax
    colors = self.palette['series']
    try:
      self.models = set(data.model.values.flat)
      

      # DATA
      if what == 'mean':
        series = data.experiment.values
        legend = [labels[s] for s in series] if labels else series
        for i in np.arange(len(series)):
          try:
            alpha=0.03 if legend[i] == 'hindcast' else 0.07
            for line in ranges[1:-1]:
              ax.plot(
                data.year, 
                line[i,:], 
                color=f'{colors[i%len(colors)]}', label=f'{legend[i]}', linewidth=1.8)
            ax.fill_between(data.year, ranges[0][i,:], ranges[-1][i,:], alpha=alpha, color=f'{colors[i]}')
          except Exception as e: print(f"Error in {legend[i]}: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)
      else:
        years = data.coords['year'].values
        legend = data.model.values

        if len(self.models)<len(data.model.values):
          data = data.groupby('model').mean()
        
        dimension = list(what.keys())[0]
        if dimension:
          data = data.where(data[dimension] == what[dimension], drop=True)


        for i, model in enumerate(data.coords['model'].values):
          try:
            model_data = data.sel(model=model, drop=True)

            # making it robust to inconsistencies in the data
            try:
              model_data = model_data[list(data.data_vars)[0] ].squeeze()
            except: pass
            model_data = model_data.dropna(dim='year') 
            aligned_years = model_data.coords['year'].values  

            #assert len(aligned_years) == len(data.values), "Mismatch in the dimensions of years and the selected data"
            ax.plot(aligned_years, model_data.values, color=f'{colors[i % len(colors)]}', label=model, linewidth=1.8)

            # TODO make it robust for multile models with the same name

          except Exception as e: 
            if len(model_data.values)==0:
              print(f'No data for {what[dimension]} in {model}'); traceback.print_exc(limit=1)
            else:
              print(f"Error in {model}: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)
              print(f"Shapes, x: {model_data.values.shape}, y: {aligned_years.shape}")
              print("Do shapes match? If not, select the variable to show.")
              #for dim in model_data.dims: print(f"  {dim}: {model_data.values.shape}")
              #print(model_data)

      handles, labels = ax.get_legend_handles_labels()
      ax.legend(handles, labels, loc='upper left', frameon=False)
    
    except Exception as e: print(f"Visualization\nError: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)



  def _xaxis_climatic(self, ax, marker=None):
    current_year = datetime.datetime.now().year
    ax.set(xlim=(1850, 2100))
    plt.subplots_adjust(left=.08, right=.97, top=0.95, bottom=0.15)
    ax.yaxis.label.set_size(14)
    xticks_major = [1850, 1900, 1950, 2000, 2015, 2050, 2075, 2100]
    #bisect.insort(xticks_major, current_year)

    xticks_minor = [1899, 1945, 1970, 1995, 2020, 2045, 2070, 2095]
    xtickvals_minor = ['Industrial\nEra', 'Baby\nBoomers', '+1 gen', '+2 gen', '+3 gen', '+4 gen', '+5 gen', '+6 gen']

    ax.set_xticks(xticks_major) 
    ax.set_xticklabels(xticks_major)
    ax.set_xticks(xticks_minor, minor=True)  
    ax.set_xticklabels(xtickvals_minor, minor=True, rotation=25, va='bottom', ha='center',  fontstyle='italic', color='#b2b2b2', fontsize=11)
    ax.xaxis.set_tick_params(which='minor', pad=70, color="white")

    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
      if tick == xticks_major[-1]: label.set_ha('right')
      elif tick == current_year: label.set_ha('left')
    
    if self.marker:
      ax.axvline(x=self.marker, color='white', linewidth=.5)
    #else: ax.axvline(x=current_year, color='lightgray', linewidth=.5)

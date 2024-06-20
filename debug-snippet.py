# with open('ClimateProjections.py', 'r') as f: exec(f.read())
# with open('debug-exec-chunks.py', 'r') as f: exec(f.read())
# debug()




data_ssp126 = data.sel(experiment='ssp126')
data_ssp126 = data_ssp126.drop_vars("height")


legend = data_ssp126.model.values

for model in data_ssp126.model:
    try:
        model_data = data_ssp126.sel(model=model)
        ax.plot(model_data.year, model_data, label=f'{model}', linewidth=1.3)
    except Exception as e:
        print(f"Error plotting model {model.values}: {e}")





exit()


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper left', frameon=False)


# OUTPUT
fig.savefig(f'chart_t_{len(set(data.model.values.flat))}m.png')
plt.show()

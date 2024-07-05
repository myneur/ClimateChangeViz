import json
import zipfile
import os
import traceback
import pandas as pd

def snippet():
  with open('debug-exec-chunks.py', 'r') as f: exec(f.read())

def loadMD(where):
  try:
    with open("metadata/"+where + ".json", 'r') as f: 
      return json.load(f)
  except Exception as e:
    print(f"Error in loading JSON: {type(e).__name__}: {e}")

def saveMD(md, where):
  try:
    with open("metadata/"+where+'.json', 'w') as f: 
      #json.dump(md, f)
      f.write(json.dumps(md, indent=2))
  except Exception as e:
    print(f"Error in saving JSON: {type(e).__name__}: {e}")


def unzip(filename, DATADIR):
  print(f"UNZIPPPING {filename} {DATADIR}")
  with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(DATADIR)

def unzipAll(DATADIR):
  for filename in os.listdir(DATADIR):
    if f'{DATADIR}filename'.endswith('.zip'):
      unzip(filename, DATADIR)

def debug(msg, e, limit=1):
  print(f"Error in {filename}: {type(e).__name__}: {e}"); traceback.print_exc(limit=limit)


def loadTCRfromNatureComArticle():
  xls = 'nature.com/CMIP6 ECS 41586_2022_CM20344978_MOESM1_ESM.20352690.xls'
  #table = pd.ExcelFile(xls)
  sheet_tcr = pd.read_excel(xls, sheet_name='ECSTCR')
  models = sheet_tcr.iloc[2:60, 0].reset_index(drop=True)
  tcrs = sheet_tcr.iloc[2:60, 3].reset_index(drop=True)
  ecs = sheet_tcr.iloc[2:60, 1].reset_index(drop=True)
  df = pd.DataFrame({'model': models, 'tcr': tcrs, 'ecs': ecs})
  return df

def addMissing(model_names, df):
  missing_values = [model for model in model_names if model not in df['model'].values]
  missing_models = pd.DataFrame({'model': missing_values})
  all_models = pd.concat([df, missing_models], ignore_index=True)
  return all_models

def selectModels():
  #nature = loadTCRfromNatureComArticle()
  #nature = nature[nature['tcrs']>0] 
  #md = loadMD('models')
  #models = pd.DataFrame(md['all_models']).T.reset_index()
  #models = models.rename(columns={"index": "model"})
  #models_all = addMissing(nature['model'], models)
  #models_all['nature.com'] = models_all['model'].isin(nature['model'].values)
  #models_all = pd.DataFrame(md)
  #status = util.loadMD('status') 
  #models_copernicus = ['INM-CM4-8', 'MCM-UA-1-0', 'BCC-CSM2-MR', 'IITM-ESM', 'MIROC-ES2L', 'FGOALS-f3-L', 'CNRM-CM6-1-HR', 'HadGEM3-GC31-LL', 'NorESM2-MM', 'IPSL-CM6A-LR', 'CanESM5-CanOE', 'MPI-ESM1-2-LR', 'CAMS-CSM1-0', 'GFDL-ESM4', 'NESM3', 'CNRM-ESM2-1', 'NorESM2-LM', 'MIROC6', 'AWI-CM-1-1-MR', 'MRI-ESM2-0', 'FIO-ESM-2-0', 'CNRM-CM6-1', 'TaiESM1', 'ACCESS-CM2', 'INM-CM5-0', 'FGOALS-g3', 'UKESM1-0-LL', 'KACE-1-0-G']
  #models_all['Copernicus'] = models_all['model'].isin(models_copernicus)
  
  
  models = pd.read_csv('metadata/models.csv')#, delimiter=';')
  data = loadMD('model_md')['model_tcrs'].items()
  df = pd.DataFrame(data)
  df.rename(columns={df.columns[0]: 'model'}, inplace=True)
  df.rename(columns={df.columns[1]: 'compare'}, inplace=True)
  print("missing")
  print ([model for model in df['model'].values if model not in models['model'].values])
  print("Merge")
  models_all = pd.merge(models, df[['model', 'compare']], on='model', how='left')
  #models_all = models_all.sort_values(by='model')
  print(models_all)
  return
  models_all.to_csv('metadata/models.csv', index=False)#, sep='\t')#, columns=columns)

  #status = util.saveMD(status, 'status') 

  #models_all.to_json('metadata/models.json', orient='records', lines=True)
  #columns = list(models_all.columns)
  #columns = columns[:1] + columns[1:]
  
  
  
  #likely = df[(df['tcrs'] >= 1.4) & (df['tcrs'] <= 2.2)]
  return models_all





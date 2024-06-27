import json
import zipfile
import os
import traceback

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
      json.dump(md, f)
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
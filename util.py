import json

def debug():
  with open('debug-exec-chunks.py', 'r') as f: exec(f.read())

def loadMD(where):
  try:
    with open(where + ".json", 'r') as f: 
      return json.load(f)
  except Exception as e:
    print(f"Error in loading JSON: {type(e).__name__}: {e}")

def saveMD(md, where):
  try:
    with open(where+'.json', 'w') as f: 
      json.dump(md, f)
  except Exception as e:
    print(f"Error in saving JSON: {type(e).__name__}: {e}")


def unzip(filename):
  with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(f'{DATADIR}')

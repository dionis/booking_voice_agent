import json
import os
import multiline

PATH_DIRECTORY = os.getcwd() + os.sep + ".." + os.sep + "datasets"

directorires = os.listdir(PATH_DIRECTORY)
print(directorires)

for ifile in directorires:
    print(f"Open file {ifile}")
    with open( PATH_DIRECTORY + os.sep + ifile, encoding="utf-8" ) as f:
        #content =multiline.load(f, multiline=True)
        content = json.load(f)
        print(content)

## Insert in Options table
## Insert in pgvector table
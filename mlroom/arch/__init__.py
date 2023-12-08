import os

# Get the directory of the current file (__init__.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
       # __import__(filename[:-3])
        __import__(f"mlroom.arch.{filename[:-3]}")
        #importlib.import_module()
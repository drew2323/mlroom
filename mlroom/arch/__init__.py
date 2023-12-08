import os

for filename in os.listdir("mlroom/arch"):
    if filename.endswith(".py") and filename != "__init__.py":
       # __import__(filename[:-3])
        __import__(f"mlroom.arch.{filename[:-3]}")
        #importlib.import_module()
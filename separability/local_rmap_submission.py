import os


dir = "logs-separability"

for file in os.scandir(dir):
    for line in file.readline():
        pass
    line
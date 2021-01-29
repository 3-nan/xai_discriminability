import os


dir = "logs-quantification"

for filename in os.listdir(dir):

    if ".o" in filename:
        file = open(dir + "/" + filename, "r")

        # print(file.readlines()[-1])
        lastrow = file.readlines()[-1]

        if lastrow != "Job executed successfully.\n":
            print(filename)

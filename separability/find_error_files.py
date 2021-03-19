import os


dir = "logs-zennit"

for filename in os.listdir(dir):

    if ".o" in filename:
        file = open(dir + "/" + filename, "r")

        # print(file.readlines()[-1])
        try:
            lastrow = file.readlines()[-1]
        except IndexError:
            print(filename)

        if lastrow != "Job executed successfully.\n":
            print(filename)

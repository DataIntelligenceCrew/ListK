f = open("requirements.txt")
file = f.read()
file = file.split("\n")
output = ""
for i in file:
    output = output +  "\"" + i + "\",\n"
print(output)
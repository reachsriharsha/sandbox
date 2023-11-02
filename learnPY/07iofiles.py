

myfile = open('test.txt')
print(myfile.read())
myfile.seek(0)
print(myfile.readlines())
myfile.close()

with open('myout.txt',mode ='w') as mywFile:
    mywFile.write("This is new record\n")
    
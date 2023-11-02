# if
# elif
# else


if 4 % 2 == 0:
    print("This is a even number")
else:
    print("Its odd number")


if True:
    print("This is true")
else:
    print("This is false")

city = "bengaluru"

if city == "bengaluru":
    print("I am in Bengaluru")
elif city == "mangaluru":
    print("I am in Mangaluru")
else:
    print("I am no where")


mylist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for num in mylist:
    if num % 2 == 0:
        print(num, " is even number")
    else:
        print(num, " is odd number")

my_tup_list = [(2, 4), (4, 6), (6, 8), (8, 10)]

for item in my_tup_list:
    print(item)

dict = {'kar': 'blr', 'tn': 'chn'}

for k, v in dict.items():
    print(k, "-->", v)

if 2 in mylist:
    print("value present")
else:
    print("2 is not in list")

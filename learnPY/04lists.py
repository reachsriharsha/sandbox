list_one = [1,2,3]
print(list_one)
list_two = ['string', 100,12]
print(len(list_two))

list_two = ['one','two',3]
print(list_two)

new_list = list_one + list_two
print(new_list)
new_list[5]= 'three'
print(new_list)
new_list.append('seven')
print(new_list)
popped_item = new_list.pop()
print("popped:", popped_item,new_list)

# import json
#
# with open('demo.txt', "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# print(data)


def test(*x):
    print(x)


x_list = [[1, 2, 3], [4, 5, 6]]

test(*x_list)

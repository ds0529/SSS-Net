import pdb
################################# S3DIS ##########################################
# list = \
#     [0.9002732967075557, 0.9648169811018599, 0.763926456421416, 0.0008677318743356972, 0.26342804910643747,
#      0.4052574631704659, 0.38454134704056625, 0.6568673095244594, 0.7577948153618194, 0.3300281744640201,
#      0.5645670023424729, 0.5973337518186823, 0.421072797136883]
#
# str = ''
#
# for i in range(len(list)):
#     str = str + '&%.2f'% (list[i]*100) + '\t'
# str = str + '\\\\'
#
# print(str)

input='72.83	94.21	41.55	67.14	76.26	60.92	59.25	33.78	41.92	64.81	10.86	38.37	43.38	43.91	26.46	46.96	74.73	44.34	72.48	33.39'
temp = 0
output=''
for i,c in enumerate(input):
    if c=='\t':
        str = input[temp: i]
        output = output+'&' + str + '\t'
        temp = i+1
str = input[temp:]
output = output+'&' + str + '\t\\\\'
print(output)

# err = 50.1 * 13 - 100 * sum(list)
#
# print(list[-2]*100 + err)

################################# ScanNet #######################################
# with open("result.txt","r",encoding="utf-8") as f:
#     content = f.read()
#
# # print(content)
# # pdb.set_trace()
#
# istr_list = content.split('\n')
# #print(len(istr_list))
# # pdb.set_trace()
#
# num_list = []
# for s in istr_list:
#     word_list = s.split(' ')
#     word = word_list[3]
#     # print(word)
#     # pdb.set_trace()
#     num = float(word[:6]) *100
#     num_list.append(num)
#
# str = ''
#
# for i in range(len(num_list)):
#     str = str + '%.2f'% (num_list[i]) + '\t'
#
# print(str)

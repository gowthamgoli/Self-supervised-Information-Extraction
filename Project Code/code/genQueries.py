#f = open('relations.txt', "r")
f1 = open('queries.txt','w')

lines = [line.rstrip('\n') for line in open('relations.txt')]

#lines = f.readlines()
sentence = ''
for line in lines:
	if line:
		if line.startswith('sentence'):
			sentence = line[9:]
		elif line.startswith('tuple'):
			x,q1, q2, q3 = line.split(',')
			f1.write(sentence +':::' +'_ '+q2+' _\n')
			f1.write(sentence +':::' +q1 + ' ' + q2 + ' _\n')
			f1.write(sentence +':::' +q1 + ' _ ' + q3 + '\n')
			f1.write(sentence +':::' +'_ ' + q2 + ' ' + q3 + '\n')

f1.close()
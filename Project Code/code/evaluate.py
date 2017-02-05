lines = [line.rstrip('\n') for line in open('answers_labeled.txt')]

right = 0
total = 0

for line in lines:
	if line:
		if line[0] == '1':
			right = right + 1
		total = total + 1

print total
print right
print 'accuracy: ', 1.0*right/total
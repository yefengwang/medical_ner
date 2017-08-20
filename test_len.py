import sys
max_len = 0
for line in open(sys.argv[1]):
	
	line = line.strip()
	if not line:
		continue
	ls = line.split(' ')
	word = ls[0]
	if len(word) > max_len:
		max_len = len(word)
		print(word, max_len)
	
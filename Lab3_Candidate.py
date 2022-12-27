import csv

f = open('ts.csv')
csv_file = csv.reader(f)
data = list(csv_file)

for i in data:
    if i[-1] == "n":
        specific = ['o'] * len(data[0][:-1])
    else:
        k = data.index(i)
        specific = data[k][:-1]
        break
data = data[k:]

general = [['?' for i in range(len(specific))] for j in range(len(specific))]

for i in data:
        if i[-1] == "y":
            for j in range(len(specific)):
                if i[j] != specific[j]:
                    specific[j] = "?"
                    general[j][j] = "?"

        elif i[-1] == "n":
            for j in range(len(specific)):
                if i[j] != specific[j]:
                    general[j][j] = specific[j]
                else:
                    general[j][j] = "?"

        print("\nStep " + str(data.index(i)+1) + " of Candidate Elimination Algorithm")
        print("S[{0}]: ".format(data.index(i)),specific)
        print("G[{0}]: ".format(data.index(i)),general)

gh = [] # gh = general Hypothesis
for i in general:
        for j in i:
            if j != '?':
                gh.append(i)
                break
print("\nFinal Specific hypothesis:\n", specific)
print("\nFinal General hypothesis:\n", gh)
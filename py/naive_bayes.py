import sys


class NaiveBayes:
    def __init__(self):
        self.ham = {}
        self.spam = {}
        self.post_spam = -1.0
        self.default_spam = -1.0
        self.default_ham = -1.0

    def fit(self, letters, classes):
        total_spam = 0
        total_messages = len(classes)

        for letter, ltype in (zip(letters, classes)):
            if ltype == "ham":
                d = self.ham
            elif ltype == "spam":
                total_spam += 1
                d = self.spam
            else:
                raise Exception("U R MOM GAY!")
            for word in letter:
                d[word] = d.get(word, 0) + 1

        total_ham = total_messages - total_spam
        self.post_spam = total_spam / total_messages

        # loop through dictionaries to normalise & Laplace smooth
        self.default_spam = 1 / (total_spam + 2)
        self.default_ham = 1 / (total_ham + 2)

        for key in iter(self.ham):
            self.ham[key] = (self.ham[key] + 1) * self.default_ham
        for key in iter(self.spam):
            self.spam[key] = (self.spam[key] + 1) * self.default_spam

    def predict(self, letter):
        p_spam = self.post_spam
        p_ham = 1 - self.post_spam
        cal = p_ham / p_spam
        # includes Laplace smoothing for words we haven't seen yet by default values
        # Calculates probability in a way that reduces rounding error
        for word in letter:
            cal = cal * (self.ham.get(word, self.default_ham) / self.spam.get(word, self.default_spam))
        prob = 1 / (cal + 1)
        return prob


if len(sys.argv) > 1 and sys.argv[1] == "debug":
    test = open("input.txt", "r")
else:
    test = sys.stdin

n = int(test.readline())
letters = []
classes = []
for i in range(n):
    _, t = test.readline().split(' ')
    a = [int(x) for x in test.readline().split(' ')]
    letters.append(a)
    if t == "L\n":
        classes.append("ham")
    elif t == "S\n":
        classes.append("spam")
    else:
        raise Exception("WTF")

clf = NaiveBayes()
clf.fit(letters, classes)

n = int(test.readline())
for i in range(n):
    test.readline()
    a = [int(x) for x in test.readline().split(' ')]
    # print(clf.predict(a))
    if clf.predict(a) > 0.8:
        print("S")
    else:
        print("L")

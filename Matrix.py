class Matrix:
    def __init__(self):
        self.n_ue = 0
        self.n_gnb = 0
        self.n_time = 0
        self.matrix = list()

    def read_from(self, path):
        fd = open(path, 'r')
        lines = fd.readlines()

        self.n_ue = len(lines[0].split())
        for line in lines:
            if line != '\n':
                self.n_gnb += 1
            else:
                break

        arr = list()
        for line in lines:
            if line != '\n':
                arr.append(list(map(int, line.split())))
            else:
                continue

        self.n_time = len(arr) // self.n_gnb

        self.matrix = [[[0 for x in range(self.n_time)] for y in range(self.n_ue)] for z in range(self.n_gnb)]
        cnt = 0
        for t in range(self.n_time):
            for g in range(self.n_gnb):
                for u in range(self.n_ue):
                    self.matrix[g][u][t] = arr[cnt][u]
                cnt += 1

    def print(self):
        for t in range(self.n_time):
            for g in range(self.n_gnb):
                for u in range(self.n_ue):
                    if self.matrix[g][u][t] == 0:
                        print('  -', end='')
                    else:
                        print(f'{self.matrix[g][u][t]:>3}', end='')
                print()
            print()


import sys

class ParamFile():
    def __init__(self, filename):
        self.filename = filename
        self.read_parameters()

    def read_parameters(self):
        self.params = {}
        with open(self.filename) as f:
            fr = f.readlines()

            fr = [f[:-1] for f in fr]

            frs = [f.split(" = ") for f in fr]

            for stuff in frs:
                try:
                    i, j = stuff
                    self.params[str(i).strip()] = eval(j)
                except (ValueError, SyntaxError) as ex:
                    print(f"ValueError of SyntaxError: Either ")#'{j}' or '{j}' are on the wrong form!")
                    #sys.exit()
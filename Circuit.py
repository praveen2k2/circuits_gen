class Cricuit:
    def __init__(self, name):
        self.name = name
        self.components = []

    def __add__(self, other):
        if isinstance(other, Cricuit):
            self.components=self.components+other.components 
            return self
        else:
            self.components.append(other)
            return self  
    
    def __str__(self):
        return self.components.__str__()
    
if __name__ == '__main__':
    c = Cricuit('circuit')
    c2 = Cricuit('circuit2')
    c=c+1
    c=c+2
    c=c+'hh'
    c2+1
    c+c2
    print(c)
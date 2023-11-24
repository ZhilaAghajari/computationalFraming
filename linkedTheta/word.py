import json

class Word():
    '''
    Word class to help in storing assigned theta role and topic for current word, along with its
    relations and word id.
    '''
    def __init__(self, content, idx, z, y, y1, y2, relns: list, arg2s:list):
        self.content = content
        self.idx = idx
        self.z = z
        self.y = y
        self.y1 = y1
        self.y2 = y2
        self.relns = relns
        self.arg2s = arg2s

    def __str__(self):
        return json.dumps(self.__dict__)
    
    
#     cute: nsubj.gov 
#         arg2: baby
        
#     baby: nsubj.dep
        
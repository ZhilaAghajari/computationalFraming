import json

class Word():
    '''
    Word class to help in storing assigned theta role and topic for current word, along with its
    relations and word id.
    '''
    def __init__(self, content, idx, z, y, relns: list):
        self.content = content
        self.idx = idx
        self.z = z
        self.y = y
        self.relns = relns

    def __str__(self):
        return json.dumps(self.__dict__)
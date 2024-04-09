import random

from conversationgenome.Utils import Utils

class logging:
    def info(a1, a2=None, a3=None):
        print(a1, a2, a3,)

class MockBt:
    def __init__(self):
        self.logging = logging()

    def getUids(self, num=10, useFullGuids=False):
        uids = []
        for i in range(num):
            # useGuids is more realistic, but harder to read in testing
            if useFullGuids:
                uids.append(Utils.guid())
            else:
                uids.append(random.randint(1000, 9999))

        return uids

from collections import namedtuple

Block = namedtuple('Block', ['previous', 'data'])


class HashPointer(object):

    def __init__(self, block):
        self._storage = (hash(block), block)

    @property
    def hash(self):
        return self._storage[0]

    @property
    def block(self):
        return self._storage[1]

    def __hash__(self):
        return hash(self._storage)


class BlockChain(object):

    def __init__(self):
        self.chain = [Block(None, None)]
        self.hash = hash(self.chain[-1])

    def append(self, data):
        previous_ptr = HashPointer(self.chain[-1])
        block = Block(previous_ptr, data)
        self.chain.append(block)
        self.hash = hash(block)

    def verify(self):
        for i, (cur, nxt) in enumerate(zip(self.chain[:-1], self.chain[1:])):
            cur_hash = hash(cur)
            if cur_hash != nxt.previous.hash:
                raise AssertionError(
                    'block with data {!r} not consistent with hash '
                    'pointer of next block: hash(block) != next_ptr.hash: '
                    '{} != {}'.format(cur.data, cur_hash, nxt.previous.hash))
        last_hash = hash(self.chain[-1])
        if last_hash != self.hash:
            raise AssertionError(
                'last block with data {!r} not consistent with top-level '
                'hash: hash(last_block) != self.hash: {} != {}'
                ''.format(self.chain[-1].data, last_hash, self.hash))

    def hashes(self):
        return (tuple(block.previous.hash for block in self.chain[1:]) +
                (self.hash,))

    def __len__(self):
        return len(self.chain)

    def __iter__(self):
        return iter(self.chain)

    def __getitem__(self, index):
        return self.chain[index]


Transaction = namedtuple('Transaction', ['id', 'message'])


class ScroogeCoin(BlockChain):

    def __init__(self):
        self.chain = [Block(None, Transaction(id=0, message=''))]
        self.hash = hash(self.chain[-1])
        self.id_counter = 1

    def append(self, message):
        previous_ptr = HashPointer(self.chain[-1])
        block = Block(previous_ptr, Transaction(self.id_counter, message))
        self.chain.append(block)
        self.hash = hash(block)
        self.id_counter += 1


def create_scrooge_coin(scrooge_chain, value, owner):
    scrooge_chain.verify()
    message = 'CREATE|{}|{}'.format(value, owner)
    scrooge_chain.append(message)


# Everything that follows must be horrible in runtime (traversal every time)


# %% Test
data1 = 'I am a message'
data2 = 'Send 1 Euro to Wikipedia'

chain = BlockChain()
chain.append(data1)
chain.verify()
chain.append(data2)
chain.verify()

scrooge_chain = ScroogeCoin()
create_scrooge_coin(scrooge_chain, value=2.0, owner=hash('Holger'))

# %%
# tamper: change data of an entry
data1_evil = 'Push the red button'
chain.chain[1] = Block(HashPointer(chain.chain[0]), data1_evil)
chain.verify()  # AssertionError

# tamper: insert an entry
data_evil = 'Push the red button'
chain.chain.insert(1, Block(HashPointer(chain.chain[0]), data_evil))
chain.verify()  # AssertionError

# tamper: remove an entry
chain.chain.pop(1)
chain.verify()  # AssertionError

# tamper: change data and next hash pointer
data1_evil = 'Push the red button'
chain.chain[1] = Block(HashPointer(chain.chain[0]), data1_evil)
chain.chain[2] = chain.chain[2]._replace(prev=HashPointer(chain.chain[1]))
chain.verify()  # AssertionError

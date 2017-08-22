# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

X = T.scalar()
# X = T.vector()
# X = T.matrix()

Y = T.scalar()

Z = X + Y

A = T.matrix()

A.max(axis = 1)
T.max(A)

A.T
#10x2
#A.reshape((20))
A.dimshuffle((1,0))

B = T.vector()
B.dimshuffle(('x', 0))

import numpy as np
W = theano.shared(np.random.rand(10,10), borrow=False)

W.get_value(borrow=False)

X = T.scalar()
Y = T.scalar()

Z = X + Y

add = theano.function([X,Y], Z)

add(1,2)

W = theano.shared(np.arange(10))

from collections import OrderedDict

update = OrderedDict()
update[W] = 2*W

multiply_by_two = theano.function([], updates = update)

W.get_value()

multiply_by_two()

L = (W**2).sum()

T.grad(L, wrt = W)

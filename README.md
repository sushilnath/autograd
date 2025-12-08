# Autograd
Simple auto grad library. This is meant for learning and most of the code is built without using any auto grad library or available helpers.

Autograd can automatically differentiate a given function. A function is built by composing other function nodes.  The autograd library can find the derivative of the resulting function w.r.t each node.

Autograd relies on product and chain rules from multi variate calculus. These rules work when the functions are well behaved, which usually means continuity, existence of directional derivatives, continuity of directional derivatives etc. More details can be found in calculus books.

There are two versions shown here one which build the function graph and calculated the derivative as the function is being built, the other is where the function is composed and lambda to compute the derivative is stored and when a "backward pass" is called then the derivatives are actually calculated. The latter version is faster when the last layer nodes in the graph are smaller in number.

In simpler terms autograd can be thought at the sum of products of the edges in the resulting graph, and depending on the structure of the graph derivative message passing can be optimized.
# Autograd
Simple auto grad library. This is meant for learning and most of the code is built wihtout using any auto grad library or available helpers.

Autograd take can automatically differentiate a given function. A function is built by composing other function nodes.  The autograd library can find the derivative of the resulting function w.r.t each node.

Autograd relies on produce and chain rules from multi variate calculus. These rules work when the functions are well behaved, which usually means continuity, existence of directional derivatives, continuitity of directional derivates etc. More details can be found in calculus books.

There are two versions shown here one which build the function graph and calculated the drivate as the function is being built, the other is where the function is composed and lambda to compute the derivate is stores and when a "backward pass" is called then the derivates are actually calculated. The latter version is faster when the last layer nodes in the graph are smaller in number.

In simpler terms autograd can be thought at the sum of products of the edges in the resulting graph, and depending on the structure of the graph derivate message passing can be optimized.
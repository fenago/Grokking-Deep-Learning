Chapter 7

I How to picture neural networks

Visualizing using letters instead of pictures
All these pictures and detailed explanations are actually
a simple piece of algebra.
Just as we defined simpler pictures for the matrix
and vector, we can perform the same visualization
in the form of letters.
How do you visualize a matrix using math? Pick
a capital letter. I try to pick one that’s easy to
remember, such as W for “weights.” The little 0
means it’s probably one of several Ws. In this case,
the network has two. Perhaps surprisingly, I could
have picked any capital letter. The little 0 is an extra
that lets me call all my weight matrices W so I can
tell them apart. It’s your visualization; make it easy
to remember.
How do you visualize a vector using math? Pick a
lowercase letter. Why did I choose the letter l? Well,
because I have a bunch of vectors that are layers,
I thought l would be easy to remember. Why did
I choose to call it l-zero? Because I have multiple
layers, it seems nice to make all them ls and number
them instead of having to think of new letters for
every layer. There’s no wrong answer here.
If that’s how to visualize matrices and vectors
in math, what do all the pieces in the network
look like? At right, you can see a nice selection of
variables pointing to their respective sections of the
neural network. But defining them doesn’t show
how they relate. Let’s combine the variables via
vector-matrix multiplication.

W0

Matrix

l0
Vector

l2

W1

weights_1_2

l1
W0

weights_0_1

Licensed to Ernesto Lee <socrates73@gmail.com>

l0

Linking the variables

141

Linking the variables
The letters can be combined to indicate functions
and operations.
Vector-matrix multiplication is simple. To visualize that two letters are being multiplied by
each other, put them next to each other. For example:
Algebra

Translation

l0W0

“Take the layer 0 vector and perform vectormatrix multiplication with the weight matrix 0.”

l1W1

“Take the layer 1 vector and perform vectormatrix multiplication with the weight matrix 1.”

You can even throw in arbitrary functions like relu using notation that looks almost exactly
like the Python code. This is crazy-intuitive stuff.

l1 = relu(l0W0)

“To create the layer 1 vector, take the layer 0 vector
and perform vector-matrix multiplication with the
weight matrix 0; then perform the relu function on
the output (setting all negative numbers to 0).”

l2 = l1W1

“To create the layer 2 vector, take the layer 1 vector
and perform vector-matrix multiplication with the
weight matrix 1.”

If you notice, the layer 2 algebra contains layer 1 as an input variable. This means you
can represent the entire neural network in one expression by chaining them together.
Thus, all the logic in the forward propagation step can
be contained in this one formula. Note: baked into this
l2 = relu(l0W0)W1
formula is the assumption that the vectors and matrices
have the right dimensions.

Licensed to Ernesto Lee <socrates73@gmail.com>

142

Chapter 7

I How to picture neural networks

Everything side by side
Let’s see the visualization, algebra formula,
and Python code in one place.
I don’t think much dialogue is necessary on this page. Take a minute and look at each piece
of forward propagation through these four different ways of seeing it. It’s my hope that you’ll
truly grok forward propagation and understand the architecture by seeing it from different
perspectives, all in one place.
layer_2 = relu(layer_0.dot(weights_0_1)).dot(weights_1_2)

l2 = relu(l0W0)W1

Inputs

Hiddens

Prediction

layer_2

weights_1_2

layer_1

weights_0_1

layer_0

Licensed to Ernesto Lee <socrates73@gmail.com>

The importance of visualization tools

143

The importance of visualization tools
We’re going to be studying new architectures.
In the following chapters, we’ll be taking these vectors and matrices and combining them
in some creative ways. My ability to describe each architecture for you is entirely dependent
on our having a mutually agreed-on language for describing them. Thus, please don’t move
beyond this chapter until you can clearly see how forward propagation manipulates these
vectors and matrices, and how these various forms of describing them are articulated.
Key takeaway
Good neural architectures channel signal so that correlation is easy to discover. Great
architectures also filter noise to help prevent overfitting.

As mentioned previously, a neural architecture controls how signal flows through a network.
How you create these architectures will affect the ways in which the network can detect
correlation. You’ll find that you want to create architectures that maximize the network’s
ability to focus on the areas where meaningful correlation exists, and minimize the
network’s ability to focus on the areas that contain noise.
But different datasets and domains have different characteristics. For example, image data
has different kinds of signal and noise than text data. Even though neural networks can be
used in many situations, different architectures will be better suited to different problems
because of their ability to locate certain types of correlations. So, for the next few chapters,
we’ll explore how to modify neural networks to specifically find the correlation you’re
looking for. See you there!

Licensed to Ernesto Lee <socrates73@gmail.com>

Licensed to Ernesto Lee <socrates73@gmail.com>

learning signal and ignoring noise:
introduction to regularization and batching

In this chapter
•	Overfitting
•	Dropout
•	

Batch gradient descent

With four parameters I can fit an elephant, and with
five I can make him wiggle his trunk.
—John von Neumann, mathematician, physicist,
computer scientist, and polymath

145

Licensed to Ernesto Lee <socrates73@gmail.com>

8

146
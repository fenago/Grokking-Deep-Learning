<img align="right" src="../logo-small.png">

# Lab : Learning to write like Shakespeare: long short-term memory (Part 1)

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/Grokking-Deep-Learning` folder. To copy and paste: use **Control-C** and to paste inside of a terminal, use **Control-V**

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab11_Exploding_Gradients`



## bCharacter language modeling
Let’s tackle a more challenging task with the RNN.

At the end of chapters 12 and 13, you trained vanilla recurrent neural networks (RNNs)
that learned a simple series prediction problem. But you were training over a toy dataset
of phrases that were synthetically generated using rules.

In this lab, you’ll attempt language modeling over a much more challenging dataset:
the works of Shakespeare. And instead of learning to predict the next word given the
previous words (as in the preceding chapter), the model will train on characters. It needs
to learn to predict the next character given the previous characters observed. Here’s what
I mean:

![](./images_14/1.png)

Whereas in chapters 12 and 13 the vocabulary was made up of the words from the dataset,
now the vocabulary is made up the characters in the dataset. As such, the dataset is also
transformed into a list of indices corresponding to characters instead of words. Above this is
the indices NumPy array:

```
embed = Embedding(vocab_size=len(vocab),dim=512)
model = RNNCell(n_inputs=512, n_hidden=512, n_output=len(vocab))

criterion = CrossEntropyLoss()
optim = SGD(parameters=model.get_parameters() + embed.get_parameters(),
alpha=0.05)
```

This code should all look familiar. It initializes the embeddings to be of dimensionality 8
and the RNN hidden state to be of size 512. The output weights are initialized as 0s (not
a rule, but I found it worked a bit better). Finally, you initialize the cross-entropy loss and
stochastic gradient descent optimizer.


## The need for truncated backpropagation
Backpropagating through 100,000 characters is intractable.

One of the more challenging aspects of reading code for RNNs is the mini-batching logic for
feeding in data. The previous (simpler) neural network had an inner for loop like this (the
bold part):

```
for iter in range(1000):
batch_size = 100
total_loss = 0

hidden = model.init_hidden(batch_size=batch_size)

for t in range(5):
input = Tensor(data[0:batch_size,t], autograd=True)
rnn_input = embed.forward(input=input)
output, hidden = model.forward(input=rnn_input, hidden=hidden)

target = Tensor(data[0:batch_size,t+1], autograd=True)
loss = criterion.forward(output, target)
loss.backward()
optim.step()
total_loss += loss.data
if(iter % 200 == 0):
p_correct = (target.data == np.argmax(output.data,axis=1)).mean()
print_loss = total_loss / (len(data)/batch_size)
print("Loss:",print_loss,"% Correct:",p_correct)
```

You might ask, "Why iterate to 5?" As it turns out, the previous dataset didn’t have any
example longer than six words. It read in five words and then attempted to predict the sixth.
Even more important is the backpropagation step. Consider when you did a simple
feedforward network classifying MNIST digits: the gradients always backpropagated all the
way through the network, right? They kept backpropagating until they reached the input
data. This allowed the network to adjust every weight to try to learn how to correctly predict
given the entire input example.

The recurrent example here is no different. You forward propagate through five input
examples and then, when you later call loss.backward(), it backpropagates gradients all
the way back through the network to the input datapoints. You can do this because you
aren’t feeding in that many input datapoints at a time. But the Shakespeare dataset has
100,000 characters! This is way too many to backpropagate through for every prediction.
What do you do?
You don’t! You backpropagate for a fixed number of steps into the past and then stop. This
is called truncated backpropagation, and it’s the industry standard. The length you backprop
becomes another tunable parameter (like batch size or alpha).



## Truncated backpropagation
Technically, it weakens the theoretical maximum
of the neural network.

The downside of using truncated backpropagation is that it shortens the distance a neural
network can learn to remember things. Basically, cutting off gradients after, say, five
timesteps, means the neural network can’t learn to remember events that are longer than five
timesteps in the past.

Strictly speaking, it’s more nuanced than this. There can accidentally be residual information
in an RNN’s hidden layer from more than five timesteps in the past, but the neural network
can’t use gradients to specifically request that the model keep information around from six
timesteps in the past to help with the current prediction. Thus, in practice, neural networks
won’t learn to make predictions based on input signal from more than five timesteps in the
past (if truncation is set at five timesteps). In practice, for language modeling, the truncation
variable is called bptt, and it’s usually set somewhere between 16 and 64:

```
batch_size = 32
bptt = 16
n_batches = int((indices.shape[0] / (batch_size)))
```

The other downside of truncated backpropagation is that it makes the mini-batching logic a
bit more complex. To use truncated backpropagation, you pretend that instead of having one
big dataset, you have a bunch of small datasets of size bptt. You need to group the datasets
accordingly:

```
trimmed_indices = indices[:n_batches*batch_size]
batched_indices = trimmed_indices.reshape(batch_size, n_batches)
batched_indices = batched_indices.transpose()

input_batched_indices = batched_indices[0:-1]
target_batched_indices = batched_indices[1:]

n_bptt = int(((n_batches-1) / bptt))
input_batches = input_batched_indices[:n_bptt*bptt]
input_batches = input_batches.reshape(n_bptt,bptt,batch_size)
target_batches = target_batched_indices[:n_bptt*bptt]
target_batches = target_batches.reshape(n_bptt, bptt, batch_size)
```

There’s a lot going on here. The top line makes the dataset an even multiple between the
batch_size and n_batches. This is so that when you group it into tensors, it’s square
(alternatively, you could pad the dataset with 0s to make it square). The second and third
lines reshape the dataset so each column is a section of the initial indices array. I’ll show
you that part, as if batch_size was set to 8 (for readability):

![](./images_14/2.png)

I’ve highlighted the first column in bold. See how the indices for the phrase "That," are in the
first column on the left? This is a standard construction. The reason there are eight columns
is that the batch_size is 8. This tensor is then used to construct a list of smaller datasets,
each of length bptt.

You can see here how the input and target are constructed. Notice that the target indices are
the input indices offset by one row (so the network predicts the next character). Note again
that batch_size is 8 in this printout so it’s easier to read, but you’re really setting it to 32.

```
print(input_batches[0][0:5])

print(target_batches[0][0:5])

array([[ 9,
[14,
[ 2,
[10,
[57,

43,
44,
41,
39,
39,

21,
39,
39,
57,
43,

10,
21,
54,
48,
1,

10,
43,
37,
21,
10,

23,
14,
21,
54,
21,

57,
1,
26,
38,
21,

46],
10],
57],
43],
33]])

array([[14,
[ 2,
[10,
[57,
[43,

44,
41,
39,
39,
43,

39,
39,
57,
43,
41,

21,
54,
48,
1,
60,

43,
37,
21,
10,
52,

14,
21,
54,
21,
12,

1,
26,
38,
21,
54,

10],
57],
43],
33],
1]])
```

Don’t worry if this doesn’t make sense to you yet. It doesn’t have much to do with deep learning
theory; it’s just a particularly complex part of setting up RNNs that you’ll run into from time to
time. I thought I’d spend a couple of pages explaining it.



## Let’s see how to iterate using truncated backpropagation.

The following code shows truncated backpropagation in practice. Notice that it looks very
similar to the iteration logic from chapter 13. The only real difference is that you generate
a batch_loss at each step; and after every bptt steps, you backpropagate and perform a
weight update. Then you keep reading through the dataset like nothing happened (even
using the same hidden state from before, which only gets reset with each epoch):

![](./images_14/3.png)


## A sample of the output
By sampling from the predictions of the model,
you can write Shakespeare!

The following code uses a subset of the training logic to make predictions using the model.
You store the predictions in a string and return the string version as output to the function.
The sample that’s generated looks quite Shakespearian and even includes characters talking:

![](./images_14/4.png)

I war ded abdons would.
CHENRO:

Why, speed no virth to her,
Plirt, goth Plish love,
Befion
hath if be fe woulds is feally your hir, the confectife to the nightion
As rent Ron my hath iom
the worse, my goth Plish love,
Befion
Ass untrucerty of my fernight this we namn?

ANG, makes:
That's bond confect fe comes not commonour would be forch the conflill
As
poing from your jus eep of m look o perves, the worse, my goth
Thould be good lorges ever word

DESS:
Where exbinder: if not conflill, the confectife to the nightion
As co move, sir, this we namn?

ANG VINE PAET:
There was courter hower how, my goth Plish lo res
Toures
ever wo formall, have abon, with a good lorges ever word.


## Vanishing and exploding gradients
Vanilla RNNs suffer from vanishing and exploding gradients.

You may recall this image from when you first put together a RNN. The idea was to be able
to combine the word embeddings in a way that order mattered. You did this by learning a
matrix that transformed each embedding to the next timestep. Forward propagation then
became a two-step process: start with the first word embedding (the embedding for "Red" in
the following example), multiply by the weight matrix, and add the next embedding ("Sox").
You then take the resulting vector, multiply it by the same weight matrix, and then add in
the next word, repeating until you’ve read in the entire series of words.

![](./images_14/5.png)

But as you know, an additional nonlinearity
was added to the hidden state-generation
process. Thus, forward propagation becomes
a three-step process: matrix multiply the
previous hidden state by a weight matrix,
add in the next word’s embedding, and apply
a nonlinearity.

![](./images_14/6.png)

Note that this nonlinearity plays an
important role in the stability of the
network. No matter how long the sequence
of words is, the hidden states (which could
in theory grow larger and larger over time) are forced to stay between the values of the
nonlinearity (between 0 and 1, in the case of a sigmoid). But backpropagation happens in
a slightly different way than forward propagation, which doesn’t have this nice property.
Backpropagation tends to lead to either extremely large or extremely small values. Large
values can cause divergence (lots of not-a-numbers [NaNs]), whereas extremely small values
keep the network from learning. Let’s take a closer look at RNN backpropagation.



## A toy example of RNN backpropagation
To see vanishing/exploding gradients firsthand,
let’s synthesize an example.

The following code shows a recurrent backpropagation loop for sigmoid and relu
activations. Notice how the gradients become very small/large for sigmoid/relu,
respectively. During backprop, they become large as the result of the matrix multiplication,
and small as a result of the sigmoid activation having a very flat derivative at its tails
(common for many nonlinearities).

![](./images_14/7.png)

## Long short-term memory (LSTM) cells
LSTMs are the industry standard model to counter
vanishing/exploding gradients.

The previous section explained how vanishing/exploding gradients result from the
way hidden states are updated in a RNN. The problem is the combination of matrix
multiplication and nonlinearity being used to form the next hidden state. The solution that
LSTMs provide is surprisingly simple.

**The gated copy trick**
LSTMs create the next hidden state by copying the previous hidden state and then
adding or removing information as necessary. The mechanisms the LSTM uses for adding
and removing information are called gates.

```
def forward(self, input, hidden):
from_prev_hidden = self.w_hh.forward(hidden)
combined = self.w_ih.forward(input) + from_prev_hidden
new_hidden = self.activation.forward(combined)
output = self.w_ho.forward(new_hidden)
return output, new_hidden
```

The previous code is the forward propagation logic for the RNN cell. Following is the new
forward propagation logic for the LSTM cell. The LSTM has two hidden state vectors: h (for
hidden) and cell.

The one you care about is cell. Notice how it’s updated. Each new cell is the previous cell
plus u, weighted by i and f. f is the "forget" gate. If it takes a value of 0, the new cell will
erase what it saw previously. If i is 1, it will fully add in the value of u to create the new cell.
o is an output gate that controls how much of the cell’s state the output prediction is allowed
to see. For example, if o is all zeros, then the self.w_ho.forward(h) line will make a
prediction ignoring the cell state entirely.

```
def forward(self, input, hidden):

prev_hidden, prev_cell = (hidden[0], hidden[1])

f = (self.xf.forward(input) +
i = (self.xi.forward(input) +
o = (self.xo.forward(input) +
u = (self.xc.forward(input) +
cell = (f * prev_cell) + (i *
h = o * cell.tanh()
output = self.w_ho.forward(h)
return output, (h, cell)

self.hf.forward(prev_hidden)).sigmoid()
self.hi.forward(prev_hidden)).sigmoid()
self.ho.forward(prev_hidden)).sigmoid()
self.hc.forward(prev_hidden)).tanh()
u)
```


## Some intuition about LSTM gates
LSTM gates are semantically similar to reading/writing
from memory.

So there you have it! There are three gates—f, i, o—and a cell-update vector u; think of
these as forget, input, output, and update, respectively. They work together to ensure that
any information to be stored or manipulated in c can be so without requiring each update
of c to have any matrix multiplications or nonlinearities applied to it. In other words, you’re
avoiding ever calling nonlinearity(c) or c.dot(weights).

This is what allows the LSTM to store information across a time series without worrying
about vanishing or exploding gradients. Each step is a copy (assuming f is nonzero) plus
an update (assuming i is nonzero). The hidden value h is then a masked version of the cell
that’s used for prediction.

Notice further that each of the three gates is formed the same way. They have their own
weight matrices, but each of them conditions on the input and the previous hidden state,
passed through a sigmoid. It’s this sigmoid nonlinearity that makes them so useful as gates,
because it saturates at 0 and 1:

```
f = (self.xf.forward(input) + self.hf.forward(prev_hidden)).sigmoid()
i = (self.xi.forward(input) + self.hi.forward(prev_hidden)).sigmoid()
o = (self.xo.forward(input) + self.ho.forward(prev_hidden)).sigmoid()
```

One last possible critique is about h. Clearly it’s still prone to vanishing and exploding
gradients, because it’s basically being used the same as the vanilla RNN. First, because the
h vector is always created using a combination of vectors that are squished with tanh and
sigmoid, exploding gradients aren’t really a problem—only vanishing gradients. But this
ends up being OK because h is conditioned on c, which can carry long-range information:
the kind of information vanishing gradients can’t learn to carry. Thus, all long-range
information is transported using c, and h is only a localized interpretation of c, useful for
making an output prediction and constructing gate activations at the following timestep. In
short, c can learn to transport information over long distances, so it doesn’t matter if h can’t.


## The long short-term memory layer
You can use the autograd system to implement an LSTM.

![](./images_14/8.png)


## Upgrading the character language model
Let’s swap out the vanilla RNN with the new LSTM cell.

Earlier in this lab, you trained a character language model to predict Shakespeare.
Now let’s train an LSTM-based model to do the same. Fortunately, the framework from the
preceding chapter makes this easy to do (the complete code from the book’s website, www.
manning.com/books/grokking-deep-learning; or on GitHub at https://github.com/iamtrask/
grokking-deep-learning). Here’s the new setup code. All edits from the vanilla RNN code are
in bold. Notice that hardly anything has changed about how you set up the neural network:

![](./images_14/9.png)



## Training the LSTM character language model
The training logic also hasn’t changed much.

The only real change you have to make from the vanilla RNN logic is the truncated
backpropagation logic, because there are two hidden vectors per timestep instead of one.
But this is a relatively minor fix (in bold). I’ve also added a few bells and whistles that make
training easier (alpha slowly decreases over time, and there’s more logging):

```
for iter in range(iterations):
total_loss, n_loss = (0, 0)

hidden = model.init_hidden(batch_size=batch_size)
batches_to_train = len(input_batches)

for batch_i in range(batches_to_train):

hidden = (Tensor(hidden[0].data, autograd=True),
Tensor(hidden[1].data, autograd=True))
losses = list()

for t in range(bptt):
input = Tensor(input_batches[batch_i][t], autograd=True)
rnn_input = embed.forward(input=input)
output, hidden = model.forward(input=rnn_input, hidden=hidden)

target = Tensor(target_batches[batch_i][t], autograd=True)
batch_loss = criterion.forward(output, target)

if(t == 0):
losses.append(batch_loss)
else:
losses.append(batch_loss + losses[-1])
loss = losses[-1]

loss.backward()
optim.step()

total_loss += loss.data / bptt
epoch_loss = np.exp(total_loss / (batch_i+1))
if(epoch_loss < min_loss):
min_loss = epoch_loss
print()
log = "\r Iter:" + str(iter)
log += " - Alpha:" + str(optim.alpha)[0:5]
log += " - Batch "+str(batch_i+1)+"/"+str(len(input_batches))
log += " - Min Loss:" + str(min_loss)[0:5]
log += " - Loss:" + str(epoch_loss)
if(batch_i == 0):
s = generate_sample(n=70, init_char='T').replace("\n"," ")
log += " - " + s
sys.stdout.write(log)
optim.alpha *= 0.99
```



## Tuning the LSTM character language model
I spent about two days tuning this model, and
it trained overnight.

Here’s some of the training output for this model. Note that it took a very long time to
train (there are a lot of parameters). I also had to train it many times in order to find a
good tuning (learning rate, batch size, and so on) for this task, and the final model trained
overnight (8 hours). In general, the longer you train, the better your results will be.

![](./images_14/10.png)

Intestay thee.

SIR:
It thou my thar the sentastar the see the see:
Imentary take the subloud I
Stall my thentaring fook the senternight pead me, the gakentlenternot
they day them.

KENNOR:
I stay the see talk :
Non the seady!

Sustar thou shour in the suble the see the senternow the antently the see
the seaventlace peake,
I sentlentony my thent:
I the sentastar thamy this not thame.



### Summary
LSTMs are incredibly powerful models.

The distribution of Shakespearian language that the LSTM learned to generate isn’t to be
taken lightly. Language is an incredibly complex statistical distribution to learn, and the
fact that LSTMs can do so well (at the time of writing, they’re the state-of-the-art approach
by a wide margin) still baffles me (and others as well). Small variants on this model either
are or have recently been the state of the art in a wide variety of tasks and, alongside word
embeddings and convolutional layers, will undoubtedly be one of our go-to tools for a long
time to come.

# Wooclap — Deep Learning block (Data Processing, MUIT)

24 multiple-choice questions, 8 per session. The correct answer is marked with `**`
and its position varies from question to question.

---

## Session 1 — Neural networks from scratch

**Q1.** Why is the classification accuracy not used as the training criterion of a neural
network?

- a) Because it is not defined for multiclass problems
- b) Because it does not take the confidence of the prediction into account
- **c) Because it is not differentiable: the threshold has zero derivative almost everywhere**
- d) Because it can only be computed on the validation set

---

**Q2.** A single-layer network with logistic activation trained by minimizing the
cross-entropy is equivalent to:

- a) A linear SVM
- **b) A logistic regression model fitted by maximum likelihood**
- c) A linear regression model fitted by least squares
- d) A naive Bayes classifier with Gaussian likelihoods

---

**Q3.** For the single-layer network, the gradient of the NLL with respect to the weights
turns out to be $\sum_k (q_k - y_k)\mathbf{x}_k$. What is the practical consequence of this
result?

- a) That the network can only be trained with the cross-entropy loss
- b) That the learning rate no longer needs to be tuned
- c) That the gradient is zero whenever the prediction is confident
- **d) That the update is simply "error times input", with no derivative of the activation left in the expression**

---

**Q4.** In the multiclass network, the softmax activation is implemented as
`exp(t - t.max(axis=1, keepdims=True))` instead of `exp(t)`. Why?

- **a) To avoid the overflow of the exponential, without changing the result**
- b) To make the outputs add up to one
- c) To centre the scores so that the gradient is unbiased
- d) To speed up the computation of the exponential

---

**Q5.** In a multi-layer network, what makes the overall model non-linear?

- a) The use of the cross-entropy loss
- b) The one-hot encoding of the labels
- **c) The non-linear activation of the hidden units**
- d) The fact that there is more than one output

If the hidden activations were linear, the whole network would collapse into a single
linear layer.

---

**Q6.** The derivative of the hyperbolic tangent, written in terms of its own output
$h = \tanh(o)$, is:

- a) $(1-h)^2$
- b) $h(1-h)$
- **c) $1-h^2$**
- d) $1-h$

---

**Q7.** A student implements back-propagation by hand, trains the network, and the loss
decreases steadily. What can be concluded about the gradient?

- a) That it is correct, because otherwise the loss would increase
- **b) Very little: a gradient with the right sign but the wrong magnitude still reduces the loss**
- c) That it is correct, provided the learning rate is small enough
- d) That it is correct only if the training accuracy reaches 100 %

---

**Q8.** The single-layer network reaches about 95 % accuracy on the sign-language digits
and only about 60 % on Dogs vs Cats. The main reason is:

- a) The Dogs vs Cats dataset is much larger, so it overfits
- b) The Dogs vs Cats images are in grayscale
- c) The learning rate was not tuned for Dogs vs Cats
- **d) The digits are in a fixed position and scale, so a linear combination of pixels separates them; a dog can be anywhere in the image**

---

## Session 2 — Deep networks with PyTorch

**Q9.** In PyTorch, what does calling `loss.backward()` do?

- a) It updates the network parameters using the current gradients
- **b) It traverses the computation graph backwards and accumulates the gradients in the `.grad` attribute of the tensors**
- c) It computes the loss on the validation set
- d) It deletes the computation graph to free memory, without computing anything

---

**Q10.** Why does every PyTorch training loop call `opt.zero_grad()` (or
`net.zero_grad()`)?

- a) Because the optimizer needs to be reinitialized at each epoch
- b) Because otherwise the learning rate decays automatically
- **c) Because gradients accumulate by default, and without clearing them each step would use the sum of all previous gradients**
- d) Because the gradients of the validation data would otherwise be included

---

**Q11.** What is the difference between `nn.Parameter` and an ordinary tensor inside an
`nn.Module`?

- **a) `nn.Parameter` is registered as a trainable parameter of the module and appears in `net.parameters()`**
- b) `nn.Parameter` is stored in the GPU and an ordinary tensor is not
- c) `nn.Parameter` cannot be modified after initialization
- d) There is no difference; it is only a matter of style

---

**Q12.** When the single-layer network is rewritten with `nn.Linear` and
`nn.CrossEntropyLoss`, it converges noticeably faster. Why?

- a) Because `nn.Linear` uses a different optimization algorithm
- b) Because `nn.CrossEntropyLoss` is a different loss function
- **c) Because `nn.Linear` initializes the weights scaling them with the number of inputs, instead of using a fixed standard deviation**
- d) Because the computation runs on the GPU

---

**Q13.** `nn.CrossEntropyLoss` expects as inputs:

- a) The softmax outputs and the labels in one-hot encoding
- **b) The raw scores of the network (no softmax) and the labels as integers**
- c) The softmax outputs and the labels as integers
- d) The log of the softmax outputs and the labels in one-hot encoding

---

**Q14.** During training on 1649 samples with a batch size of 64, how many parameter
updates happen in one epoch?

- a) 1, because the gradient is computed over the whole training set
- b) 64
- **c) 26**
- d) 1649

---

**Q15.** A network is trained with dropout and evaluated without calling `net.eval()`.
What happens?

- **a) Units are still being switched off at random, so the predictions are noisy and the measured performance is worse than the real one**
- b) Nothing: dropout only affects the backward pass
- c) PyTorch raises an error
- d) The network uses all the units but with the weights scaled incorrectly

---

**Q16.** During training, the validation accuracy saturates while the validation loss
starts to grow. The best interpretation is:

- a) The learning rate is too large
- b) There is a bug in the evaluation code, because both must move together
- c) The validation set is too small
- **d) The network is not making many more mistakes, but it is making them with more confidence**

---

## Session 3 — Convolutional networks

**Q17.** The two structural ideas that distinguish a convolutional layer from a dense layer
are:

- a) Non-linear activation and pooling
- **b) Local connectivity and weight sharing**
- c) Weight sharing and dropout
- d) Local connectivity and batch normalization

---

**Q18.** A `nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)`
applied to a batch of images of size $3\times32\times32$ produces outputs of size:

- a) $6\times30\times30$
- b) $3\times32\times32$
- **c) $6\times32\times32$**
- d) $6\times16\times16$

---

**Q19.** How many trainable parameters does that convolutional layer have?

- a) 18
- **b) 168**
- c) 1152
- d) It depends on the size of the input image

$6\times3\times3\times3 = 162$ weights, plus 6 biases.

---

**Q20.** Which of these statements about pooling layers is correct?

- a) Pooling layers have one weight per element of the window
- b) Max pooling and average pooling give the same result on normalized data
- **c) Pooling has no trainable parameters and reduces the spatial size of the activation maps**
- d) Pooling is applied before the convolution in a standard CNN block

---

**Q21.** Data augmentation is applied:

- **a) Only to the training set, generating a different random transformation at each epoch**
- b) To the training and the validation sets, so that both follow the same distribution
- c) Once before training, generating a larger fixed dataset
- d) Only to the classes with fewer samples

---

**Q22.** For a dataset of handwritten digits, which augmentation is clearly a bad idea?

- a) A small rotation of $\pm 10$ degrees
- b) A small translation
- **c) A horizontal flip**
- d) A small change in brightness

---

**Q23.** In the transfer learning experiment, ResNet18 has about 11.2 million parameters,
of which only 5130 are trained. Those 5130 parameters correspond to:

- a) The first convolutional layer, which is adapted to the new images
- b) The batch normalization layers of the whole network
- **c) The new fully connected layer that replaces the ImageNet classification head**
- d) A random subset of the network, selected by dropout

---

**Q24.** Transfer learning with a frozen backbone, trained on 4000 CIFAR10 images, beats a
small CNN trained from scratch on all 50000. The reason is:

- a) ResNet18 has more parameters, so it has more capacity to fit those 4000 images
- b) The images were resized to $224\times224$, which contains more information
- **c) The frozen convolutional layers already extract generic visual features learnt from a far larger dataset, so only the final decision has to be learnt**
- d) Adam converges faster than SGD on the subset

---

## Suggested use

* **Session 1**: Q1–Q4 after the first notebook block, Q5–Q8 at the end.
* **Session 2**: Q9–Q11 after the autograd block, Q12–Q14 after the abstraction levels,
  Q15–Q16 at the end.
* **Session 3**: Q17–Q20 after the convolution/pooling block, Q21–Q24 at the end.

Q7, Q16 and Q24 are the ones that usually generate discussion; they are worth keeping even
if the quiz has to be shortened.

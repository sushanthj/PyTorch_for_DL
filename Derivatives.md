---
layout: page
title: Derivatives in PyTorch
permalink: /Derivatives/
nav_order: 4
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Derviates

![](/images/derivates.jpeg)

# Partial Derivaties

![](/images/partial_derivatives.jpeg)

# Differentiation in Forward/Backward pass of a Neural Net

Previously we used 'y.backward' when doing the differentiation

Now, let's customize the 'backward' function according the math operator

Let's say: **y = 2x**

```python
class SQ(torch.autograd.Function):


    @staticmethod
    def forward(ctx,i):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        result=i**2
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        i, = ctx.saved_tensors
        grad_output = 2*i
        return grad_output
```

We will now apply the class above using a .apply tag as shown below:

```python
x=torch.tensor(2.0,requires_grad=True )
sq=SQ.apply

y=sq(x)
y
print(y.grad_fn)
y.backward()
x.grad
```
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

## Understanding the variables in derivatives

If we have a simple y=x**2 function, the way each tensor gets defined changes.

Notice how we define the tensor x:
```python
# Create a tensor x

x = torch.tensor(2.0, requires_grad = True)
print("The tensor x: ", x)
```

Now notice how we define y:
```python
# Create a tensor y according to y = x^2

y = x ** 2
print("The result of y = x^2: ", y)
```

Now that the tensors are defined, there are two steps we need to follow to get the derivative at a specific point:
1. y.backward()
2. x.grad

On their own, each would not be sufficient to find grad. However, y.backward() does the actual math part of derivation in the background and the *gradient is stored in the x tensor*

Notice the following attributes of x and y and their respective outputs:

![](/images/derivates.jpeg)

## Passing multiple values as input to a function whose gradient we find

Here we use the similar linspace function (like numpy) to get a range of values:
```python
# Calculate the derivative with multiple values

x = torch.linspace(-10, 10, 10, requires_grad = True)
Y = x ** 2
y = torch.sum(x ** 2)
```

## Plotting the above function in matplotlib

```python
# Take the derivative with respect to multiple value. Plot out the function and its derivative

y.backward()

plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()
```

In the example, we use x.detach and y.detach \
What this does is ensure that further grads or any values are not added as attributes to x and y tensors

Note: The tensors use graphs to populate values of x.grad and y.grad. Hence, these detach functions \
essentially just disable any further sub-graphs being populated to x or y

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

## Torch.is_leaf

![](/images/is_leaf.jpeg)
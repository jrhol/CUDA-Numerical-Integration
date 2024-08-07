# A CUDA implementation of the rectangle mid-point rule

## Summary
1. This program evaluates the integral of a graphical function over a finite range using the rectangle mid-point rule.
2. In this particular example we are evaluating $f(x) = {e^{-x}}^{2}$ between −∞, ∞.
3. However, when plotting this function it can be seen the area under the curve reaches 0 rapidly, hence a finite integration over the range -3,3 is suitable.

<p align="center">
   <img width="316" alt="Screenshot 2024-08-07 at 09 00 23" src="https://github.com/user-attachments/assets/bcf42290-183c-4e29-8d3e-4da6a9a0f67b">
</p>

<p align="center">
Fig 1. Plot of $$f(x) = {e^{-x}}^{2}$$
</p>


## Optimisations
1. Instead of Integrating between -5 to 5, we can integrate between 0-5 and double the result as the function is symmetrical!
   

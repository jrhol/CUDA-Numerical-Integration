# A CUDA implementation of the rectangle mid-point rule

## Summary
1. This program evaluates the integral of a graphical function over a finite range using the rectangle mid-point rule.
2. In this particular example we are evaluating $$ f(x) = {e^{-x}}^{2} $$ between −∞, ∞.
3. However, when plotting this function it can be seen the area under the curve reaches 0 rapidly, hence a finite integration over the range -5,5 is suitable.


## Optimisations
1. Instead of Integrating between -5 to 5, we can integrate between 0-5 and double the result as the function is symmetrical!
   

# mat-dim-checker

`mat-dim-checker` is a tool for checking the correctness of tensor operations in [PyTorch](https://github.com/pytorch/pytorch) using refinement types. Essentially all common tensor operations should be easily checkable with refinement types (very few of them impose more than an equality constraint), and by extension most ML code should be checkable as well.


## Unimplemented
* Refinements
* Arg types for function types
* torch types
* Class variables
* Object variables

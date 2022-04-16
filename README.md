
#Title
A collection of functions, for the testing of different knowledge discovery tasks, and AL models.
The functions can be applied to vectors, but for the sake of the representation, they are applied here to a single 
dimension.

##Hypercube
![img.png](distribution_data_generation/images/cube3d.png)
![img.png](distribution_data_generation/images/cube2d.png)


## Cross
![img.png](distribution_data_generation/images/cross3d.png)
![img.png](distribution_data_generation/images/cross2d.png)

## Double Linear

![img.png](distribution_data_generation/images/DL3d.png)
![img.png](distribution_data_generation/images/DL2d.png)

## Hourglass
![img.png](distribution_data_generation/images/hg3d.png)
![img.png](distribution_data_generation/images/hg2d.png)

## Z
![img.png](distribution_data_generation/images/Z3d.png)
![img.png](distribution_data_generation/images/Z2d.png)

## Linear Periodic
![img.png](distribution_data_generation/images/LP2d.png)

## Linear Step
![img.png](distribution_data_generation/images/LS2d.png)

## Linear Then Dummy
This represents the function f(x) = (x,0.5,0.5,...)
![img.png](distribution_data_generation/images/LTD3d.png)
![img.png](distribution_data_generation/images/LTD2d.png)

## Linear Then Noise
This represents the function f(x) = (x,random, random,...)
![img.png](distribution_data_generation/images/LTN2d.png)
![img.png](distribution_data_generation/images/LTN3d.png)

## Multi Gaussian
Uses a gaussian process to generate random functions
![img.png](distribution_data_generation/images/MG3d.png)
![img.png](distribution_data_generation/images/MG2D.png)

## Non Coexistence
Represents the function f(x<sub>1</sub>,...,x<sub>n</sub>) = f(0,...,x<sub>i</sub>,...) where i is selected randomly 
each time.
![img.png](distribution_data_generation/images/NC2d.png)

## Plus
![img.png](distribution_data_generation/images/p3d.png)
![img.png](distribution_data_generation/images/p2d.png)

## Power
Calculates integer powers of vectors using vector multiplication
![img.png](distribution_data_generation/images/p2.png)
![img.png](distribution_data_generation/images/p3.png)
![img.png](distribution_data_generation/images/p4.png)

## Sine
![img.png](distribution_data_generation/images/Sine.png)

## Star
![img.png](distribution_data_generation/images/star2d.png)
![img.png](distribution_data_generation/images/star3d.png)

## Random
Outputs random vector of the same shape as the input

## Graph
Applies a given lambda to the input to give the output

## Chaotic
Applies a chaotic function to the data to give an output of the same shape
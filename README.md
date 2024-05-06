# Eikonal CUDA Solver
Eikonal CUDA implementation for the Advanced Methods for Scientific Computing (AMSC) Course @Polimi

**Students**:
- [Sabrina Cesaroni](https://github.com/SabrinaCesaroni)
- [Melanie Tonarelli](https://github.com/melanie-t27)
- [Tommaso Trabacchin](https://github.com/tommasotrabacchinpolimi) 

## Introduction
An Eikonal equation is a non-linear first-order partial differential equation 
that is encountered in problems of wave propagation. <br>

An Eikonal equation is one of the form:

$$\begin{cases} 
H(x, \nabla u(x)) = 1 & \quad x \in \Omega \subset \mathbb{R}^d \\  
u(x) = g(x) & \quad x \in \Gamma \subset \partial\Omega 
\end{cases} $$

where 
- $d$ is the dimension of the problem, either 2 or 3;
- $u$ is the eikonal function, representing the travel time of a wave;
- $\nabla u(x)$ is the gradient of $u$, a vector that points in the direction of the wavefront;
- $H$ is the Hamiltonian, which is a function of the spatial coordinates $x$ and the gradient $\nabla u$;
- $\Gamma$ is a set smooth boundary conditions.

In most cases, 
$$H(x, \nabla u(x)) = |\nabla u(x)|_{M} = \sqrt{(\nabla u(x))^{T} M(x) \nabla u(x)}$$
where $M(x)$ is a symmetric positive definite function encoding the speed information on $\Omega$. <br> 
In the simplest cases, $M(x) = c^2 I$ therefore the equation becomes:

$$\begin{cases}
|\nabla u(x)| = \frac{1}{c} & \quad x \in \Omega \\  
u(x) = g(x) & \quad x \in \Gamma \subset \partial\Omega
\end{cases}$$

where $c$ represents the celerity of the wave.

## Description

## Usage
After cloning the repo with the command `git clone https://github.com/AMSC22-23/Eikonal-Cuda-Cesaroni-Tonarelli-Trabacchin.git`, the installation of [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) software is required. To install it, access the repo and run the following:
```bash
$ mkdir lib
$ cd lib
$ git clone https://github.com/KarypisLab/GKlib.git
$ make config
$ make
$ cd ..
$ git clone https://github.com/KarypisLab/METIS.git
$ cd METIS
$ make config shared=1 cc=gcc prefix=~/local
$ make install
```

Then, to build the executable, from the root folder run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
An executable for each test will be created into `/build`, and can be executed through:
```bash
$ ./test_name input-filename num-partitions output-filename
```

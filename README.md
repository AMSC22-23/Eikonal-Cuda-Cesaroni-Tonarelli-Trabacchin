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

The project is a CUDA library designed for computing the Eikonal equation based on the FSM algorithm on 3D unstructured tetrahedrical meshes. It is designed to be highly extensible, and easy to integrate in different applications.<br>

This repository contains a main component, `src`, which is a library for the computation of the numerical solution of the Eikonal equation described in the introduction paragraph. The library contains:
- `Mesh` which is a class that represents a mesh in 3D.
- `LocalSolver` which is a class responsable for the resolution of the local problem.
- `Solver` which is the implementation of the CUDA solver.
For more details, always refer to the documentation.

 The repo also contains an utility component, `test`, which contains a test case and some input meshes.

## Usage
After cloning the repo with the command `git clone https://github.com/AMSC22-23/Eikonal-Cuda-Cesaroni-Tonarelli-Trabacchin.git`, 
the installation of [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) and [GKlib](https://github.com/KarypisLab/GKlib) software is required. To install them, access the repo and run the following:
```bash
$ ./install_dependences.sh
```

Then, to build the executable, from the root folder run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
An executable for the test will be created into `/build`, and can be executed through:
```bash
$ ./eikonal input-filename num-partitions output-filename
```

where:
- `input-filename` is the input file path where the mesh will be retrieved. The program only accepts file in vtk format.
- `num-partitions` is the number of partitions dividing the domain.
- `output-filename` is the name of the output file. The file will be located in the folder `test/meshes/output_meshes`.

However, these are only examples. To fully exploit our library, it should be directly used in code to access further 
features, such as the possibility to modify the velocity matrix and the boundary conditions (which in our example are 
defaulted respectively to the identity matrix and the vertex nearest to the origin).

We provide test meshes at this [link](https://drive.google.com/drive/folders/12RzhUeLXBtaghX2UrWTKeBanNaRocNOn?usp=sharing).<br> 
One example is:
```bash
$ ./eikoanl ../test/meshes/input_meshes/cube-5.vtk 4 output-cube5
```
will execute the algorithm on a cubic test model and will save the output into the file `output-cube5`.

## Results
Performance analysis and results can be found in the documentation.

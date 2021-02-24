<!-- readme based on template: https://github.com/othneildrew/Best-README-Template -->

<!-- PROJECT LOGO -->
<p align="center">
  <h1 align="center">shared-memory-sgd</h1>

  <p align="center">
    C++ framework for implementing shared-memory parallel SGD for Deep Neural Network training
    <br />
    <br />
    <a href="https://github.com/dcs-chalmers/shared-memory-sgd/issues">Report Bug</a>
    ·
    <a href="https://github.com/dcs-chalmers/shared-memory-sgd/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#input">Input</a></li>
        <li><a href="#output">Ouput</a></li>
        <li><a href="#examples">Examples</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Framework for implementing parallel shared-memory Artificial Neural Network (ANN) training in C++ with SGD, supporting various synchronization mechanisms degrees of consistency. The code builds upon the <a href="https://github.com/yixuan/MiniDNN">MiniDNN</a> implementation, and relies on Eigen and OpenMP. In particular, the project includes the implementation of **LEASHED** which guarantees consistency and lock-freedom.

For technical details of **LEASHED** please see the original paper:

> Bäckström, K., Walulya, I., Papatriantafilou, M., & Tsigas, P. (2021, February). *Consistent Lock-free Parallel Stochastic Gradient Descent for Fast and Stable Convergence*. In Proceedings of the 35th IEEE International Parallel & Distributed Processing Symposium *(to appear)*. <a href="https://arxiv.org/abs/2102.09032">Full version</a>.

The following shared-memory parallel SGD methods are implemented:
* Lock-based consistent asynchronous SGD
* LEASHED - Lock-free implementation of consistent asynchronous SGD
* Hogwild! - Lock-free asynchronous SGD without consistency
* Synchronous parallel SGD



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/dcs-chalmers/shared-memory-sgd.git
   ```
2. Build project
   ```sh
   bash build.sh
   ```
3. Compile
   ```sh
   bash compile.sh
   ```



<!-- USAGE EXAMPLES -->
## Usage

### Input

Arguments and options - reference list:

Flag | Meaning | Values
--- | --- | ---
`a` | *algorithm* | ['ASYNC', 'HOG', 'LSH', 'SYNC']
`n` | *n.o. threads* | Integer
`A` | *architecture* | ['MLP', 'CNN']
`L` | *n.o. hidden layers* | Integer (applies for MLP only)
`U` | *n.o. hidden neurons per layer* | Integer (applies for MLP only)
`B` | *persistence bound* | Integer (applies for LEASHED only)
`e` | *n.o. epochs* | Integer
`r` | *n.o. rounds per epochs* | Integer
`b` | *mini-batch size* | Integer
`l` | *Step size* | Float

to see all options:
 ```sh
 ./cmake-build/debug/mininn --help
 ```

### Output

Output is a JSON object containing the following data:

Field | Meaning
--- | ---
`epoch_loss` | *list of loss values corresponding to each epoch*
`epoch_time` | *wall-clock time measure upon completing corresponding epoch*
`staleness_dist` | *distribution of staleness*
`numtriesdist` | *distribution of n.o. CAS attempts (applies to LSH only)*

### Examples

Multi-layer perceptron (MLP) training for 5 epochs batch size 512 and step size 0.005 with 8 threads using LEASHED-SGD:
 ```sh
 ./cmake-build-debug/mininn -a LSH -n 8 -A MLP -L 3 -U 128 -e 5 -r 469 -b 512 -l 0.005
 ```

Multi-layer perceptron (MLP) training with 8 threads using Hogwild!:
 ```sh
 ./cmake-build-debug/mininn -a HOG -n 8 -A MLP -L 3 -U 128 -e 5 -r 469 -b 512 -l 0.005
 ```

Convolutional neural network (CNN) training with 8 threads using LEASHED-SGD:
 ```sh
 ./cmake-build-debug/mininn -a LSH -n 8 -A CNN -e 5 -r 469 -b 512 -l 0.005
 ```



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/dcs-chalmers/shared-memory-sgd/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the AGPL-3.0 License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Karl Bäckström - bakarl@chalmers.se

Project Link: [https://github.com/dcs-chalmers/shared-memory-sgd](https://github.com/dcs-chalmers/shared-memory-sgd)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

A big thanks to the Wallenberg AI, Autonomous Systems and Software Program (WASP) for funding this work.

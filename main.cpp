#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <sys/time.h>
#include <type_traits>

#include <MiniDNN.h>
#include <mnist.h>

#include <ParameterContainer.h>

#include <NetworkExecutor.h>

#include "cifar10_reader.hpp"

using namespace MiniDNN;

int rand_seed = 1337;
double learning_rate;
Scalar momentum_mu;
int num_threads, batch_size, num_epochs, run_n;
int num_hidden_layers = 1;
int num_hidden_units = 128;
int rounds_per_epoch = -1;
int cas_backoff = 200;
bool check_concurrent_updates = 0;

enum class ALGORITHM {
    ASYNC, HOG, LSH, SEQ, SYNC
};
std::vector<std::string> AlgoTypes = {
        "ASYNC",
        "HOG",
        "LSH",
        "SEQ",
        "SYNC"
};
ALGORITHM run_algo = ALGORITHM::SEQ;

enum class ARCHITECTURE {
    MLP, CNN, LENET
};
std::vector<std::string> ArchTypes = {
        "MLP",
        "CNN",
        "LENET"
};
ARCHITECTURE use_arch = ARCHITECTURE::MLP;

std::string tauadaptstrat = "NONE";

std::string use_dataset = "MNIST";

template<typename T>
std::ostream &operator<<(typename std::enable_if<std::is_enum<T>::value, std::ostream>::type &stream, const T &e) {
    return stream << static_cast<typename std::underlying_type<T>::type>(e);
}

int main(int argc, char *argv[]) {

    struct option long_options[] = {
            // These options don't set a flag
            {"help",                no_argument,       nullptr, 'h'},
            {"algo",                required_argument, nullptr, 'a'},
            {"arch",                required_argument, nullptr, 'A'},
            {"epochs",              required_argument, nullptr, 'e'},
            {"num-threads",         required_argument, nullptr, 'n'},
            {"print-vals",          required_argument, nullptr, 'v'},
            {nullptr, 0,                                  nullptr, 0}
    };

    if (argc == 1) {
        printf("Use -h or --help for help\n");
        exit(0);
    }
    int i, c;

    unsigned long arch_id;

    while (1) {
        i = 0;
        c = getopt_long(argc, argv, "a:b:e:n:r:l:m:B:R:C:A:L:U:t:D:", long_options, &i);

        if (c == -1) {
            //printf("Use -h or --help for help\n");
            //exit(0);
            break;
        }

        if (c == 0 && long_options[i].flag == 0)
            c = long_options[i].val;

        unsigned long algo_id = AlgoTypes.size() - 1;
        std::string algo_name;
        arch_id = ArchTypes.size() - 1;
        std::string arch_name;

        switch (c) {
            case 0:
                /* Flag is automatically set */
                break;
            case 'h':
                printf("Parallel SGD shared Memory Benchmarks"
                       "\n"
                       "\n"
                       "Usage:\n"
                       "  %s [options...]\n"
                       "\n"
                       "Options:\n"
                       "  -h, --help\n"
                       "        Print this message\n"
                       "  -D, --data-set <int>\n"
                       "        Data set {MNIST, FASHION-MNIST, CIFAR10}\n"
                       "  -b, --batch-size <int>\n"
                       "        Batch size\n"
                       "  -e, --epochs <int>\n"
                       "        Number of epochs\n"
                       "  -r, <int>\n"
                       "        Number of rounds per epochs\n"
                       "  -n, --num-threads <int>\n"
                       "        Number of threads\n"
                       "  -a, --algorithm <string>\n"
                       "        {SEQ, SYNC, ASYNC, HOG, LSH}\n"
                       "  -A, --architecture <string>\n"
                       "        {MLP, CNN, LENET}\n"
                       "  -L, <int>\n"
                       "        Number of hidden layers\n"
                       "  -U, <int>\n"
                       "        Number of units per hidden layer\n"
                       "  -l, <float>\n"
                       "        Learning rate\n"
                       "  -B, <int>\n"
                       "        CAS backoff threshold, max n.o. failed CAS per step\n"
                       "  -C, <bool>\n"
                       "        In LSH, check for concurrent updates in retry loop\n"
                       "  -t, <bool>\n"
                       "        Staleness-adaptive step size strategy\n"
                       "  -v, --print-vals <int>\n"
                       "        Print debug informations\n", argv[0]);
                exit(0);
            case 'a':
                algo_name = optarg;
                while (algo_id >= 0) {
                    if (algo_name == AlgoTypes[algo_id]) {
                        break;
                    }
                    algo_id--;
                }
                run_algo = static_cast<ALGORITHM>(algo_id);
                break;
            case 'A':
                arch_name = optarg;
                while (arch_id >= 0) {
                    if (arch_name == ArchTypes[arch_id]) {
                        break;
                    }
                    arch_id--;
                }
                use_arch = static_cast<ARCHITECTURE>(arch_id);
                break;
            case 'b':
                batch_size = atoi(optarg);
                break;
            case 'e':
                num_epochs = atoi(optarg);
                break;
            case 'r':
                rounds_per_epoch = atoi(optarg);
                break;
            case 'B':
                cas_backoff = atoi(optarg);
                break;
            case 'n':
                num_threads = atoi(optarg);
                break;
            case 'R':
                run_n = atoi(optarg);
                break;
            case 'l':
                learning_rate = atof(optarg);
                break;
            case 'C':
                check_concurrent_updates = atoi(optarg);
                break;
            case 't':
                tauadaptstrat = optarg;
                break;
            case 'D':
                use_dataset = optarg;
                break;
            case 'L':
                num_hidden_layers = atoi(optarg);
                break;
            case 'U':
                num_hidden_units = atoi(optarg);
                break;
            case 'm':
                momentum_mu = atof(optarg);
                break;
            case '?':
            default:
                printf("Use -h or --help for help\n");
                exit(1);
        }
    }


    // data

    Matrix x;
    Matrix y;

    int in_dim_x;
    int in_dim_y;
    int in_no_chs;

    if (use_dataset == "CIFAR10") {

        in_dim_x = 32;
        in_dim_y = 32;
        in_no_chs = 3;

        auto DATASET = cifar::read_dataset<std::vector, std::vector, double, double>();

        long n_train = DATASET.training_images.size(); // 50K
        long dim_in = DATASET.training_images[0].size(); // 3072

        typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;

        x = Matrix::Zero(dim_in, n_train);

        for (int i = 0; i < n_train; i++)
            x.col(i) = Vector::Map(&DATASET.training_images[i][0], DATASET.training_images[i].size());

        x /= 255; // normalize

        y = Matrix::Zero(10, n_train);

        int T;
        for (int i = 0; i < n_train; i++){
            T = DATASET.training_labels[i];
            if (T < 10 && T >= 0)
                y(T, i) = 1;//Vector::Map(&DATASET.training_labels[i][0], DATASET.training_labels[i].size());
            else
                std::cout << "Label value error: " << T << std::endl;
        }
    } else {

        in_dim_x = 28;
        in_dim_y = 28;
        in_no_chs = 1;

        std::string data_dir;
        if (use_dataset == "MNIST") {
            data_dir = "./data/mnist/";
        } else if (use_dataset == "FASHION-MNIST") {
            data_dir = "./data/fashion-mnist/";
        }
        MNIST dataset(data_dir);
        dataset.read();

        x = dataset.train_data;
        y = dataset.train_labels;
    }

    // Construct a network object

    auto *init_param = new ParameterContainer(); // variable parameter container pointer

    NetworkTopology network(init_param);

    if (use_arch == ARCHITECTURE::MLP) {
        network.add_layer(new FullyConnected<ReLU>(in_dim_x * in_dim_y * in_no_chs, num_hidden_units));
        for (int i = 0; i < num_hidden_layers - 1; i++) {
            network.add_layer(new FullyConnected<ReLU>(num_hidden_units, num_hidden_units));
        }
        network.add_layer(new FullyConnected<Softmax>(num_hidden_units, 10));
    } else if (use_arch == ARCHITECTURE::CNN) {

        network.add_layer(new Convolutional<ReLU>(28, 28, 1, 4, 3, 3));
        network.add_layer(new MaxPooling<ReLU>(26, 26, 4, 2, 2));

        network.add_layer(new Convolutional<ReLU>(13, 13, 4, 8, 3, 3));
        network.add_layer(new MaxPooling<ReLU>(11, 11, 8, 2, 2));

        network.add_layer(new FullyConnected<ReLU>(5 * 5 * 8, 128));
        network.add_layer(new FullyConnected<Softmax>(128, 10));

    } else if (use_arch == ARCHITECTURE::LENET) {

        if (use_dataset == "CIFAR10") {

            network.add_layer(new Convolutional<ReLU>(32, 32, 3, 6, 5, 5));
            network.add_layer(new MaxPooling<ReLU>(28, 28, 6, 2, 2));

            network.add_layer(new Convolutional<ReLU>(14, 14, 6, 16, 5, 5));
            network.add_layer(new MaxPooling<ReLU>(10, 10, 16, 2, 2));

            network.add_layer(new FullyConnected<ReLU>(5 * 5 * 16, 120));

        } else if (use_dataset == "MNIST" || use_dataset == "FASHION-MNIST") {

            network.add_layer(new Convolutional<ReLU>(28, 28, 1, 6, 5, 5));
            network.add_layer(new MaxPooling<ReLU>(24, 24, 6, 2, 2));

            network.add_layer(new Convolutional<ReLU>(12, 12, 6, 16, 5, 5));
            network.add_layer(new MaxPooling<ReLU>(8, 8, 16, 2, 2));

            network.add_layer(new FullyConnected<ReLU>(4 * 4 * 16, 120));

        }

        network.add_layer(new FullyConnected<Softmax>(120, 10));

    }

    // Set output layer
    network.set_output(new MultiClassEntropy());

    // (Optional) set callback function object
    VerboseCallback callback;
    network.set_callback(callback);

    // Initialize parameters with N(0, 0.01^2) using random seed 123
    network.init(0, 0.01, rand_seed);

    // Create optimizer object
    auto *opt = new SGDM();
    opt->initialize_state(init_param->param_list.size());
    opt->m_lrate = learning_rate;
    opt->mu = momentum_mu;

    std::vector<Optimizer *> thread_local_opts(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        thread_local_opts[i] = opt->clone();
    }

    int algorithm_id = static_cast<int>(run_algo);
    int architecture_id = static_cast<int>(use_arch);
    NetworkExecutor<Matrix, Matrix> executor(&network, opt, thread_local_opts, x, y, tauadaptstrat, num_threads, learning_rate, algorithm_id, architecture_id);


    struct timeval start, end;
    gettimeofday(&start, nullptr);

    switch (run_algo) {
        case ALGORITHM::SEQ:
            if (num_threads > 1)
                std::cout << "WARNING: using only 1 thread for sequential optimization" << std::endl;
            executor.run_training(batch_size, num_epochs, rounds_per_epoch, rand_seed);
            break;
        case ALGORITHM::SYNC:
            executor.run_parallel_sync(batch_size, num_epochs, rounds_per_epoch, rand_seed);
            break;
        case ALGORITHM::ASYNC:
            executor.run_parallel_async(batch_size, num_epochs, rounds_per_epoch, rand_seed, true);
            break;
        case ALGORITHM::HOG:
            executor.run_parallel_async(batch_size, num_epochs, rounds_per_epoch, rand_seed);
            break;
        case ALGORITHM::LSH:
            executor.run_parallel_leashed(batch_size, num_epochs, rounds_per_epoch, cas_backoff, check_concurrent_updates, rand_seed);
            break;
        default:
            printf("Use -h or --help for help\n");
            exit(1);
            break;
    }

    gettimeofday(&end, nullptr);
    time_t duration;
    duration = end.tv_sec - start.tv_sec;

    Scalar epoch_loss = executor.get_loss();

    i = 0;
    auto epoch_losses = executor.get_losses_per_epoch();
    auto epoch_times = executor.get_times_per_epoch();
    auto tau_dist = executor.get_tau_dist();
    auto num_tries_dist = executor.get_num_tries_dist();

    std::cout << "{";

    std::cout << "\"epoch_loss\": [ ";

    bool frst = true;
    for (float l : epoch_losses) {
        if (!frst) {
            std::cout << ", ";
        }
        frst = false;
        std::cout << l;
    }
    std::cout << " ], ";

    std::cout << "\"epoch_time\": [ ";

    frst = true;
    for (auto t : epoch_times) {
        if (!frst) {
            std::cout << ", ";
        }
        frst = false;
        std::cout << (t - start.tv_sec);
    }
    std::cout << " ], ";

    std::cout << "\"staleness_dist\": [ ";

    frst = true;
    for (long tau_count : tau_dist) {
        if (!frst) {
            std::cout << ", ";
        }
        frst = false;
        std::cout << tau_count;
    }
    std::cout << " ], ";

    std::cout << "\"numtriesdist\": [ ";

    frst = true;
    for (long num_tries_count : num_tries_dist) {
        if (!frst) {
            std::cout << ", ";
        }
        frst = false;
        std::cout << num_tries_count;
    }
    std::cout << " ], ";

    std::cout << "\"numfailedcas\": ";
    std::cout << executor.failed_cas;

    std::cout << "}";

    return 0;
}

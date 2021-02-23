#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <random>
#include <sched.h>
#include <inttypes.h>
#include <sys/time.h>
#include <unistd.h>
#include "Config.h"

#define BOOL_CAS(addr, old_val, new_val) __sync_bool_compare_and_swap(addr, old_val, new_val)

typedef Eigen::Matrix<MiniDNN::Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<MiniDNN::Scalar, Eigen::Dynamic, 1> Vector;
typedef Eigen::Array<MiniDNN::Scalar, 1, Eigen::Dynamic> RowVector;

static std::default_random_engine generator;

// Normal distribution: N(mu, sigma^2)
inline void set_normal_random(MiniDNN::Scalar *arr, int n, MiniDNN::Scalar mu, MiniDNN::Scalar sigma) {
    std::normal_distribution<MiniDNN::Scalar> distribution(mu, sigma);
    for (int i = 0; i < n; i++) {
        arr[i] = distribution(generator);
    }
}

// shuffle cols of matrix
inline void shuffle_data(Matrix &data, Matrix &labels) {
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(data.cols());
    perm.setIdentity();
    std::random_shuffle(perm.indices().data(), perm.indices().data()
                                               + perm.indices().size());
    data = data * perm;  // permute columns
    labels = labels * perm;
}

// encode discrete values to one-hot values
inline Matrix one_hot_encode(const Matrix &y, int n_value) {
    int n = y.cols();
    Matrix y_onehot = Matrix::Zero(n_value, n);
    for (int i = 0; i < n; i++) {
        y_onehot(int(y(i)), i) = 1;
    }
    return y_onehot;
}

// classification accuracy
inline MiniDNN::Scalar compute_accuracy(const Matrix &preditions, const Matrix &labels) {
    int n = preditions.cols();
    MiniDNN::Scalar acc = 0;
    for (int i = 0; i < n; i++) {
        Matrix::Index max_index;
        MiniDNN::Scalar max_value = preditions.col(i).maxCoeff(&max_index);
        acc += int(max_index) == labels(i);
    }
    return acc / n;
}

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
typedef uint64_t ticks;

#if defined(__i386__)
static inline ticks
getticks(void) 
{
  ticks ret;

  __asm__ __volatile__("rdtsc" : "=A" (ret));
  return ret;
}
#elif defined(__x86_64__)
static inline ticks
getticks(void) {
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long) lo) | (((unsigned long long) hi) << 32);
}
#elif defined(__sparc__)
static inline ticks
getticks()
{
  ticks ret = 0;
  __asm__ __volatile__ ("rd %%tick, %0" : "=r" (ret) : "0" (ret)); 
  return ret;
}
#elif defined(__tile__)
#  include <arch/cycle.h>
static inline ticks
getticks()
{
  return get_cycle_count();
}
#endif

#ifdef __cplusplus
}
#endif


static inline double wtime(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double) t.tv_sec + ((double) t.tv_usec) / 1000000.0;
}

static inline
void set_cpu(int cpu) {
#ifndef NO_SET_CPU
#  ifdef __sparc__
    processor_bind(P_LWPID,P_MYID, the_cores[cpu], NULL);
#  elif defined(__tile__)
    if (cpu>=tmc_cpus_grid_total()) {
      perror("Thread id too high");
    }
    // cput_set_t cpus;
    if (tmc_cpus_set_my_cpu(the_cores[cpu])<0) {
      tmc_task_die("tmc_cpus_set_my_cpu() failed."); 
    }
#  else

    int n_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    //    cpu %= (NUMBER_OF_SOCKETS * CORES_PER_SOCKET);
    if (cpu < n_cpus) {
        int cpu_use = cpu;
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu_use, &mask);
#    if defined(PLATFORM_NUMA)
        numa_set_preferred(get_cluster(cpu_use));
#    endif
        pthread_t thread = pthread_self();
        if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &mask) != 0) {
            fprintf(stderr, "Error setting thread affinity\n");
        }
    }
#  endif
#endif
}

#endif  // SRC_UTILS_H_

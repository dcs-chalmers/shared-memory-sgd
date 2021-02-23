#ifndef OPTIMIZER_SGD_H_
#define OPTIMIZER_SGD_H_

#include <Eigen/Core>
#include "../Config.h"
#include "../Optimizer.h"

namespace MiniDNN {


///
/// \ingroup Optimizers
///
/// The Stochastic Gradient Descent (SGD) algorithm
///
    class SGD : public Optimizer {
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
        typedef Vector::AlignedMapType AlignedMapVec;

    public:
        Scalar m_lrate;
        Scalar m_decay;

        SGD() :  m_lrate(Scalar(0.01)), m_decay(Scalar(0)) {}

        SGD(SGD &) = default;

        SGD *clone() override {
            return new SGD(*this);
        }

        void update(ConstAlignedMapVec &dvec, AlignedMapVec &vec) {
            vec.noalias() -= m_lrate * (dvec + m_decay * vec);
        }

        void initialize_state(int num_layers) {}

        void initialize_layer_state(int layer_n, int m_dw_size, int m_db_size) {}

        Scalar get_w_delta(int layer_n, int i, Scalar m_dw_i) {
            Scalar delta_i = (- m_lrate * m_dw_i);
            if (layer_n == 0 && i == 0) {
                //std::cout << "this update: " << delta_i << std::endl;
            }
            return delta_i;
        }

        Scalar get_b_delta(int layer_n, int i, Scalar m_db_i) {
            Scalar delta_i = (- m_lrate * m_db_i);
            return delta_i;
        }
    };


} // namespace MiniDNN


#endif /* OPTIMIZER_SGD_H_ */

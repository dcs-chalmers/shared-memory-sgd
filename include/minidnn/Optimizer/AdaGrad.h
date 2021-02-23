#ifndef OPTIMIZER_ADAGRAD_H_
#define OPTIMIZER_ADAGRAD_H_

#include <Eigen/Core>
#include "../Config.h"
#include "../Optimizer.h"
#include "../Utils/sparsepp.h"

namespace MiniDNN {


///
/// \ingroup Optimizers
///
/// The AdaGrad algorithm
///
    class AdaGrad : public Optimizer {
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;
        typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
        typedef Vector::AlignedMapType AlignedMapVec;

        spp::sparse_hash_map<const Scalar *, Array> m_history;

    public:
        Scalar m_lrate;
        Scalar m_eps;

        AdaGrad() :
                m_lrate(Scalar(0.01)), m_eps(Scalar(1e-7)) {}

        AdaGrad(AdaGrad &) = default;

        AdaGrad *clone() override {
            return new AdaGrad(*this);
        }

        void reset() {
            m_history.clear();
        }

        void initialize_state(int num_layers) {}

        void initialize_layer_state(int layer_n, int m_dw_size, int m_db_size) {}

        void update(ConstAlignedMapVec &dvec, AlignedMapVec &vec) {
            // Get the accumulated squared gradient associated with this gradient
            Array &grad_square = m_history[dvec.data()];

            // If length is zero, initialize it
            if (grad_square.size() == 0) {
                grad_square.resize(dvec.size());
                grad_square.setZero();
            }

            // Update accumulated squared gradient
            grad_square += dvec.array().square();
            // Update parameters
            vec.array() -= m_lrate * dvec.array() / (grad_square.sqrt() + m_eps);
        }

        Scalar get_w_delta(int layer_n, int i, Scalar m_dw_i) {
            return - m_lrate * m_dw_i;
        }

        Scalar get_b_delta(int layer_n, int i, Scalar m_db_i) {
            return - m_lrate * m_db_i;
        }
    };


} // namespace MiniDNN


#endif /* OPTIMIZER_ADAGRAD_H_ */

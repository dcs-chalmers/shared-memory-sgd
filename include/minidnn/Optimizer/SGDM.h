#ifndef OPTIMIZER_SGDM_H_
#define OPTIMIZER_SGDM_H_

#include <Eigen/Core>
#include "../Config.h"
#include "../Optimizer.h"

namespace MiniDNN {


///
/// \ingroup Optimizers
///
/// The Stochastic Gradient Descent (SGD) algorithm
///
    class SGDM : public Optimizer {
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    public:
        Scalar m_lrate;
        Scalar mu;
        std::vector<Vector> prev_delta_w;
        std::vector<Vector> prev_delta_b;

        SGDM() :  m_lrate(Scalar(0.01)), mu(Scalar(0.9)) {}

        SGDM(SGDM &) = default;

        SGDM *clone() override {
            return new SGDM(*this);
        }

        void update(ConstAlignedMapVec &dvec, AlignedMapVec &vec) {}

        void initialize_state(int num_layers) {
            prev_delta_w.resize(num_layers);
            prev_delta_b.resize(num_layers);
        }

        void initialize_layer_state(int layer_n, int m_dw_size, int m_db_size) {
            if (prev_delta_w[layer_n].size() == 0) {
                prev_delta_w[layer_n].resize(m_dw_size);
                prev_delta_w[layer_n].setZero();
            }
            if (prev_delta_b[layer_n].size() == 0) {
                prev_delta_b[layer_n].resize(m_db_size);
                prev_delta_b[layer_n].setZero();
            }
        }

        Scalar get_w_delta(int layer_n, int i, Scalar m_dw_i) {
            Scalar delta_i = (- step_scale_factor * m_lrate * m_dw_i) + mu * prev_delta_w[layer_n](i);
            if (layer_n == 0 && i == 0) {
                //std::cout << "prev update: " << prev_delta_w[layer_n](i) << std::endl;
                //std::cout << "this update: " << delta_i << std::endl;
            }
            prev_delta_w[layer_n](i) = delta_i;
            return delta_i;
        }

        Scalar get_b_delta(int layer_n, int i, Scalar m_db_i) {
            Scalar delta_i = (- step_scale_factor * m_lrate * m_db_i) + mu * prev_delta_b[layer_n](i);
            prev_delta_b[layer_n](i) = delta_i;
            return delta_i;
        }
    };


} // namespace MiniDNN


#endif /* OPTIMIZER_SGDM_H_ */

#ifndef FULLYCONNECTEDPARAMETERS_H_
#define FULLYCONNECTEDPARAMETERS_H_

#include <Eigen/Core>
#include <vector>
#include "Config.h"
#include "RNG.h"
#include "Optimizer.h"
#include "Parameters.h"

namespace MiniDNN {

    class FullyConnectedParameters : public Parameters {
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    public:

        Matrix m_weight;  // Weight parameters, W(in_size x out_size)
        Vector m_bias;    // Bias parameters, b(out_size x 1)

        ///
        /// Virtual destructor
        ///
        virtual ~FullyConnectedParameters() = default;

        void update_cw(Matrix m_dw, Vector m_db, Optimizer *opt) {
            opt->initialize_layer_state(layer_n, m_dw.size(), m_db.size());
            for (int i = 0; i < m_weight.size() ; i++) {  // for each column, update value at each row
                m_weight(i) += opt->get_w_delta(layer_n, i, m_dw(i));
            }
            for (int i = 0; i < m_bias.size() ; i++) {
                m_bias(i) += opt->get_b_delta(layer_n, i, m_db(i));
            }
        }

        ///
        /// Constructor
        ///
        FullyConnectedParameters() = default;

        FullyConnectedParameters(FullyConnectedParameters &) = default;

        FullyConnectedParameters *clone() override {
            return new FullyConnectedParameters(*this);
        }

    };


} // namespace MiniDNN


#endif /* FULLYCONNECTEDPARAMETERS_H_ */

#ifndef CONVOLUTIONALPARAMETERS_H_
#define CONVOLUTIONALPARAMETERS_H_

#include <Eigen/Core>
#include <vector>
#include "Config.h"
#include "RNG.h"
#include "Optimizer.h"
#include "Parameters.h"

namespace MiniDNN {

    class ConvolutionalParameters : public Parameters {
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    public:

        Vector m_filter_data; // Filter parameters. Total length is (in_channels x out_channels x filter_rows x filter_cols)
        Vector m_bias;        // Bias term for the output channels, out_channels x 1. (One bias term per channel)

        ///
        /// Constructor
        ///
        ConvolutionalParameters() = default;

        void update_cw(Matrix m_df_data, Vector m_db, Optimizer *opt) {
            opt->initialize_layer_state(layer_n, m_df_data.size(), m_db.size());
            for (int i = 0; i < m_filter_data.size() ; i++) {
                m_filter_data(i) += opt->get_w_delta(layer_n, i, m_df_data(i));
            }
            for (int i = 0; i < m_bias.size() ; i++) {
                m_bias(i) += opt->get_b_delta(layer_n, i, m_db(i));
            }
        }

        ///
        /// Virtual destructor
        ///
        virtual ~ConvolutionalParameters() = default;

        ConvolutionalParameters(ConvolutionalParameters &) = default;

        ConvolutionalParameters *clone() override {
            return new ConvolutionalParameters(*this);
        }

    };


} // namespace MiniDNN


#endif /* CONVOLUTIONALPARAMETERS_H_ */

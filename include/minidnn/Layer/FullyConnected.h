#ifndef LAYER_FULLYCONNECTEDSGD_H_
#define LAYER_FULLYCONNECTEDSGD_H_

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "../Config.h"
#include "../Layer.h"
#include "../Utils/Random.h"
#include "../Parameters/FullyConnectedParameters.h"

namespace MiniDNN {


///
/// \ingroup Layers
///
/// Fully connected hidden layer
///
    template<typename Activation>
    class FullyConnected : public Layer {
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
        typedef Vector::AlignedMapType AlignedMapVec;

        Parameters *fully_connected_initial_parameters_pointer;
        FullyConnectedParameters *current_parameters = nullptr;


        Matrix m_dw;      // Derivative of weights
        Vector m_db;      // Derivative of bias
        Matrix m_z;       // Linear term, z = W' * in + b
        Matrix m_a;       // Output of this layer, a = act(z)
        Matrix m_din;     // Derivative of the input of this layer.
        // Note that input of this layer is also the output of previous layer

    public:
        ///
        /// Constructor
        ///
        /// \param in_size  Number of input units.
        /// \param out_size Number of output units.
        ///
        FullyConnected(const int in_size, const int out_size) :
                Layer(in_size, out_size) {}

        FullyConnected(FullyConnected &) = default;

        void init(const Scalar &mu, const Scalar &sigma, RNG &rng) {
            layer_initial_parameters_ptr = new FullyConnectedParameters();
            FullyConnectedParameters *fully_connected_initial_parameters_pointer = dynamic_cast<FullyConnectedParameters *>(layer_initial_parameters_ptr);

            //fully_connected_initial_parameters_pointer->m_weight = new Matrix();
            fully_connected_initial_parameters_pointer->m_weight.resize(this->m_in_size, this->m_out_size);
            //fully_connected_initial_parameters_pointer->m_bias = new Vector();
            fully_connected_initial_parameters_pointer->m_bias.resize(this->m_out_size);
            m_dw.resize(this->m_in_size, this->m_out_size);
            m_db.resize(this->m_out_size);
            // Set random coefficients
            internal::set_normal_random(fully_connected_initial_parameters_pointer->m_weight.data(),
                                        fully_connected_initial_parameters_pointer->m_weight.size(), rng, mu, sigma);
            internal::set_normal_random(fully_connected_initial_parameters_pointer->m_bias.data(),
                                        fully_connected_initial_parameters_pointer->m_bias.size(), rng, mu, sigma);
        }

        void set_param_ptr(ParameterContainer *current_param_cont_ptr) {
            Parameters *par = current_param_cont_ptr->param_list[layer_n];
            current_parameters = dynamic_cast<FullyConnectedParameters *>(par);
        }

        // prev_layer_data: in_size x nobs
        void forward(const Matrix &prev_layer_data) {
            const int nobs = prev_layer_data.cols();
            // Linear term z = W' * in + b
            m_z.resize(this->m_out_size, nobs);
            m_z.noalias() = current_parameters->m_weight.transpose() * prev_layer_data;
            m_z.colwise() += (current_parameters->m_bias);
            // Apply activation function
            m_a.resize(this->m_out_size, nobs);
            Activation::activate(m_z, m_a);
        }

        const Matrix &output() const {
            return m_a;
        }

        // prev_layer_data: in_size x nobs
        // next_layer_data: out_size x nobs
        void backprop(const Matrix &prev_layer_data, const Matrix &next_layer_data) {
            const int nobs = prev_layer_data.cols();
            // After forward stage, m_z contains z = W' * in + b
            // Now we need to calculate d(L) / d(z) = [d(a) / d(z)] * [d(L) / d(a)]
            // d(L) / d(a) is computed in the next layer, contained in next_layer_data
            // The Jacobian matrix J = d(a) / d(z) is determined by the activation function
            Matrix &dLz = m_z;
            Activation::apply_jacobian(m_z, m_a, next_layer_data, dLz);
            // Now dLz contains d(L) / d(z)
            // Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
            m_dw.noalias() = prev_layer_data * dLz.transpose() / nobs;
            // Derivative for bias, d(L) / d(b) = d(L) / d(z)
            m_db.noalias() = dLz.rowwise().mean();
            // Compute d(L) / d_in = W * [d(L) / d(z)]
            m_din.resize(this->m_in_size, nobs);
            m_din.noalias() = (current_parameters->m_weight) * dLz;
        }

        const Matrix &backprop_data() const {
            return m_din;
        }

        void update(Optimizer *opt) {
            ConstAlignedMapVec dw(m_dw.data(), m_dw.size());
            ConstAlignedMapVec db(m_db.data(), m_db.size());
            AlignedMapVec w(current_parameters->m_weight.data(), current_parameters->m_weight.size());
            AlignedMapVec b(current_parameters->m_bias.data(), current_parameters->m_bias.size());
            opt->update(dw, w);
            opt->update(db, b);
        }

        void update_cw(Optimizer *opt) {
            current_parameters->update_cw(m_dw, m_db, opt);
        }

        std::vector<Scalar> get_parameters() const {
            std::vector<Scalar> res(current_parameters->m_weight.size() + current_parameters->m_bias.size());
            // Copy the data of weights and bias to a long vector
            std::copy(current_parameters->m_weight.data(),
                      current_parameters->m_weight.data() + current_parameters->m_weight.size(), res.begin());
            std::copy(current_parameters->m_bias.data(),
                      current_parameters->m_bias.data() + current_parameters->m_bias.size(),
                      res.begin() + current_parameters->m_weight.size());
            return res;
        }

        void set_parameters(const std::vector<Scalar> &param) {

            if (static_cast<int>(param.size()) !=
                current_parameters->m_weight.size() + current_parameters->m_bias.size()) {
                throw std::invalid_argument("Parameter size does not match");
            }

            std::copy(param.begin(), param.begin() + current_parameters->m_weight.size(),
                      current_parameters->m_weight.data());
            std::copy(param.begin() + current_parameters->m_weight.size(), param.end(),
                      current_parameters->m_bias.data());
        }

        std::vector<Scalar> get_derivatives() const {
            std::vector<Scalar> res(m_dw.size() + m_db.size());
            // Copy the data of weights and bias to a long vector
            std::copy(m_dw.data(), m_dw.data() + m_dw.size(), res.begin());
            std::copy(m_db.data(), m_db.data() + m_db.size(), res.begin() + m_dw.size());
            return res;
        }


        Layer *clone() override {
            return new FullyConnected(*this);
        }

        void aggregate(const Layer &_other) {
            const FullyConnected &other = dynamic_cast<const FullyConnected &>(_other);
            m_dw += other.m_dw;
            m_db += other.m_db;
        }


        void copy(const Layer &_other) override {
            const FullyConnected &other = dynamic_cast<const FullyConnected &>(_other);
            current_parameters->m_weight = other.current_parameters->m_weight;
            current_parameters->m_bias = other.current_parameters->m_bias;
            m_z = other.m_z;
            m_a = other.m_a;
            m_din = other.m_din;
        }

        void reset() {
            m_dw.setZero();
            m_db.setZero();
        }

        void normalize_derivatives(float factor) override {
            m_dw /= factor;
            m_db /= factor;
        }
    };

} // namespace MiniDNN


#endif /* LAYER_FULLYCONNECTEDSGD_H_ */

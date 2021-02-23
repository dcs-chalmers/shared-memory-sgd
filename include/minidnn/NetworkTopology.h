#ifndef NETWORKTOPOLOGY_H_
#define NETWORKTOPOLOGY_H_

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include <unistd.h>
#include "Config.h"
#include "RNG.h"
#include "Layer.h"
#include "Output.h"
#include "Callback.h"
#include "Utils/Random.h"
#include "ParameterContainer.h"


namespace MiniDNN {

///
/// \defgroup Network Neural Network Model
///

///
/// \ingroup Network
///
/// This class represents a neural network model that typically consists of a
/// number of hidden layers and an output layer. It provides functions for
/// network building, model fitting, and prediction, etc.
///
    class NetworkTopology {
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::RowVectorXi IntegerVector;

        RNG m_default_rng;      // Built-in RNG
        RNG &m_rng;              // Reference to the RNG provided by the user,
        // otherwise reference to m_default_rng
        std::vector<Layer *> m_layers;           // Pointers to hidden layers
        Output *m_output;           // The output layer
        Callback m_default_callback; // Default callback function
        Callback *m_callback;         // Points to user-provided callback function,
        // otherwise points to m_default_callback

        Scalar loss = 0;
        int nlayer = 0;

        // Check dimensions of layers
        void check_unit_sizes() const {
            if (nlayer <= 1) {
                return;
            }

            for (int i = 1; i < nlayer; i++) {
                if (m_layers[i]->in_size() != m_layers[i - 1]->out_size()) {
                    throw std::invalid_argument("Unit sizes do not match");
                }
            }
        }

        void propagate_param_ptr() {
            for (int i = 0; i < nlayer; i++) {
                m_layers[i]->set_param_ptr(current_param_container_ptr);
            }
        }

    public:

        ParameterContainer **param_pointer;
        ParameterContainer *latest_pointer;
        ParameterContainer *current_param_container_ptr;

        NetworkTopology(ParameterContainer *_parameters) :
                m_default_rng(1),
                m_rng(m_default_rng),
                m_output(NULL),
                m_default_callback(),
                m_callback(&m_default_callback),
                param_pointer(0),
                latest_pointer(0),
                current_param_container_ptr{_parameters} {}

        ///
        /// Default constructor that creates an empty neural network
        ///
        NetworkTopology() :
                m_default_rng(1),
                m_rng(m_default_rng),
                m_output(NULL),
                m_default_callback(),
                m_callback(&m_default_callback) {}

        NetworkTopology(NetworkTopology &other) :
                m_default_rng(other.m_default_rng),
                m_rng(other.m_default_rng),
                m_callback(other.m_callback),
                current_param_container_ptr(other.current_param_container_ptr),
                param_pointer(other.param_pointer) {
            nlayer = other.nlayer;
            for (int i = 0; i < nlayer; ++i) {
                m_layers.push_back(other.m_layers[i]->clone());
            }

        }

        NetworkTopology &operator=(NetworkTopology &) = delete;

        void set_pointer(ParameterContainer *new_param_cont_ptr) {
            current_param_container_ptr = new_param_cont_ptr;
            propagate_param_ptr();
        }

        // Let each layer compute its output
        void forward(const Matrix &input) {
            if (nlayer <= 0) {
                return;
            }

            // First layer
            if (input.rows() != m_layers[0]->in_size()) {
                throw std::invalid_argument("Input data have incorrect dimension");
            }

            m_layers[0]->forward(input);

            // The following layers
            for (int i = 1; i < nlayer; i++) {
                m_layers[i]->forward(m_layers[i - 1]->output());
            }
        }

        // Let each layer compute its gradients of the parameters
        // target has two versions: Matrix and RowVectorXi
        // The RowVectorXi version is used in classification problems where each
        // element is a class label
        template<typename TargetType>
        void backprop(const Matrix &input, const TargetType &target) {
            if (nlayer <= 0) {
                return;
            }

            Layer *first_layer = m_layers[0];
            Layer *last_layer = m_layers[nlayer - 1];
            // Let output layer compute back-propagation data

            m_output->check_target_data(target);
            m_output->evaluate(last_layer->output(), target);

            // If there is only one hidden layer, "prev_layer_data" will be the input data
            if (nlayer == 1) {
                first_layer->backprop(input, m_output->backprop_data());
                return;
            }

            // Compute gradients for the last hidden layer
            last_layer->backprop(m_layers[nlayer - 2]->output(), m_output->backprop_data());

            // Compute gradients for all the hidden layers except for the first one and the last one
            for (int i = nlayer - 2; i > 0; i--) {
                m_layers[i]->backprop(m_layers[i - 1]->output(),
                                      m_layers[i + 1]->backprop_data());
            }

            // Compute gradients for the first layer
            first_layer->backprop(input, m_layers[1]->backprop_data());
        }

        // Update parameters
        void update(Optimizer *opt) {
            if (nlayer <= 0) {
                return;
            }
            current_param_container_ptr->timestamp++;
            for (int i = 0; i < nlayer; i++) {
                m_layers[i]->update(opt);
            }
        }

        // Update parameters
        void update_cw(Optimizer *opt) {
            if (nlayer <= 0) {
                return;
            }
            current_param_container_ptr->timestamp++;
            for (int i = 0; i < nlayer; i++) {
                if (param_pointer && latest_pointer && *param_pointer != latest_pointer)
                    return;
                m_layers[i]->update_cw(opt);
            }
        }


        ///
        /// Destructor that frees the added hidden layers and output layer
        ///
        ~NetworkTopology() {
            for (int i = 0; i < nlayer; i++) {
                delete m_layers[i];
                m_layers[i] = nullptr;
            }

            if (m_output) {
                delete m_output;
            }
        }


        ///
        /// Add a hidden layer to the neural network
        ///
        /// \param layer A pointer to a Layer object, typically constructed from
        ///              layer classes such as FullyConnected and Convolutional.
        ///              **NOTE**: the pointer will be handled and freed by the
        ///              network object, so do not delete it manually.
        ///
        void add_layer(Layer *layer) {
            m_layers.push_back(layer);
            ++nlayer;
        }

        ///
        /// Set the output layer of the neural network
        ///
        /// \param output A pointer to an Output object, typically constructed from
        ///               output layer classes such as RegressionMSE and MultiClassEntropy.
        ///               **NOTE**: the pointer will be handled and freed by the
        ///               network object, so do not delete it manually.
        ///
        void set_output(Output *output) {
            /*
            if (m_output)
            {
                delete m_output;
            }
            */
            m_output = output;
        }

        ///
        Output *get_output() {


            return m_output;
        }

        ///
        /// Set the callback function that can be called during model fitting
        ///
        /// \param callback A user-provided callback function object that inherits
        ///                 from the default Callback class.
        ///
        void set_callback(Callback &callback) {
            m_callback = &callback;
        }

        ///
        /// Set the default silent callback function
        ///
        void set_default_callback() {
            m_callback = &m_default_callback;
        }

        ///
        /// Initialize layer parameters in the network using normal distribution
        ///
        /// \param mu    Mean of the normal distribution.
        /// \param sigma Standard deviation of the normal distribution.
        /// \param seed  Set the random seed of the %RNG if `seed > 0`, otherwise
        ///              use the current random state.
        ///
        void init(const Scalar &mu = Scalar(0), const Scalar &sigma = Scalar(0.01),
                  int seed = -1) {
            check_unit_sizes();

            if (seed > 0) {
                m_rng.seed(seed);
            }

            for (int i = 0; i < nlayer; i++) {
                // initialize layer object with random parameters
                m_layers[i]->init(mu, sigma, m_rng);
                Parameters *layer_init_param = m_layers[i]->get_initial_parameter_pointer();
                layer_init_param->layer_n = i;
                current_param_container_ptr->param_list.push_back(layer_init_param);
                m_layers[i]->set_layer_n(i);
                m_layers[i]->set_param_ptr(current_param_container_ptr);
            }

            current_param_container_ptr->timestamp = 0;
            current_param_container_ptr->active_readers.store(0);
            current_param_container_ptr->stale_flag = false;
            current_param_container_ptr->deleted = false;

            auto size = current_param_container_ptr->param_list.size();
        }

        ///
        /// Debugging tool to check parameter gradients
        ///
        template<typename TargetType>
        void check_gradient(const Matrix &input, const TargetType &target, int npoints,
                            int seed = -1) {
            if (seed > 0) {
                m_rng.seed(seed);
            }

            this->forward(input);
            this->backprop(input, target);
            std::vector<std::vector<Scalar>> param = this->get_parameters();
            std::vector<std::vector<Scalar>> deriv = this->get_derivatives();
            const Scalar eps = 1e-5;
            const int nlayer = deriv.size();

            for (int i = 0; i < npoints; i++) {
                // Randomly select a layer
                const int layer_id = int(m_rng.rand() * nlayer);
                // Randomly pick a parameter, note that some layers may have no parameters
                const int nparam = deriv[layer_id].size();

                if (nparam < 1) {
                    continue;
                }

                const int param_id = int(m_rng.rand() * nparam);
                // Turbulate the parameter a little bit
                const Scalar old = param[layer_id][param_id];
                param[layer_id][param_id] -= eps;
                this->set_parameters(param);
                this->forward(input);
                this->backprop(input, target);
                const Scalar loss_pre = m_output->loss();
                param[layer_id][param_id] += eps * 2;
                this->set_parameters(param);
                this->forward(input);
                this->backprop(input, target);
                const Scalar loss_post = m_output->loss();
                const Scalar deriv_est = (loss_post - loss_pre) / eps / 2;
                std::cout << "[layer " << layer_id << ", param " << param_id <<
                          "] deriv = " << deriv[layer_id][param_id] << ", est = " << deriv_est <<
                          ", diff = " << deriv_est - deriv[layer_id][param_id] << std::endl;
                param[layer_id][param_id] = old;
            }
            // Restore original parameters
            this->set_parameters(param);
        }

        ///
        /// Get the serialized derivatives of layer parameters
        ///
        std::vector<std::vector<Scalar>> get_derivatives() const {

            std::vector<std::vector<Scalar>> res;
            res.reserve(nlayer);

            for (int i = 0; i < nlayer; i++) {
                res.push_back(m_layers[i]->get_derivatives());
            }

            return res;
        }


        ///
        /// Get the serialized layer parameters
        ///
        std::vector<std::vector<Scalar>> get_parameters() const {

            std::vector<std::vector<Scalar>> res;
            res.reserve(nlayer);

            for (int i = 0; i < nlayer; i++) {
                res.push_back(m_layers[i]->get_parameters());
            }

            return res;
        }

        ///
        /// Set the layer parameters
        ///
        /// \param param Serialized layer parameters
        ///
        void set_parameters(const std::vector<std::vector<Scalar>> &param) {


            if (static_cast<int>(param.size()) != nlayer) {
                throw std::invalid_argument("Parameter size does not match");
            }

            for (int i = 0; i < nlayer; i++) {
                m_layers[i]->set_parameters(param[i]);
            }
        }

        void reset() {
            for (auto layer: m_layers) {
                layer->reset();
            }
        }

        void aggregate(NetworkTopology &other_net) {
            if (other_net.nlayer != nlayer) {
                throw std::invalid_argument("Unit sizes do not match");
            }
            for (int i = 0; i < nlayer; i++) {
                m_layers[i]->aggregate(*other_net.m_layers[i]);
            }
        }

        Scalar get_loss() {
            return m_output->loss();
        }

        void normalize_derivatives(float factor) {
            for (int i = 0; i < nlayer; i++) {
                m_layers[i]->normalize_derivatives(factor);
            }
        }
    };
} // namespace MiniDN
#endif /* NETWORKTOPOLOGY_H_ */

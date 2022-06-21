//definition for neural network neuron
#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <Dense>
float sigmoid(float x){		//applies sigmoid to x
	float e = 2.71828;
	return 1/(1 + (pow(e, -1 * (x))));
}
class layer{
	public:
		bool is_second;
		bool is_third;
		int num_neurons;
		bool is_input;
		bool is_output;
		Eigen::MatrixXd activations;
		Eigen::MatrixXd description;
		Eigen::MatrixXd adj_activations;
		Eigen::MatrixXd biases;
		Eigen::MatrixXd propagation_data;
		Eigen::MatrixXd suggested_gradient;
		Eigen::MatrixXd suggested_bias;
		layer* front_layer;
		layer* back_layer;
		layer(float neurons){
			this -> is_third = false;
			this -> is_output = false;
			this -> is_input = false;
			this -> front_layer = NULL;
			this -> back_layer = NULL;
			this -> num_neurons = neurons;
			(this -> activations).resize(num_neurons, 1);
			(this -> adj_activations).resize(num_neurons, 1);
			(this -> biases) = Eigen::MatrixXd::Zero(num_neurons, 1);
			this -> propagation_data = Eigen::MatrixXd::Zero(num_neurons, 1);
			(this -> suggested_bias).resize(num_neurons, 1);
			for(int i = 0; i < num_neurons; i++){
				(this -> suggested_bias)(i, 0) = 0;
			}
		}
		void generate_description(){ 	//creates matrix for calculating layer activations
			int m = (this -> num_neurons);
			int n = (*(this -> back_layer)).num_neurons;
			Eigen::MatrixXd holder(m, n);
			for(int i = 0; i < m; i++){	//filling each cell with proper value
				for(int j = 0; j < n; j++){
					holder(i, j) = 1;
				}
			}
			this -> description = holder;
		}
		void forward_connect(layer* next_layer){	//connects current layer to next
			(this -> front_layer) = next_layer;
			(*next_layer).back_layer = this;
			(*next_layer).description.resize((*next_layer).num_neurons, (this -> num_neurons));
			for(int i = 0; i < (*next_layer).description.rows(); i++){
				for(int j = 0; j < (*next_layer).description.cols(); j++){
					(*next_layer).description(i, j) = 1;
				}
			}
			(*next_layer).suggested_gradient.resize((*next_layer).num_neurons, this -> num_neurons);
		}
		void squishify(){
			for(int i = 0; i < num_neurons; i++){
				(this -> adj_activations)(i, 0) = sigmoid((this -> activations)(i, 0));
			}
		}
		void calculate_layer(){
			if(this -> is_input == true){
				this -> activations = (this -> activations) + (this -> biases);
				return;
			}
			this -> activations = (this -> description) * (*(this -> back_layer)).adj_activations;
			this -> activations = (this -> activations) + (this -> biases);
		}
		void clear_suggestion(){
			for(int i = 0; i < (this -> suggested_gradient).rows(); i++){
				for(int j = 0; j < (this -> suggested_gradient).cols(); j++){
					(this -> suggested_gradient)(i, j) = 0;
				}
			}
		}
};

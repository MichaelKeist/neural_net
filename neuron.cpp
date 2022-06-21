//definition for neural network neuron
#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <Dense>
class neuron{
	public:
		//defining class variables
		std::string debug_string;
		float activation;
		float bias;
		std::vector<float> forward_weights;
		std::vector<neuron*> forward_cons;
		std::vector<neuron*> backward_cons;
		neuron(){	//class constructor
			this -> activation = 0;
			this -> bias = 0;
		}
		void set_debug(std::string new_debug){	//setting debug string
			this -> debug_string = new_debug;
		}
		std::string get_debug(){	//returns debug string
			return this -> debug_string;
		}
		void fconnect(neuron* new_connection){	//appends input neuron pointer to forward connections vector
			(this -> forward_cons).push_back(new_connection);
			(this -> forward_weights).push_back(1);
			(*new_connection).backward_cons.push_back(this);
		}
		int num_fcons(){	//returns number of forward connections
			return (this -> forward_cons).size();
		}
		int num_bcons(){	//returns number of backward connections
			return (this -> backward_cons).size();
		}
		float get_bias(){	//returns current bias value
			return (this -> bias);
		}
		void set_bias(float new_bias){	//sets bias value
			this -> bias = new_bias;
		}
		void adj_bias(float factor){	//adds input value to bias
			(this -> bias) = (this -> bias) + factor;
		}
		void sigmoid(){		//applies bias to raw activation, applies sigmoid, sets as new activation
			float e = 2.71828;
			this -> activation = 1/(1 + (pow(e, -1 * (this -> activation))));
		}
};
class layer{
	public:
		std::vector<neuron*> neurons;
		layer* front_layer;
		layer* back_layer;
		Eigen::MatrixXd description;
		layer(){
		}
		void add_neuron(neuron* new_neuron){
			(this -> neurons).push_back(new_neuron);
		}
		void forward_connect(){		//connects every neuron in current layer to every neuron in forward layer
			neuron* active_neuron;
			for(int i = 0; i < (this -> neurons).size(); i++){
				active_neuron = ((this -> neurons)[i]);
				for(int j = 0; j < (*(this -> front_layer)).neurons.size(); j++){
					(*active_neuron).fconnect((*front_layer).neurons[j]);
				}
			}
		}
		void squishify(){
			for(int i = 0; i < (this -> neurons).size(); i++){
				(*(this -> neurons)[i]).sigmoid();
			}
		}
		void generate_description(){ 	//creates matrix for calculating layer activations
			int m = (this -> neurons).size();
			int n = (*(this -> back_layer)).neurons.size();
			Eigen::MatrixXd holder(m, n);
			for(int i = 0; i < m; i++){	//filling each cell with proper value
				for(int j = 0; j < n; j++){
					holder(i, j) = (*(*(this -> back_layer)).neurons[j]).forward_weights[i];
				}
			}
			this -> description = holder;
		}
		void calculate_layer(){
			int m = (*(this -> back_layer)).neurons.size();
			Eigen::MatrixXd back_activation(m, 1);
			for(int i = 0; i < m; i++){ 	//filling column vector with proper values
				back_activation(i, 0) = (*(*(this -> back_layer)).neurons[i]).activation;
			}
			Eigen::MatrixXd output((this -> description).rows(), 0);
			output = (this -> description) * back_activation;		//performing calculations
			for(int i = 0; i < output.rows(); i++){		//assigning new calculated values
				(*(this -> neurons)[i]).activation = output(i, 0) + (*(this -> neurons)[i]).bias;
			}
		}
};

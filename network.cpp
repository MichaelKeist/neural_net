#include <iostream>
#include </home/michael/fun_projects/cpp_learning/neural_net/neuron2.cpp>
#include <math.h>
#include </home/michael/fun_projects/cpp_learning/neural_net/read_in.cpp>
#include <stdlib.h>
#include <time.h>
float get_cost(layer final_layer, Eigen::MatrixXd expected_values){	//gets the cost of a single training example
	float sum = 0;
	Eigen::MatrixXd holder = final_layer.adj_activations;
	holder = holder - expected_values;
	for(int i = 0; i < holder.rows(); i++){
		sum = sum + pow(holder(i, 0), 2);
	}
	return sum;
}
float dx_sigmoid(float x){	//gets derivative of sigmoid at x = x
	return sigmoid(sigmoid(x)) * (1 - sigmoid(sigmoid(x)));
}
void gradient(layer* active_layer, bool force_correct){	//finds cost function gradient at given layer
	Eigen::MatrixXd holder(((*active_layer).num_neurons), 1);
	Eigen::MatrixXd dx_holder(((*active_layer).num_neurons), 1);
	holder = (*(*active_layer).front_layer).description.transpose() * (*(*active_layer).front_layer).propagation_data;
	for(int i = 0; i < (*active_layer).num_neurons; i++){
		dx_holder(i, 0) = dx_sigmoid((*active_layer).activations(i, 0)) * holder(i, 0);
		if(dx_holder(i, 0) < 0.005 and force_correct == true){
			dx_holder(i,0) = .2 * holder(i, 0);
		}
	}
	(*active_layer).propagation_data = dx_holder;
	if((*active_layer).is_input == false){
		holder = (((*active_layer).propagation_data) * ((*(*active_layer).back_layer).adj_activations.transpose()));
		(*active_layer).suggested_gradient = (*active_layer).suggested_gradient + holder;
	}
	(*active_layer).suggested_bias = (*active_layer).suggested_bias + (*active_layer).propagation_data;
}
void output_gradient(layer* active_layer, Eigen::MatrixXd expected_values, bool force_correct){	//propagation for output layer neurons
	Eigen::MatrixXd holder((*active_layer).propagation_data.rows(), 1);
	Eigen::MatrixXd dx_holder((*active_layer).propagation_data.rows(), 1);
	for(int i = 0; i < (*active_layer).propagation_data.rows(); i++){
		holder(i, 0) = ((*active_layer).adj_activations(i, 0) - expected_values(i, 0));//added pow2
		dx_holder(i, 0) = dx_sigmoid((*active_layer).activations(i, 0));
		if(dx_holder(i, 0) < 0.0005 and force_correct == true){
			dx_holder(i,0) = .2 * holder(i, 0);
		}
	}
	Eigen::MatrixXd final_holder = dx_holder.cwiseProduct(holder);
	(*active_layer).propagation_data = (final_holder);
	(*active_layer).suggested_gradient = (*active_layer).suggested_gradient + (((*active_layer).propagation_data * (*(*active_layer).back_layer).adj_activations.transpose()));
	(*active_layer).suggested_bias = (*active_layer).suggested_bias + final_holder;
}
void back_propagate(layer* last_layer, Eigen::MatrixXd expected_values, bool force_correct){	//full-network back propagation
	output_gradient(last_layer, expected_values, force_correct);
	layer* active_layer = last_layer;
	while(true){
		active_layer = (*active_layer).back_layer;
		gradient(active_layer, force_correct);
		if((*active_layer).is_input == true){
			return;
		}
	}
}
void enact_suggestions(layer* last_layer, bool bias){	//subtracts suggested gradient from weights matrix for all layers
	layer* active_layer = last_layer;
	while(true){
		if((*active_layer).is_input == false){
			(*active_layer).description = (*active_layer).description - (*active_layer).suggested_gradient;
			if(bias == true){
				(*active_layer).biases = (*active_layer).biases - (*active_layer).suggested_bias;
			}
		}
		for(int i = 0; i < (*active_layer).suggested_gradient.rows(); i++){
			for(int j = 0; j < (*active_layer).suggested_gradient.cols(); j++){
				(*active_layer).suggested_gradient(i, j) = 0;
			}
		}
		for(int i = 0; i < (*active_layer).suggested_bias.rows(); i++){
			(*active_layer).suggested_bias(i, 0) = 0;
		}
		if((*active_layer).is_input == true){
			return;
		}
		active_layer = (*active_layer).back_layer;
	}
}
void calculate_network(layer* first_layer){	//given first layer activations, calculate activations for all layers
	(*first_layer).adj_activations = (*first_layer).activations;
	layer* active_layer = (*first_layer).front_layer;	//not doing anything to calculate 1st layer (changed)
	while(true){
		(*active_layer).calculate_layer();
		(*active_layer).squishify();
		if((*active_layer).is_output == true){
			return; 
		}
		active_layer = (*active_layer).front_layer;
	}
}
void normalize_correction(layer* first_layer, int num_cases){
	layer* active_layer = first_layer;
	while(true){
		(*active_layer).suggested_gradient = ((*active_layer).suggested_gradient / (num_cases));
		(*active_layer).suggested_bias = (*active_layer).suggested_bias / (num_cases);
		if((*active_layer).is_output == true){
			return;
		}
		active_layer = (*active_layer).front_layer;
	}
}
void clear_suggestions(layer* first_layer){
	layer* active_layer = first_layer;
	while(true){
		(*active_layer).clear_suggestion();
		if((*active_layer).is_output == true){
			return;
		}
		active_layer = (*active_layer).front_layer;
	}
}
void train_batch(layer* first_layer, layer* last_layer, int begin, int end, bool bias, bool force_correct){ //train
	clear_suggestions(first_layer);
	float cost_sum = 0;
	int saved_begin = begin;
	Eigen::MatrixXd expected;
	layer* active_layer;
	for(begin; begin <= end; begin++){
		expected = read_training_label(begin, begin);
		read_image(first_layer, begin);
		calculate_network(first_layer);
		back_propagate(last_layer, expected, force_correct);
		cost_sum = cost_sum + get_cost(*last_layer, expected);
		//std::cout << (*last_layer).propagation_data << "\n\n\n\n";
	}
	std::cout << "Average cost of training round: " << (cost_sum)/((end-saved_begin)+1) << "\n";
	normalize_correction(first_layer, (end-saved_begin)+1);
	enact_suggestions(last_layer, bias);
}
int test_single(layer* input_layer, layer* output_layer, int position){
	int key = read_test_label(position, position);
	read_test_image(input_layer, position);
	calculate_network(input_layer);
	int network_answer = 0;
	float curr_max = 0;
	for(int i = 0; i < (*output_layer).adj_activations.rows(); i++){
		if((*output_layer).adj_activations(i, 0) > curr_max){
			curr_max = (*output_layer).adj_activations(i, 0);
			network_answer = i;
		}
	}
	if(network_answer == key){
		return 1;
	}else{
		return 0;
	}
}
void generate_description(layer* active_layer){
	int rand_num;
	float new_num;
	if((*active_layer).is_input == true){
		for(int i = 0; i < (*active_layer).biases.rows(); i++){
			rand_num = rand() % 200;
			rand_num = rand_num - 100;
			new_num = (rand_num)/100;	//can change up initial bias values and see what happens
			(*active_layer).biases(i, 0) = new_num;
		}
		return;
	}
	for(int i = 0; i < (*active_layer).description.rows(); i++){
		for(int j = 0; j < (*active_layer).description.cols(); j++){
			rand_num = rand() % 200;	//try removing the +1?????
			rand_num = rand_num - 100;
			new_num = (float)rand_num;
			new_num = new_num / ((*(*active_layer).back_layer).num_neurons);	//the 10* factor is neat
			(*active_layer).description(i, j) = new_num;
		}
	}
	for(int i = 0; i < (*active_layer).biases.rows(); i++){
		rand_num = rand() % 200;
		rand_num = rand_num - 100;
		new_num = (rand_num)/100;	//can change up initial bias values and see what happens
		(*active_layer).biases(i, 0) = new_num;
	}
}
//IMPORTANT-------------------------------
//increased num neurons in layers 2 and 3, had the network do an extra round of training (try having second round use larger batches), biases are now randomly generated @ the start, late layers re-generate themselves once training has already begun//
int main(){
srand (time(NULL));
//making layers
layer first = layer(784);
first.is_input = true;
layer second = layer(64);
layer third = layer(64);
layer fourth = layer(10);
fourth.is_output = true;
//connecting_layers
first.forward_connect(&second);
second.forward_connect(&third);
third.forward_connect(&fourth);
//intializing weights
generate_description(&first);
generate_description(&second);
generate_description(&third);
generate_description(&fourth);
int i = 0;
while(i < 1200){
	std::cout << 1 + (50 * i) << " - " << (50 * (i + 1)) << ": ";
	if(i < 10){
		train_batch(&first, &fourth, (1 + (50 * i)), (50 * (i + 1)), true, false);
	}else{
		train_batch(&first, &fourth, (1 + (50 * i)), (50 * (i + 1)), true, false);
	}
	if(i == 100){
		generate_description(&third);
	}
	if(i == 200){
		generate_description(&fourth);
	}
	i++;
}
//again because why not
i = 0;
while(i < 1200){
	std::cout << 1 + (50 * i) << " - " << (50 * (i + 1)) << ": ";
	if(i < 10){
		train_batch(&first, &fourth, (1 + (50 * i)), (50 * (i + 1)), true, false);
	}else{
		train_batch(&first, &fourth, (1 + (50 * i)), (50 * (i + 1)), true, false);
	}
	i++;
}
std::cout << "Testing network on 10000 images...\n";
i = 1;
int new_sum;
int correct_sum = 0;
int num_tests = 10000;
while(i < num_tests){
	new_sum = test_single(&first, &fourth, i);
	correct_sum = correct_sum + new_sum;
	i++;
	//std::cout << second.adj_activations << "\n\n";
}
float percent_sum = correct_sum;
std::cout << "Total Accuracy: " << ((percent_sum / num_tests) * 100) << "%\n";
}
//todo:
//make cost function
//make file read-in for neurons and cost function
//sort out back propagation

#include <iostream>
#include <fstream>
Eigen::MatrixXd read_training_label(int beginning, int end){
	char buffer[1];
	unsigned int test;
	std::ifstream input("/home/michael/fun_projects/cpp_learning/neural_net/data/train-labels-idx1-ubyte", std::ios::in |  std::ios::binary);
	for(int i = 0; i < beginning + 7; i++){
		input.read(buffer, 1);
	}
	for(int i = 0; i < (end - beginning) + 1; i++){	
		input.read(buffer, 1);
		test = (*buffer);
		*buffer = 0;
	}
	Eigen::MatrixXd expected(10, 1);
	for(int i = 0; i < 10; i++){
		expected(i, 0) = 0.1;
	}
	expected(test, 0) = 0.9;
	return expected;
}
void read_image(layer* input_layer, int position){	//reads indicated mnist number into first layer of network
	char buffer[1];
	unsigned char buffer2[1];
	float test;
	int row = 0;
	int col = 0;
	std::ifstream input("/home/michael/fun_projects/cpp_learning/neural_net/data/train-images-idx3-ubyte");
	input.seekg(168 + 784 * (position - 1));
	for(int i = 0; i < 784; i++){
		input.read(buffer, 1);
		*buffer2 = *buffer;
		test = *buffer2;
		(*input_layer).activations(row, 0) = (test / 255);
		row++;
	}
}
int read_test_label(int beginning, int end){	//reads image and label from test dataset
	char buffer[1];
	unsigned int test;
	std::ifstream input("/home/michael/fun_projects/cpp_learning/neural_net/data/t10k-labels-idx1-ubyte", std::ios::in |  std::ios::binary);
	for(int i = 0; i < beginning + 7; i++){
		input.read(buffer, 1);
	}
	for(int i = 0; i < (end - beginning) + 1; i++){	
		input.read(buffer, 1);
		test = (*buffer);
		*buffer = 0;
	}
	return test;
}
void read_test_image(layer* input_layer, int position){
	char buffer[1];
	unsigned char buffer2[1];
	float test;
	int row = 0;
	int col = 0;
	std::ifstream input("/home/michael/fun_projects/cpp_learning/neural_net/data/t10k-images-idx3-ubyte");
	input.seekg(168 + 784 * (position - 1));
	for(int i = 0; i < 784; i++){
		input.read(buffer, 1);
		*buffer2 = *buffer;
		test = *buffer2;
		(*input_layer).activations(row, 0) = ((test) / 255);
		row++;
	}
}

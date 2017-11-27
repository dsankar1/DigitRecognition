/*
 * Daryan Sankar
 * CS 4242
 * Project 04
 * 04/04/2017
 */
package neuralNet;

import java.util.ArrayList;

public class Main {

	public static void main(String[] args) {
		Network network = new Network(64, 32, 10);
		ArrayList<String> tests = Utility.ReadFile("optdigits_test.txt");
		ArrayList<String> training = Utility.ReadFile("optdigits_train.txt");
		
		// Below trains network given the training set and number of epochs
		network.train(training, 60);
		
		// Below loads weights from "weights.txt"
		//Utility.loadWeights(network, "weights.txt");
		
		// Below runs one epoch of test
		network.test(tests);
		
		// Below Stores weights into "weights.txt"
		//Utility.storeWeights(network, "weights.txt");

	}
	
}

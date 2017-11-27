/*
 * Daryan Sankar
 * CS 4242
 * Project 04
 * 04/04/2017
 */
package neuralNet;

import java.util.ArrayList;
import java.util.Collections;

public class Network {
	
	private static double LEARNING_RATE = .05;
	private ArrayList<Matrix> weights = null;
	private ArrayList<ArrayList<Double>> outputs = null;
	private ArrayList<ArrayList<Double>> gradients = null;
	
	public Network(int input, int hidden, int output) {
		weights = new ArrayList<Matrix>();
		weights.add(new Matrix(input + 1, hidden));
		weights.add(new Matrix(hidden + 1, output));
	}
	
	public void train(ArrayList<String> trainingSet, int epochs) {
		System.out.println("Training...Please wait...");
		int count = 1;
		while (count <= epochs) {
			int correct = 0, total = 0;
			for (String train : trainingSet) {
				String substr = train.substring(train.length() - 1);
				int expected = Integer.parseInt(substr);
				train = train.substring(0, train.length() - 1);
				int answer = getResponse(train);
				
				if (answer == expected) {
					correct++;
				}
				total++;
				
				ArrayList<Double> expectedOutputs = Utility.ExpectedOutput(expected);
				updateWeights(expectedOutputs);
			}
			
			double score = ((double)correct/(double)total) * 100;
			System.out.println("Epoch " + count + " Score: " + score + "%");
			count++;
		}
		System.out.println("Training complete!");
	}
	
	public double test(ArrayList<String> testingSet) {
		System.out.println("Testing...Please wait...");
		double score = 0, correct = 0, total = 0;
		for (String test : testingSet) {
			String substr = test.substring(test.length() - 1);
			int expected = Integer.parseInt(substr);
			test = test.substring(0, test.length() - 1);
			double response = getResponse(test);
			
			if (response == expected) {
				correct++;
			}
			total++;
		}
		score = ((double)correct/(double)total);
		System.out.println("Testing complete! Score: " + (score * 100) + "%");
		return score;
	}
	
	public int getResponse(String test) {
		ArrayList<Double> arrayAnswer = computeOutput(test);
		
		int numberAnswer = 0;
		for (int i = 0; i < arrayAnswer.size(); i++) {
			if (arrayAnswer.get(numberAnswer) < arrayAnswer.get(i)) {
				numberAnswer = i;
			}
		}
		
		return numberAnswer;
	}
	
	public ArrayList<Matrix> getWeightMatrices() {
		return weights;
	}
	
	/*
	 * Below are functions for forward prop and back prop
	 */
	private void updateWeights(ArrayList<Double> expected) {
		computeGradients(expected);
		
		for (int i = 0; i < gradients.size(); i++) {
			Matrix weight = weights.get(i);
			
			ArrayList<Double> input = outputs.get(i);
			ArrayList<Double> gradient = gradients.get(i);
			
			for (int j = 0; j < weight.columnCount(); j++) {
				ArrayList<Double> column = weight.getColumn(j);
				double grad = gradient.get(j);
				
				for (int k = 0; k < column.size(); k++) {
					double delta = grad * LEARNING_RATE * input.get(k);
					double value = column.get(k) + delta;
					weights.get(i).set(value, k, j);
				}
			}
		}
	}
	
	private ArrayList<Double> computeOutput(String data) {
		ArrayList<Double> inputs = new ArrayList<Double>();
		for (Integer i : Utility.ExtractInts(data)) {
			inputs.add((double)i);
		}
		inputs.add(1.0);
		
		outputs = new ArrayList<ArrayList<Double>>();
		outputs.add(inputs);
		
		int count = 0;
		for (Matrix weight : weights) {
			
			ArrayList<Double> output = Matrix.dotProduct(inputs, weight);
			output = Utility.Sigmoid(output);
			
			inputs = new ArrayList<Double>();
			for (Double temp : output) {
				inputs.add(temp);
			}
			
			if (count < (weights.size() - 1)) {
				inputs.add(1.0);
			}
			
			outputs.add(inputs);
			count++;
		}
		return inputs;
	}
	
	private void computeGradients(ArrayList<Double> expected) {
		gradients = new ArrayList<ArrayList<Double>>();

		for (int i = (outputs.size() - 1); i > 0; i--) {
			ArrayList<Double> output = outputs.get(i);
			
			if (i == (outputs.size() - 1)) {
				ArrayList<Double> gradient = Utility.OutputGradients(expected, output);
				gradients.add(gradient);
			}
			else {
				Matrix weight = weights.get(i);
				ArrayList<Double> prevGradient = gradients.get(gradients.size() - 1);
				ArrayList<Double> gradient = Utility.HiddenGradients(output, prevGradient, weight);
				gradients.add(gradient);
			}
		}
		
		Collections.reverse(gradients);
	}
	
}

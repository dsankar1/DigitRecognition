/*
 * Daryan Sankar
 * CS 4242
 * Project 04
 * 04/04/2017
 */
package neuralNet;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;

public class Utility {
	
	public static void storeWeights(Network network, String filename) {
		ArrayList<Matrix> weights = network.getWeightMatrices();

		try {
			PrintWriter writer = new PrintWriter(filename);
			
			for (Matrix matrix : weights) {
				ArrayList<Double> listForm = matrix.getWeights();
				String line = listForm.toString();
				line = line.replaceAll(" ", "");
				line = line.substring(1, line.length() - 1);
				line = line.trim();
				writer.println(line);
			}
			writer.close();
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
	}
	
	public static void loadWeights(Network network, String filename) {
		ArrayList<String> lines = ReadFile(filename);
		ArrayList<Matrix> weights = network.getWeightMatrices();
		
		for (int i = 0; i < lines.size(); i++) {
			String line = lines.get(i);
			
			Matrix m = weights.get(i);
			m.clear();
			
			ArrayList<Double> weight = Utility.ExtractDoubles(line);
			
			for (Double temp : weight) {
				m.insert(temp);
			}
			
			weights.set(i, m);
		}
	}
	
	public static ArrayList<Double> ExpectedOutput(int number) {
		ArrayList<Double> output = new ArrayList<Double>();
		for (int i = 0; i < 10; i++) {
			output.add(0.0);
		}
		
		output.set(number, 1.0);
		return output;
	}
	
	public static ArrayList<Double> OutputGradients(ArrayList<Double> expected, ArrayList<Double> actual) {
		ArrayList<Double> gradients = new ArrayList<Double>();
		
		for (int i = 0; i < expected.size(); i++) {
			double e = expected.get(i);
			double a = actual.get(i);
			
			double gradient = (e - a) * (a * (1 - a));
			gradients.add(gradient);
		}
		
		return gradients;
	}
	
	public static ArrayList<Double> HiddenGradients(ArrayList<Double> outputs, ArrayList<Double> prevGradients, Matrix weights) {
		ArrayList<Double> gradients = new ArrayList<Double>();
		
		for (int i = 0; i < outputs.size(); i++) {
			double output = outputs.get(i);
			output = output * (1 - output);
			
			double sumOfProducts = 0;
			ArrayList<Double> weight = weights.getRow(i);
			for (int j = 0; j < weight.size(); j++) {
				sumOfProducts += (weight.get(j) * prevGradients.get(j));
			}
			
			double gradient = output * sumOfProducts;
			gradients.add(gradient);
		}
		
		return gradients;
	}
	
	public static ArrayList<Double> Sigmoid(ArrayList<Double> preactivations) {
		ArrayList<Double> activations = new ArrayList<Double>();
		
		for (Double preactivated : preactivations) {
			double activated = (1 / (1 + Math.pow(Math.E, -preactivated)));
			activations.add(activated);
		}
		
		return activations;
	}
	
	public static ArrayList<String> ReadFile(String filename) {
		ArrayList<String> lines = new ArrayList<String>();
		try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
		    String line;
		    while ((line = br.readLine()) != null) {
		    	line = line.trim();
		    	line = line.replaceAll(",", " ");
		    	lines.add(line);
		    }
		}
		catch(Exception e) {
			System.out.println("File wasn't found.");
		}
		
		return lines;
	}
	
	public static ArrayList<Double> ExtractDoubles(String line) {
		ArrayList<Double> data = new ArrayList<Double>();
		line = line + " ";
		
		int start = 0;
		int end = line.indexOf(" ", start);
		while (end > start) {
			String subString = line.substring(start, end);
			data.add(Double.parseDouble(subString));
			start = end + 1;
			end = line.indexOf(" ", start);
		}
		return data;
	}
	
	public static ArrayList<Integer> ExtractInts(String line) {
		ArrayList<Integer> data = new ArrayList<Integer>();
		line = line + " ";
		
		int start = 0;
		int end = line.indexOf(" ", start);
		while (end > start) {
			String subString = line.substring(start, end);
			data.add(Integer.parseInt(subString));
			start = end + 1;
			end = line.indexOf(" ", start);
		}
		return data;
	}
	
}

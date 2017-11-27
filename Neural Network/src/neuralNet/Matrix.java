/*
 * Daryan Sankar
 * CS 4242
 * Project 04
 * 04/04/2017
 */
package neuralNet;

import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

public class Matrix {

	private double[][] matrix = null;
	private int rows, columns;
	private int x = 0, y = 0;
	
	public Matrix(int rows, int columns) {
		matrix = new double[rows][columns];
		this.rows = rows;
		this.columns = columns;
		
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				double rand = ((ThreadLocalRandom.current().nextDouble())/100);
				matrix[i][j] = rand;
			}
		}
	}
	
	public int size() {
		return (rows * columns);
	}
	
	public int rowCount() {
		return rows;
	}
	
	public int columnCount() {
		return columns;
	}
	
	public double get(int row, int column) {
		return matrix[row][column];
	}
	
	public void set(double value, int row, int column) {
		matrix[row][column] = value;
	}
	
	public void clear() {
		x = 0;
		y = 0;
	}
	
	public void insert(double value) {
		if (x < columns && y < rows) {
			matrix[y][x] = value;
			x++;
			
			if (x == columns) {
				x = 0;
				y++;
			}
		}
	}
	
	public ArrayList<Double> getRow(int index) {
		ArrayList<Double> row = new ArrayList<Double>();
		
		for (int i = 0; i < columns; i++) {
			double value = matrix[index][i];
			row.add(value);
		}
		
		return row;
	}
	
	public ArrayList<Double> getColumn(int index) {
		ArrayList<Double> column = new ArrayList<Double>();
		
		for (int i = 0; i < rows; i++) {
			double value = matrix[i][index];
			column.add(value);
		}
		
		return column;
	}
	
	public ArrayList<Double> getWeights() {
		ArrayList<Double> weights = new ArrayList<Double>();
		
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				double weight = matrix[i][j];
				weights.add(weight);
			}
		}
		
		return weights;
	}
	
	public void print() {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				System.out.print(matrix[i][j] + " | ");
			}
			System.out.println();
		}
	}
	
	public static ArrayList<Double> dotProduct(ArrayList<Double> inputs, Matrix weights) {
		ArrayList<Double> results = new ArrayList<Double>();
		
		try {
			for (int i = 0; i < weights.columnCount(); i++) {
				ArrayList<Double> column = weights.getColumn(i);
				
				double product = 0;
				for (int j = 0; j < inputs.size(); j++) {
					product += (inputs.get(j) * column.get(j));
				}
				
				results.add(product);
			}			
		}
		catch (Exception e) {
			System.out.println("Multiplication Error Occurred.");
			System.exit(0);
		}
		
		return results;
	}
	
}

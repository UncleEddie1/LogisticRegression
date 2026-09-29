import java.io.*;
import java.util.*;

public class LogisticRegression{
    public double[] weights;
    public double lr;
    public int NUM_EPOCHS;
    public int BATCH_SIZE;

    public LogisticRegression(int NUM_FEATURES, double lr, int NUM_EPOCHS, int BATCH_SIZE) {
        //this.weights = new double[NUM_FEATURES + 1];
        this.weights = initializeRandomWeights(new double[NUM_FEATURES + 1]);
        this.lr = lr;
        this.NUM_EPOCHS = NUM_EPOCHS;
        this.BATCH_SIZE = BATCH_SIZE;
    }

    public void train(double[][] X, double[] y, double lr_decay) {
        int m = X.length;
        int n = X[0].length;
        int o = X.length/BATCH_SIZE;

        for (int iter = 0; iter < NUM_EPOCHS; iter++) {
            for (int batch = 0; batch < o; batch++){
                double[] gradient = new double[weights.length];
                for (int i = 0; i < BATCH_SIZE; i++){
                    double[] x_i = X[i+batch*BATCH_SIZE];
                    double y_i = y[i+batch*BATCH_SIZE];
                    double predicted = predictProbability(x_i);
                    double error = predicted - y_i;
    
                    gradient[0] += error;
                    
                    for (int j = 0; j < n; j++) {
                        gradient[j + 1] += error * x_i[j];
                    }
                }
                for (int j = 0; j < weights.length; j++) {
                    weights[j] -= lr * gradient[j] / m;
                }
            }
            if (iter % (Math.max(NUM_EPOCHS, 10)/10) == 0 || iter == NUM_EPOCHS-1){
                System.out.println("Epoch: " + iter + " | Loss: " + calculateLoss(X, y) + " | Model Accuracy: " + (int)(calculateIndividualAccuracy(X, y)*10000)/100.0 + "%");
            }
            if (iter % (Math.max(NUM_EPOCHS, 5)/5) == 0){
                updateLr(this.lr, lr_decay);
            }
            /*
            if (iter % 5 == 0){
                this.lr = calculateLoss(X, y);
            }
            */
        }
    }

    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }
    
    public double[] initializeRandomWeights(double[] weights){
        Random weightGenerator = new Random(42);
        for (int i = 0; i<weights.length; i++){
            weights[i] = weightGenerator.nextDouble();
        }
        return weights;
    }

    private void updateLr(double lr, double decay_coeff){
        lr *= Math.exp(-decay_coeff);
    }

    public double predictProbability(double[] x) {
        double z = weights[0];
        for (int i = 0; i < x.length; i++) {
            z += weights[i+1] * x[i];
        }
        return sigmoid(z);
    }

    public int predict(double[] x) {
        if (predictProbability(x) >= 0.5){
            return 1;
        }
        return 0;
    }

    public double calculateLoss(double[][] X, double[] y) {
        int m = X.length;
        double loss = 0.0;

        for (int i = 0; i < m; i++) {
            double[] x_i = X[i];
            double y_i = y[i];
            double predicted = predictProbability(x_i);
            loss += y_i * Math.log(predicted) + (1 - y_i) * Math.log(1 - predicted);
        }
        return -loss / m;
    }

    public double calculateIndividualAccuracy(double[][] X, double[] y){
        double accuracy = 0;
        for (int i = 0; i<X.length; i++){
            if (predict(X[i]) == y[i]){
                accuracy++;
            }
        }
        accuracy /= X.length;
        return accuracy;
    }

    public void normalizeData(double[][] X){
        int n = X.length;
        int m = X[0].length;
        for (int j = 0; j<m; j++){
            double maxVal = 0;
            for (int i = 0; i<n; i++){
                maxVal = Math.max(maxVal, X[i][j]);
            }
            if (maxVal == 0){
                continue;
            }
            for (int i = 0; i<n; i++){
                X[i][j] /= maxVal;
            }
        }
    }
}
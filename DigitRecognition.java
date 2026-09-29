import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

public class DigitRecognition {
    public static void main(String[] args)throws IOException {

        // Hyperparameters
        int NUM_FEATURES = 64;
        int NUM_CLASSES = 10;
        int NUM_EPOCHS = 1000;
        int NUM_TRAIN_DATA = 3823;
        int NUM_TEST_DATA = 1797;
        double lr = 10;
        int BATCH_SIZE = 50;
        double decay_coeff = 0.2;

        String FILE_NAME = "Digit Recognition";
        String WRITE_LOCATION = "BestAccuracies.txt";

        String train_directory = "DigitTrain.csv";
        String test_directory = "DigitTest.csv";

        // Load Train data
        double[][] train_data = loadData(NUM_TRAIN_DATA, NUM_FEATURES, train_directory);
        double[][] train_labels = loadLabels(NUM_CLASSES, NUM_TRAIN_DATA, train_directory);
        double[] train_answers = loadCorrectAnswers(NUM_TRAIN_DATA, train_directory);

        // Load Test data
        double[][] test_data = loadData(NUM_TEST_DATA, NUM_FEATURES, test_directory);
        double[][] test_labels = loadLabels(NUM_CLASSES, NUM_TEST_DATA, test_directory);
        double[] test_answers = loadCorrectAnswers(NUM_TRAIN_DATA, test_directory);

        // Create Models
        LogisticRegression[] models = new LogisticRegression[NUM_CLASSES];
        for (int k = 0; k<NUM_CLASSES; k++){
            models[k] = new LogisticRegression(NUM_FEATURES, lr, NUM_EPOCHS, BATCH_SIZE);
        }

        // Normalize Data
        models[0].normalizeData(train_data);
        models[0].normalizeData(test_data);

        System.out.println(Arrays.toString(train_data[0]));

        // Train Models
        for (int k = 0; k<NUM_CLASSES; k++){
            System.out.println("\nTraining Model " + k + ":");
            models[k].train(train_data, train_labels[k], decay_coeff);
            System.out.println(Arrays.toString(models[k].weights));
        }
        System.out.println();

        // Train Accuracy
        double train_accuracy = calculateAccuracy(models, NUM_CLASSES, train_data, train_answers);
        System.out.println("Train Accuracy: " + train_accuracy);

        // Test Accuracy
        double test_accuracy = calculateAccuracy(models, NUM_CLASSES, test_data, test_answers);
        writeAccuracy(FILE_NAME, WRITE_LOCATION, test_accuracy, NUM_EPOCHS, NUM_TEST_DATA, BATCH_SIZE);
        System.out.println("Test Accuracy: " + test_accuracy);
        System.out.println();
        
        // Model-specific Test Accuracy
        double[] accuracies = new double[NUM_CLASSES];
        for (int k = 0; k<NUM_CLASSES; k++){
            accuracies[k] = models[k].calculateIndividualAccuracy(test_data, test_labels[k]);

            System.out.printf("Model %d accuracy: %.2f%%\n", k, (accuracies[k]*100));
        }
    }

    public static double[][] loadLabels(int NUM_CLASSES, int NUM_DATA, String location)throws IOException{
        double[][] labels = new double[NUM_CLASSES][NUM_DATA];
        BufferedReader br = new BufferedReader(new FileReader(location));
        String nextLine;
        int i = 0;
        while ((nextLine = br.readLine()) != null){
            String[] values = nextLine.split(",");
            labels[Integer.parseInt(values[values.length-1])][i] = 1;
            i++;
        }
        return labels;
    }

    public static double[] loadCorrectAnswers(int NUM_DATA, String location)throws IOException {
        double[] answers = new double[NUM_DATA];
        BufferedReader br = new BufferedReader(new FileReader(location));
        String nextLine;
        int i = 0;
        while ((nextLine = br.readLine()) != null){
            String[] values = nextLine.split(",");
            answers[i] = Integer.parseInt(values[values.length-1]);
            i++;
        }
        return answers;
    }

    public static double[][] loadData(int NUM_DATA, int NUM_FEATURES, String location)throws IOException {
        double[][] data = new double[NUM_DATA][NUM_FEATURES];
        BufferedReader br = new BufferedReader(new FileReader(location));
        String nextLine;
        int i = 0;
        while ((nextLine = br.readLine()) != null){
            String[] values = nextLine.split(",");
            for (int j = 0; j<values.length-1; j++){
                data[i][j] = Double.parseDouble(values[j]);
            }
            i++;
        }
        return data;
    }

    public static double calculateAccuracy(LogisticRegression[] models, int NUM_CLASSES, double[][] X, double[] y) {
        int m = X.length;
        int correctAnswers = 0;
        double accuracy;
        for (int k = 0; k<m; k++){
            double maxProb = 0;
            int highestIndex = 0;

            for (int l = 0; l<NUM_CLASSES; l++){
                double y_prob = models[l].predictProbability(X[k]);
                if (y_prob > maxProb){
                    maxProb = y_prob;
                    highestIndex = l;
                }
            }

            if (y[k] == highestIndex){
                correctAnswers++;
            }
        }
        accuracy = (1.0*correctAnswers)/m;
        return accuracy;
    }

    public static void writeAccuracy(String name, String location, double accuracy, int NUM_EPOCHS, int NUM_TEST_DATA, int BATCH_SIZE)throws IOException{
        String timeStamp = new SimpleDateFormat("yyyy.MM.dd | HH:mm:ss").format(new Date());

        FileWriter myWriter = new FileWriter("BestAccuracies.txt", true);
        myWriter.write(name + ": " + timeStamp + "\n Epochs: " + NUM_EPOCHS + "\n TEST_DATA: " + NUM_TEST_DATA + "\n BATCH_SIZE: " + BATCH_SIZE + "\n Accuracy: " +  Math.floor(accuracy * 10000)/100 + "%\n\n");
        myWriter.close();
    }
}

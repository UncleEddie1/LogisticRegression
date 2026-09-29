import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

public class DryBeanRecognition {
    public static void main(String[] args)throws IOException {

        // Hyperparameters
        int NUM_FEATURES = 16;
        int NUM_CLASSES = 7;
        int NUM_EPOCHS = 2000;
        int NUM_TRAIN_DATA = 10800;
        int NUM_TEST_DATA = 2811;
        double lr = 10;
        int BATCH_SIZE = 100;
        double decay_coeff = 0.5;

        String FILE_NAME = "Dry Bean Recognition";
        String WRITE_LOCATION = "BestAccuracies.txt";

        String train_directory = "DryBeanTrain.csv";
        String test_directory = "DryBeanTest.csv";

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

        // Normalizing Data
        models[0].normalizeData(train_data);
        models[0].normalizeData(test_data);

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
            System.out.println("Model " + k + " accuracy: " + accuracies[k]);
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
                    maxProb = models[l].predictProbability(X[k]);
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

/*
 * 
 data[i][0] = Double.parseDouble(values[0])/100000;
            data[i][1] = Double.parseDouble(values[1])/1000;
            data[i][2] = Double.parseDouble(values[2])/1000;
            data[i][3] = Double.parseDouble(values[3])/500;
            data[i][4] = Double.parseDouble(values[4]);
            data[i][5] = Double.parseDouble(values[5]);
            data[i][6] = Double.parseDouble(values[6])/100000;
            data[i][7] = Double.parseDouble(values[7])/500;
            data[i][8] = Double.parseDouble(values[8]);
            data[i][9] = Double.parseDouble(values[9]);
            data[i][10] = Double.parseDouble(values[10]);
            data[i][11] = Double.parseDouble(values[11]);
            data[i][12] = Double.parseDouble(values[12])*100;
            data[i][13] = Double.parseDouble(values[13])*100;
            data[i][14] = Double.parseDouble(values[14]);
            data[i][15] = Double.parseDouble(values[15]);
 */
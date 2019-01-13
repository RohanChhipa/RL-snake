import org.apache.commons.math3.analysis.function.Tanh;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.analysis.function.Sigmoid;

import java.io.*;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public class FeedForwardNetwork
{
    int inputLayerSize;
    int hiddenLayerSize;
    int outputLayerSize;

    double[] inputLayer;
    double[] hiddenLayer;
    double[] outputLayer;

    double[][] inputToHidden;
    double[][] hiddenToOutput;

    ActivationFunction activationFunction;

    public FeedForwardNetwork(int inputLayerSize, int hiddenLayerSize, int outputLayerSize)
    {
        this.inputLayerSize = inputLayerSize;
        this.hiddenLayerSize = hiddenLayerSize;
        this.outputLayerSize = outputLayerSize;

        inputLayer = new double[this.inputLayerSize];
        hiddenLayer = new double[this.hiddenLayerSize];
        outputLayer = new double[this.outputLayerSize];

        inputToHidden = new double[this.inputLayerSize][hiddenLayerSize];
        hiddenToOutput = new double[this.hiddenLayerSize][this.outputLayerSize];

        initWeights();
    }

    public FeedForwardNetwork(FeedForwardNetwork feedForwardNetwork)
    {
        this(feedForwardNetwork.inputLayerSize, feedForwardNetwork.hiddenLayerSize, feedForwardNetwork.outputLayerSize);

        for (int k = 0; k < inputToHidden.length; k++)
            for (int j = 0; j < inputToHidden[k].length; j++)
                inputToHidden[k][j] = feedForwardNetwork.inputToHidden[k][j];

        for (int k = 0; k < hiddenToOutput.length; k++)
            for (int j = 0; j < hiddenToOutput[k].length; j++)
                hiddenToOutput[k][j] = feedForwardNetwork.hiddenToOutput[k][j];
    }

    public FeedForwardNetwork(File file) throws IOException
    {
        BufferedReader bufferedReader = new BufferedReader(new FileReader(file));

        this.inputLayerSize = Integer.valueOf(bufferedReader.readLine());
        this.hiddenLayerSize = Integer.valueOf(bufferedReader.readLine());
        this.outputLayerSize = Integer.valueOf(bufferedReader.readLine());

        inputLayer = new double[this.inputLayerSize];
        hiddenLayer = new double[this.hiddenLayerSize];
        outputLayer = new double[this.outputLayerSize];

        inputToHidden = new double[this.inputLayerSize][hiddenLayerSize];
        hiddenToOutput = new double[this.hiddenLayerSize][this.outputLayerSize];

        for (int k = 0; k < inputToHidden.length; k++)
            for (int j = 0; j < inputToHidden[k].length; j++)
                inputToHidden[k][j] = Double.valueOf(bufferedReader.readLine());

        for (int k = 0; k < hiddenToOutput.length; k++)
            for (int j = 0; j < hiddenToOutput[k].length; j++)
                hiddenToOutput[k][j] = Double.valueOf(bufferedReader.readLine());
    }

    public void feedForward(double[] inputs)
    {
        setInputs(inputs);

        Sigmoid activation = new Sigmoid();

//        Function<Double, Double> activation = (value) -> Math.max(0, value);

//        Tanh activation = new Tanh();

        for (int k = 0; k < inputToHidden.length; k++)
            for (int j = 0; j < inputToHidden[k].length; j++)
                hiddenLayer[j] += inputLayer[k] * inputToHidden[k][j];

        for (int k = 0; k < hiddenLayer.length; k++)
            hiddenLayer[k] = activation.value(hiddenLayer[k]);

        for (int k = 0; k < hiddenToOutput.length; k++)
            for (int j = 0; j < hiddenToOutput[k].length; j++)
                outputLayer[j] += hiddenLayer[k] * hiddenToOutput[k][j];

        for (int k = 0; k < outputLayerSize; k++)
            outputLayer[k] = activation.value(outputLayer[k]);
    }

    public void clearLayers()
    {
        Arrays.fill(inputLayer, 0, inputLayerSize, 0);
        Arrays.fill(hiddenLayer, 0, hiddenLayerSize, 0);
        Arrays.fill(outputLayer, 0, outputLayerSize, 0);
    }

    private void setInputs(double[] inputs)
    {
        for (int k = 0; k < inputs.length; k++)
            inputLayer[k] = inputs[k];
    }

    private void initWeights()
    {
        MersenneTwister mersenneTwister = new MersenneTwister(System.nanoTime());

        double fanin = 1 / Math.sqrt(inputLayerSize);
        for (int k = 0; k < inputToHidden.length; k++)
            for (int j = 0; j < inputToHidden[k].length; j++)
//                inputToHidden[k][j] = mersenneTwister.nextGaussian() * fanin;
                inputToHidden[k][j] = (mersenneTwister.nextDouble() * (2 * fanin)) - fanin;

        fanin = 1 / Math.sqrt(hiddenLayerSize);
        for (int k = 0; k < hiddenToOutput.length; k++)
            for (int j = 0; j < hiddenToOutput[k].length; j++)
//                hiddenToOutput[k][j] = mersenneTwister.nextGaussian() * fanin;
                hiddenToOutput[k][j] = (mersenneTwister.nextDouble() * (2 * fanin)) - fanin;
    }

    public File writeToFile(String fileName)
    {
        File file = new File(fileName);
        if (file.exists())
            file.delete();

        try (PrintWriter printWriter = new PrintWriter(file))
        {
            printWriter.println(inputLayerSize);
            printWriter.println(hiddenLayerSize);
            printWriter.println(outputLayerSize);

            for (int k = 0; k < inputToHidden.length; k++)
                for (int j = 0; j < inputToHidden[k].length; j++)
                    printWriter.println(inputToHidden[k][j]);

            for (int k = 0; k < hiddenToOutput.length; k++)
                for (int j = 0; j < hiddenToOutput[k].length; j++)
                    printWriter.println(hiddenToOutput[k][j]);
        }

        catch (FileNotFoundException e)
        {
            System.out.println("Failed to write NN state");
            return null;
        }

        return file;
    }
}

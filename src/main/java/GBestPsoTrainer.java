import org.apache.commons.math3.random.MersenneTwister;

import java.util.Scanner;
import java.util.function.BiFunction;

class GBestPsoTrainer
{
    private static final int INPUT_LAYER_SIZE = 8;
    private static final int HIDDEN_LAYER_SIZE = 12;
    private static final int OUTPUT_LAYER_SIZE = 4;

    private BiFunction<Particle, Particle, Boolean> fitnessComparator = (a, b) -> a.fitness > b.fitness;

    int maxIteration = 2000;
    int particlesSize = 50;

    double c1 = 1.49618;
    double c2 = 1.49618;
    double w = 0.729844;

    private Particle[] particles;
    private Particle[] personalBest;
    private Particle globalBest;

    GBestPsoTrainer()
    {
        particles = initParticles(particlesSize);

        personalBest = new Particle[particles.length];
        for (int k = 0; k < personalBest.length; k++)
            personalBest[k] = new Particle(particles[k]);

        globalBest = new Particle(personalBest[0]);
    }

    void train()
    {
        for (int z = 0; z < maxIteration; z++)
        {
            evaluateFitness();
            evaluatePersonalBests();
            evaluateGlobalBest();
            updatePositionAndVelocity();

            System.out.println(String.format("Iteration: %d - Global best: %f", z, globalBest.fitness));
        }

        Scanner sc = new Scanner(System.in);
        sc.nextLine();

        globalBest.feedForwardNetwork.writeToFile("./SNEK-01.NN");
        evaluateAccuracy();
    }

    private void reinitParticles()
    {
        for (int k = 0; k < (0.25 * particles.length); k++)
        {
            int i = tournamentSelect();

            particles[i] = new Particle();
            personalBest[i] = new Particle(particles[i]);
        }

        evaluatePersonalBests();
        evaluateGlobalBest();
    }

    private int tournamentSelect()
    {
        MersenneTwister mersenneTwister = new MersenneTwister(System.nanoTime());

        int i = mersenneTwister.nextInt(particles.length);
        for (int k = 0; k < 5; k++)
        {
            int j = mersenneTwister.nextInt(particles.length);
            if (fitnessComparator.apply(particles[i], particles[j]))
                i = j;
        }

        return i;
    }

    private void evaluateFitness()
    {
        for (Particle p : particles)
            p.fitness = new Snake().run(p.feedForwardNetwork, false, 0);
    }

    private void evaluatePersonalBests()
    {
        for (int k = 0; k < particles.length; k++)
            if (fitnessComparator.apply(particles[k], personalBest[k]))
                personalBest[k] = new Particle(particles[k]);
    }

    private void evaluateGlobalBest()
    {
        Particle best = globalBest;
        for (Particle p : personalBest)
            if (fitnessComparator.apply(p, best))
                best = p;

        globalBest = new Particle(best);
    }

    private void updatePositionAndVelocity()
    {
        MersenneTwister mersenneTwister = new MersenneTwister(System.nanoTime());
        for (int k = 0; k < particles.length; k++)
        {
            Particle p = particles[k];

            for (int i = 0; i < p.inputToHiddenVelocity.length; i++)
            {
                for (int j = 0; j < p.inputToHiddenVelocity[i].length; j++)
                {
                    double r1 = mersenneTwister.nextDouble();
                    double r2 = mersenneTwister.nextDouble();

                    p.inputToHiddenVelocity[i][j] *= w;
                    p.inputToHiddenVelocity[i][j] += r1 * c1 * (personalBest[k].feedForwardNetwork.inputToHidden[i][j] - p.feedForwardNetwork.inputToHidden[i][j]);
                    p.inputToHiddenVelocity[i][j] += r2 * c2 * (globalBest.feedForwardNetwork.inputToHidden[i][j] - p.feedForwardNetwork.inputToHidden[i][j]);
                }
            }

            for (int i = 0; i < p.hiddenToOutputVelocity.length; i++)
            {
                for (int j = 0; j < p.hiddenToOutputVelocity[i].length; j++)
                {
                    double r1 = mersenneTwister.nextDouble();
                    double r2 = mersenneTwister.nextDouble();

                    p.hiddenToOutputVelocity[i][j] *= w;
                    p.hiddenToOutputVelocity[i][j] += r1 * c1 * (personalBest[k].feedForwardNetwork.hiddenToOutput[i][j] - p.feedForwardNetwork.hiddenToOutput[i][j]);
                    p.hiddenToOutputVelocity[i][j] += r2 * c2 * (globalBest.feedForwardNetwork.hiddenToOutput[i][j] - p.feedForwardNetwork.hiddenToOutput[i][j]);
                }
            }

            p.updatePosition();
        }
    }

    private void evaluateAccuracy()
    {
        new Snake().run(globalBest.feedForwardNetwork, true, 0);
    }

    private Particle[] initParticles(int n)
    {
        Particle[] particles = new Particle[n];
        for (int k = 0; k < n; k++)
            particles[k] = new Particle();

        return particles;
    }

    private class Particle
    {
        FeedForwardNetwork feedForwardNetwork;

        double fitness;

        double[][] inputToHiddenVelocity;
        double[][] hiddenToOutputVelocity;

        Particle()
        {
            feedForwardNetwork = new FeedForwardNetwork(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);

            inputToHiddenVelocity = new double[feedForwardNetwork.inputToHidden.length][feedForwardNetwork.inputToHidden[0].length];
            hiddenToOutputVelocity = new double[feedForwardNetwork.hiddenToOutput.length][feedForwardNetwork.hiddenToOutput[0].length];

            fitness = Double.MIN_VALUE;
        }

        Particle(Particle p)
        {
            feedForwardNetwork = new FeedForwardNetwork(p.feedForwardNetwork);

            inputToHiddenVelocity = new double[feedForwardNetwork.inputToHidden.length][feedForwardNetwork.inputToHidden[0].length];
            hiddenToOutputVelocity = new double[feedForwardNetwork.hiddenToOutput.length][feedForwardNetwork.hiddenToOutput[0].length];

            for (int k = 0; k < inputToHiddenVelocity.length; k++)
                for (int j = 0; j < inputToHiddenVelocity[k].length; j++)
                    inputToHiddenVelocity[k][j] = p.inputToHiddenVelocity[k][j];

            for (int k = 0; k < hiddenToOutputVelocity.length; k++)
                for (int j = 0; j < hiddenToOutputVelocity[k].length; j++)
                    hiddenToOutputVelocity[k][j] = p.hiddenToOutputVelocity[k][j];

            fitness = p.fitness;
        }

        void updatePosition()
        {
            for (int k = 0; k < inputToHiddenVelocity.length; k++)
                for (int j = 0; j < inputToHiddenVelocity[k].length; j++)
                    feedForwardNetwork.inputToHidden[k][j] += inputToHiddenVelocity[k][j];

            for (int k = 0; k < hiddenToOutputVelocity.length; k++)
                for (int j = 0; j < hiddenToOutputVelocity[k].length; j++)
                    feedForwardNetwork.hiddenToOutput[k][j] += hiddenToOutputVelocity[k][j];
        }

        public boolean valid()
        {
            for (int k = 0; k < inputToHiddenVelocity.length; k++)
                for (int j = 0; j < inputToHiddenVelocity[k].length; j++)
                    if (Math.abs(feedForwardNetwork.inputToHidden[k][j]) >= 6.0)
                        return false;

            for (int k = 0; k < hiddenToOutputVelocity.length; k++)
                for (int j = 0; j < hiddenToOutputVelocity[k].length; j++)
                    if (Math.abs(feedForwardNetwork.hiddenToOutput[k][j]) >= 6.0)
                        return false;

            return true;
        }
    }

    private <T> void log(T t)
    {
        System.out.println(t);
    }
}
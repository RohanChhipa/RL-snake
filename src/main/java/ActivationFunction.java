import java.util.function.Function;

public enum ActivationFunction
{
    SIGMOID( x -> 1.0 / (1.0 + Math.pow(Math.E, -1.0 * x)) );

    private Function<Double, Double> function;

    ActivationFunction(Function<Double, Double> function)
    {
        this.function = function;
    }

    double apply(double value)
    {
        return function.apply(value);
    }
}

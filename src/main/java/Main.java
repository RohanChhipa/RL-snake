import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class Main
{
    public static void main(String[] args) throws IOException
    {
        List<String> arguments = Arrays.asList(args);

        // GBestPsoTrainer psoTrainer = new GBestPsoTrainer();
        // psoTrainer.train();

//        if (arguments.contains("-t"))
//        {
//            GBestPsoTrainer psoTrainer = new GBestPsoTrainer();
//            psoTrainer.train();
//        }
//        else
//        {
           int obstacleType = 0;
           if (arguments.stream().anyMatch(s -> s.equals("-o")))
           {
               int i = arguments.indexOf("-o") + 1;
               obstacleType = Integer.valueOf(arguments.get(i));
           }
//
           new Snake().run(new FeedForwardNetwork(new File("./SNEK-01.NN")), true, obstacleType);
    //    }
    }
}
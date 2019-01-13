import org.apache.commons.math3.random.MersenneTwister;

import java.util.ArrayList;
import java.util.Random;

class Snake
{
    private IntTuple mapSize;
    private int lastEatenCount;

    private IntTuple mouse;
    private ArrayList<IntTuple> snake;

    private ArrayList<IntTuple> obstacles;

    double run(FeedForwardNetwork feedForwardNetwork, boolean evaluationMode, int obstacleType)
    {
        double score = 0;

        lastEatenCount = 0;
        mapSize = new IntTuple(20, 10);

        obstacles = initObstacles(obstacleType);

        snake = initSnake();
        mouse = new IntTuple(new Random().nextInt(mapSize.x), new Random().nextInt(mapSize.y));

        double prevDistance = mouse.distance(snake.get(0));

        IntTuple move = new IntTuple(1, 0);
        while (!isGameComplete(evaluationMode))
        {
            boolean mouseMove = false;
            if (evaluationMode)
            {
                try
                {
                    printMap();
                    Thread.sleep(50);
                }

                catch (InterruptedException e)
                {
                    e.printStackTrace();
                }
            }

            move = getMove(feedForwardNetwork, move);
            updateSnakePosition(move);
            mouseMove = updateGameState();

            double distance = mouse.distance(snake.get(0));
            if (mouseMove)
                score += 10;
            else
            if (distance < prevDistance)
                score += 1.0;
            else
                score -= 0.5;

            prevDistance = distance;
        }

        return score;
    }

    private ArrayList<IntTuple> initObstacles(int obstacletType)
    {
        ArrayList<IntTuple> obstacles = new ArrayList<>();

        if (obstacletType == 1)
        {
            int x = mapSize.x / 2;
            int y = mapSize.y / 2;

            for (int k = x - 3; k < x + 3; k++)
                for (int j = y - 3; j < y + 3; j++)
                    obstacles.add(new IntTuple(k, j));
        }
        else if (obstacletType == 2)
        {
            MersenneTwister mersenneTwister = new MersenneTwister(System.nanoTime());
            for (int k = 0; k < 10; k++)
                obstacles.add(new IntTuple(mersenneTwister.nextInt(mapSize.x-3)+3, mersenneTwister.nextInt(mapSize.y-3)+3));
        }

        return obstacles;
    }

    private IntTuple getMove(FeedForwardNetwork feedForwardNetwork, IntTuple prevMove)
    {

        double[] output = buildRelativeInputs(feedForwardNetwork, prevMove);

        int max = 0;
        for (int k = 0; k < output.length; k++)
            if (output[k] > output[max])
                max = k;

        IntTuple move = null;
        if (max == 0)
            move = new IntTuple(0, -1);
        else
        if (max == 1)
            move = new IntTuple(0, 1);
        else
        if (max == 2)
            move = new IntTuple(-1, 0);
        else
        if (max == 3)
            move = new IntTuple(1, 0);

        if (move.equals(new IntTuple(-1 * prevMove.x, -1 * prevMove.y)))
            move = prevMove;

        return move;
    }

    private double[] buildRelativeInputs(FeedForwardNetwork feedForwardNetwork, IntTuple prevMove)
    {
        IntTuple head = snake.get(0);

        double[] input = new double[feedForwardNetwork.inputLayerSize];

        IntTuple top = new IntTuple(head.x, head.y - 1);
        IntTuple left = new IntTuple(head.x - 1, head.y);
        IntTuple right = new IntTuple(head.x + 1, head.y);
        IntTuple bottom = new IntTuple(head.x, head.y + 1);

        input[0] = snake.contains(top) || outOfBound(top) ? 0 : 1;
        input[1] = snake.contains(bottom) || outOfBound(bottom) ? 0 : 1;
        input[2] = snake.contains(left) || outOfBound(left) ? 0 : 1;
        input[3] = snake.contains(right) || outOfBound(right) ? 0 : 1;

        input[4] = mouse.y <= head.y ? 1 : 0;
        input[5] = mouse.y > head.y ? 1 : 0;
        input[6] = mouse.x <= head.x ? 1 : 0;
        input[7] = mouse.x > head.x ? 1 : 0;

//        input[8] = (mouse.distance(head) / mapSize.distance(new IntTuple(0, 0))) * 1;
        // input[8] -= 1;
        // input[8] *= -1;

        feedForwardNetwork.clearLayers();
        feedForwardNetwork.feedForward(input);

        return feedForwardNetwork.outputLayer;
    }

    private boolean isGameComplete(boolean evaluationMode)
    {
        if (!evaluationMode)
            if (lastEatenCount >= 75)
                return true;

        IntTuple head = snake.get(0);
        for (int k = 1; k < snake.size(); k++)
            if (head.equals(snake.get(k)))
                return true;

        if (outOfBound(snake.get(0)))
            return true;

        return false;
    }

    private boolean outOfBound(IntTuple intTuple)
    {
        if (intTuple.x < 0 || intTuple.x >= mapSize.x)
            return true;

        if (intTuple.y < 0 || intTuple.y >= mapSize.y)
            return true;

        if (obstacles.contains(intTuple))
            return true;

        return false;
    }

    private void updateSnakePosition(IntTuple velocity)
    {
        for (int k = snake.size() - 1; k > 0 ; k--)
        {
            IntTuple prev = snake.get(k - 1);
            IntTuple curr = snake.get(k);

            curr.x = prev.x;
            curr.y = prev.y;
        }

        IntTuple head = snake.get(0);
        head.x += velocity.x;
        head.y += velocity.y;
    }

    private boolean updateGameState()
    {
        boolean mouseMove = false;

        IntTuple head = snake.get(0);
        if (head.equals(mouse))
        {
            IntTuple tail = snake.get(snake.size() - 1);
            snake.add(new IntTuple(tail));

            moveMouse();
            mouseMove = true;

            lastEatenCount = 0;
        } else
        {
            lastEatenCount++;
        }

        return mouseMove;
    }

    private void moveMouse()
    {
        MersenneTwister mersenneTwister = new MersenneTwister(System.nanoTime());

        do
        {
            mouse.x = mersenneTwister.nextInt(mapSize.x);
            mouse.y = mersenneTwister.nextInt(mapSize.y);
        }
        while(snake.contains(mouse) || outOfBound(mouse));
    }

    private ArrayList<IntTuple> initSnake()
    {
        ArrayList<IntTuple> snake = new ArrayList<>();

        snake.add(new IntTuple(0, 2));
        snake.add(new IntTuple(0, 1));

        return snake;
    }

    private void printMap()
    {
        System.out.println("\033[2J");
        ArrayList<StringBuilder> rows = new ArrayList<>();

        for (int k = 0; k < mapSize.y; k++)
        {
            StringBuilder s = new StringBuilder();
            for (int j = 0; j < mapSize.x; j++)
                s.append("-");

            rows.add(s);
        }

        for (IntTuple tuple : snake)
            rows.get(tuple.y).setCharAt(tuple.x, '*');

        rows.get(mouse.y).setCharAt(mouse.x, '#');

        for (IntTuple tuple : obstacles)
            rows.get(tuple.y).setCharAt(tuple.x, '+');

        for (StringBuilder row : rows)
            System.out.println(row);
    }

    public class IntTuple
    {
        int x;
        int y;

        IntTuple(int x, int y)
        {
            this.x = x;
            this.y = y;
        }

        IntTuple(IntTuple intTuple)
        {
            this(intTuple.x, intTuple.y);
        }

        public double distance(IntTuple intTuple)
        {
            return Math.sqrt(Math.pow(x - intTuple.x, 2) + Math.pow(y - intTuple.y, 2));
        }

        @Override
        public boolean equals(Object object)
        {
            if (object.getClass() == this.getClass())
            {
                IntTuple intTuple = (IntTuple) object;
                return x == intTuple.x && y == intTuple.y;
            }

            return false;
        }

        public IntTuple rotateAntiClockwise()
        {
            return new IntTuple(y, -1 * x);
        }

        public IntTuple rotateClockwise()
        {
            return new IntTuple(-1 * y, x);
        }
    }
}

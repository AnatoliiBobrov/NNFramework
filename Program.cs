namespace NNFramework
{
    using System.Diagnostics;

    /// <summary>
    /// Provides exceptions of this module
    /// </summary>
    public class NNException : Exception
    {
        /// <summary>
        /// Create new exception with message from object
        /// </summary>
        /// <param name="message">either object that has ToString() method</param>
        public NNException(object message) : base(message.ToString())
        {
        }

        /// <summary>
        /// Create new exception with message from object and inner exception
        /// </summary>
        /// <param name="message">either object that has ToString() method</param>
        /// <param name="innerException">inner exception</param>
        public NNException(object message, Exception? innerException) : base(message.ToString(), innerException)
        {
        }
    }

    /// <summary>
    /// Implement random generator
    /// </summary>
    public static class RandomGenerator
    {
        static Random Random = new();

        /// <summary>
        /// Returns next pseudorandom number from -0.5 including to 0.5
        /// </summary>
        /// <returns>decimal from -0.5 including to 0.5</returns>
        public static decimal Next() => (decimal)(Random.NextDouble() - 0.5);

        /// <summary>
        /// Set seed for random generator. Use it before creation of NN
        /// </summary>
        /// <param name="seed">integer number, seed</param>
        public static void SetSeed(int seed) => Random = new Random(seed);
    }

    /// <summary>
    /// Is not ready right now -----------------------
    /// </summary>
    public static class Criterion
    {
        /// <summary>
        /// Small incrementing of argument for calculation of derived function in point
        /// </summary>
        private static decimal _delta = 0.00001M;

        /// <summary>
        /// Returns value of quadratic error
        /// </summary>
        /// <returns></returns>
        public static decimal QuadraticError(decimal predict, decimal perfect)
        {
            var err = predict - perfect;
            err *= err;
            return err / 2;
        }

        /// <summary>
        /// Returns value of quadratic error
        /// </summary>
        /// <returns></returns>
        public static decimal GetQuadraticErrorDerivative(decimal currentValue, decimal currentError) =>
            currentError * currentValue * (1 - currentValue);
    }

    /// <summary>
    /// For future ----------------------
    /// </summary>
    public class Optimizer(decimal lr)
    {
        public decimal LearningRate = lr;
    }

    /// <summary>
    /// Value but reference type
    /// </summary>
    public class Variable
    {
        /// <summary>
        /// Current value of variable
        /// </summary>
        public decimal Value;

        /// <summary>
        /// Create new variable with random value from RandomGenerator.Next()
        /// </summary>
        /// <param name="value">current value</param>
        public Variable() => Value = RandomGenerator.Next();

        /// <summary>
        /// Create new variable with current value
        /// </summary>
        /// <param name="value">current value</param>
        public Variable(decimal value) => Value = value;

    }

    /// <summary>
    /// Provides methods for neural connections
    /// </summary>
    public class Connection
    {
        /// <summary>
        /// Current weight
        /// </summary>
        public Variable Weight;

        // Input neuron of current connection
        private Neuron _input;
        // Output neuron of current connection
        private Neuron _output;


        /// <summary>
        /// Initialize connection between neurons
        /// </summary>
        /// <param name="input">Input neuron</param>
        /// <param name="output">Output neuron</param>
        /// <param name="value">weight of connection</param>
        private void Initialize(Neuron input, Neuron output, Variable value)
        {
            ArgumentNullException.ThrowIfNull(input, nameof(input));
            ArgumentNullException.ThrowIfNull(output, nameof(output));
            ArgumentNullException.ThrowIfNull(value, nameof(value));
            _input = input;
            _output = output;
            Weight = value;
        }

        /// <summary>
        /// Create new connection between neurons
        /// </summary>
        /// <param name="input">Input neuron</param>
        /// <param name="output">Output neuron</param>
        public Connection(Neuron input, Neuron output) =>
            Initialize(input, output, new Variable(RandomGenerator.Next()));

        /// <summary>
        /// Create new connection between neurons
        /// </summary>
        /// <param name="input">Input neuron</param>
        /// <param name="output">Output neuron</param>
        /// <param name="value">weight of connection</param>
        public Connection(Neuron input, Neuron output, Variable value) =>
            Initialize(input, output, value);

        /// <summary>
        /// Correct weight
        /// </summary>
        /// <param name="lr">learning rate</param>
        /// <param name="derivative">derive of neuron's activation function</param>
        public void UpdateWeight(decimal lr, decimal derivative) =>
            Weight.Value -= lr * derivative * _output.Output;

        /// <summary>
        /// Add value to next neuron
        /// </summary>
        public void GoForward() => _output.Input += _input.Output * Weight.Value;

        /// <summary>
        /// Get error of output neuron
        /// </summary>
        public decimal GetError() => Weight.Value * _output.Error;
    }

    /// <summary>
    /// Provides methods for neurons
    /// </summary>
    public class Neuron
    {
        /// <summary>
        /// Current input value of neuron
        /// </summary>
        public decimal Input = 0;

        /// <summary>
        /// Current derivative of neuron
        /// </summary>
        public decimal Derivative = 0;

        /// <summary>
        /// Current output value of neuron
        /// </summary>
        public decimal Output = 0;

        /// <summary>
        /// Current error of neuron
        /// </summary>
        public decimal Error = 0;

        /// <summary>
        /// Input connections of neurons
        /// </summary>
        public List<Connection> InputConnections = [];

        /// <summary>
        /// Output connections of neurons
        /// </summary>
        public List<Connection> OutputConnections = [];

        /// <summary>
        /// Get values to next layer
        /// </summary>
        public virtual void GoForward()
        {
            if (OutputConnections.Count != 0)
                foreach (Connection i in OutputConnections)
                    i.GoForward();
        }

        /// <summary>
        /// Calculate an output value
        /// </summary>
        public virtual void Activation()
        {
            if (InputConnections.Count != 0)
            {
                Output = (decimal)(1 / (1 + Math.Exp(decimal.ToDouble(-1 * Input))));
            }
            else
            {
                Output = Input;
            }
            if (OutputConnections.Count != 0) GoForward();
        }

        /// <summary>
        /// Add output connection
        /// </summary>
        /// <param name="connection">connection from this neuron</param>
        public virtual void AddOutputConnection(Connection connection)
        {
            ArgumentNullException.ThrowIfNull(connection, nameof(connection));
            OutputConnections.Add(connection);
        }

        /// <summary>
        /// Add input connection
        /// </summary>
        /// <param name="connection">connection to this neuron</param>
        public virtual void AddInputConnection(Connection connection)
        {
            ArgumentNullException.ThrowIfNull(connection, nameof(connection));
            InputConnections.Add(connection);
        }

        /// <summary>
        /// Set errors
        /// </summary>
        public virtual void SetErrors()
        {
            if (OutputConnections.Count == 0)
                throw new NNException("This neurons has not output connections");

            foreach (Connection i in OutputConnections)
                Error += i.GetError();
        }

        /// <summary>
        /// Set derivative
        /// </summary>
        /// <returns></returns>
        public virtual void SetDerivative() => Derivative = Error * Output * (1 - Output);

        /// <summary>
        /// Update weights
        /// </summary>
        /// <param name="net">Current NN which possess current Neuron</param>
        public virtual void UpdateWeights(Net net)
        {
            SetDerivative();
            var lr = net.LearningRate;
            foreach (Connection i in InputConnections)
            {
                i.UpdateWeight(lr, Derivative);
            }
        }
    }


    /// <summary>
    /// Provides methods for convolutional neurons
    /// </summary>
    public class ConvolutionalNeuron : Neuron
    {
        public override void UpdateWeights(Net net)
        {
            SetDerivative();
            var lr = net.LearningRate / InputConnections.Count;
            foreach (Connection i in InputConnections)
            {
                i.UpdateWeight(lr, Derivative);
            }
        }
    }

    /// <summary>
    /// Provides methods for layer of fully connected neurons
    /// </summary>
    public class Layer
    {
        /// <summary>
        /// Array of all weights
        /// </summary>
        public Variable[] Weights;

        /// <summary>
        /// Size of array of neurons
        /// </summary>
        public int[] Size;

        /// <summary>
        /// Current output of layer
        /// </summary>
        public Variable[]? Outputs;

        /// <summary>
        /// Array of neurons
        /// </summary>
        public Neuron[]? Neurons;

        /// <summary>
        /// Set random weight of connections
        /// </summary>
        /// <param name="nextLayer">neighbour layer</param>
        public virtual void InitializeConnections(Layer nextLayer) { }

        /// <summary>
        /// Set weights
        /// </summary>
        /// <param name="weights">array of weights</param>
        public virtual void SetWeights(decimal[] weights)
        {
            ArgumentNullException.ThrowIfNull(weights, nameof(weights));

            if (weights.Length != Weights.Length)
            {
                throw new NNException("Weights' size is not equal count of Layer.Weights.Length");
            }

            for (int i = 0; i < weights.Length; i++)
                Weights[i].Value = weights[i];
        }

        /// <summary>
        /// Set inputs to neurons
        /// </summary>
        /// <param name="input">input data</param>
        public virtual void SetInputs(decimal[] input)
        {
            ArgumentNullException.ThrowIfNull(input, nameof(input));
            if (input.Length != Neurons.Length)
                throw new NNException("Input's size is not equal count of neurons");

            for (int i = 0; i < input.Length; i++)
                Neurons[i].Input = input[i];
        }

        /// <summary>
        /// Get output of layer
        /// </summary>
        /// <returns>array of output values</returns>
        public virtual decimal[] GetOutputs()
        {
            var result = new decimal[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                result[i] = Neurons[i].Output;
            }
            return result;
        }

        /// <summary>
        /// Clear inputs of neurons
        /// </summary>
        public virtual void ClearInputs()
        {
            foreach (Neuron i in Neurons) i.Input = 0M;
        }

        /// <summary>
        /// Set errors of neurons
        /// </summary>
        /// <param name="right">Required outputs of layer</param>
        public virtual void SetErrors(decimal[] right)
        {
            ArgumentNullException.ThrowIfNull(right, nameof(right));
            if (right.Length != Neurons.Length)
                throw new NNException("Parameter 'right' size is not equal count of neurons");

            for (int i = 0; i < Neurons.Length; i++)
                Neurons[i].Error = Neurons[i].Output - right[i];
        }

        /// <summary>
        /// Set errors of neurons
        /// </summary>
        public virtual void SetErrors()
        {
            try
            {
                for (int i = 0; i < Size[0]; i++)
                {
                    Neurons[i].SetErrors();
                }
            }
            catch (Exception e)
            {
                throw new NNException("Can not handle this exception", e);
            }
        }

        /// <summary>
        /// Clear errors of neurons
        /// </summary>
        public virtual void ClearErrors()
        {
            foreach (Neuron i in Neurons) i.Error = 0M;
        }

        /// <summary>
        /// Activate layer
        /// </summary>
        public virtual void Activation()
        {
            foreach (Neuron i in Neurons) i.Activation();
        }

        /// <summary>
        /// Update weights
        /// </summary>
        /// <param name="net">Net whith contains that layer</param>
        public virtual void UpdateWeights(Net net)
        {
            foreach (Neuron i in Neurons) i.UpdateWeights(net);
        }
    }

    /// <summary>
    /// Provides methods for layer of fully connected neurons
    /// </summary>
    class LinearLayer : Layer
    {
        /// <summary>
        /// Create new layer of neurons
        /// </summary>
        /// <param name="size">count of neurons</param>
        public LinearLayer(int inputSize, int outputSize)
        {
            if (inputSize < 1)
                throw new NNException("Input size of neural network can not be less than 1");
            if (outputSize < 1)
                throw new NNException("Output size of neural network can not be less than 1");

            Size = [inputSize];
            Outputs = new Variable[outputSize];
            for (int i = 0; i < outputSize; i++) Outputs[i] = new Variable(0);
            Neurons = new Neuron[inputSize];
            for (int i = 0; i < inputSize; i++) Neurons[i] = new Neuron();
        }

        public override void InitializeConnections(Layer nextLayer)
        {
            ArgumentNullException.ThrowIfNull(nextLayer, nameof(nextLayer));
            int countOfWeights = Size[0] * nextLayer.Neurons.Length;
            Weights = new Variable[countOfWeights];
            for (int i = 0; i < countOfWeights; i++) Weights[i] = new Variable();

            for (int begin = 0; begin < Size[0]; begin++)
            {
                var beginNeuron = Neurons[begin];
                for (int end = 0; end < nextLayer.Neurons.Length; end++)
                {
                    var endNeuron = nextLayer.Neurons[end];
                    var connection = new Connection(beginNeuron, endNeuron, Weights[begin * nextLayer.Neurons.Length + end]);
                    beginNeuron.AddOutputConnection(connection);
                    endNeuron.AddInputConnection(connection);
                }
            }
        }
    }

    /// <summary>
    /// Provides methods for layer of fully connected neurons
    /// </summary>
    class ConvolutionLayer : Layer
    {
        /// <summary>
        /// total count of neurons
        /// </summary>
        private int _size;

        private int _outputRows; // 
        private int _outputColumns; // size of output layer
        private int _countInChannel; // count neurons in channel

        /// <summary>
        /// Stride of convolution
        /// </summary>
        public int Stride;

        /// <summary>
        /// Size of convolutional mask
        /// </summary>
        public int MaskSize;

        /// <summary>
        /// Zero padding of edges
        /// </summary>
        public int Padding;

        /// <summary>
        /// Count of mask per channel
        /// </summary>
        public int CountOfMasks;

        /// <summary>
        /// Create new convolutional layer
        /// </summary>
        /// <param name="countOfRows">count of rows</param>
        /// <param name="countOfColumns">count of columns</param>
        /// <param name="countOfChannels">count of input channels</param>
        /// <param name="maskSize">size of convolutional mask</param>
        /// <param name="countOfMasks">count of mask per channel</param>
        /// <param name="stride">stride of convolution</param>
        /// <param name="padding">zero padding of edges</param>
        public ConvolutionLayer(int countOfRows, int countOfColumns, int countOfChannels = 1, int maskSize = 3, int countOfMasks = 3, int stride = 1, int padding = 0)
        {
            if (countOfChannels < 1) throw new NNException("'countOfChannels' can not be less than 1");
            if (countOfRows < 1) throw new NNException("'countOfRows' can not be less than 1");
            if (countOfColumns < 1) throw new NNException("'countOfColumns' can not be less than 1");
            if (maskSize < 1) throw new NNException("'maskSize' can not be less than 1");
            if (countOfMasks < 1) throw new NNException("'countOfMasks' can not be less than 1");
            if (stride < 1) throw new NNException("'stride' can not be less than 1");
            if (padding < 0) throw new NNException("'stride' can not be less than 0");

            CountOfMasks = countOfMasks;
            MaskSize = maskSize;
            Stride = stride;
            Padding = padding;
            _size = countOfChannels * countOfRows * countOfColumns;
            _countInChannel = countOfRows * countOfColumns;
            Size = [countOfChannels, countOfRows, countOfColumns];
            Neurons = new Neuron[_size];
            for (int i = 0; i < _size; i++)
                Neurons[i] = new ConvolutionalNeuron();
        }

        public override void InitializeConnections(Layer nextLayer)
        {
            ArgumentNullException.ThrowIfNull(nextLayer, nameof(nextLayer));

            ///[countOfChannels, countOfRows, countOfColumns]
            var weightsCount = MaskSize * MaskSize * Size[0] * CountOfMasks;
            Weights = new Variable[weightsCount];
            int rowSteps = (int)Math.Ceiling((decimal)((2 * Padding + Size[1] - MaskSize + 1) / Stride));
            int columnSteps = (int)Math.Ceiling((decimal)((2 * Padding + Size[2] - MaskSize + 1) / Stride));
            for (int i = 0; i < weightsCount; i++) Weights[i] = new Variable(RandomGenerator.Next());
            for (int inputChannel = 0; inputChannel < Size[0]; inputChannel++)
            {
                for (int maskNumber = 0; maskNumber < CountOfMasks; maskNumber++)
                {
                    for (int row = 0; row < rowSteps; row++)
                    {
                        for (int column = 0; column < columnSteps; column++)
                        {
                            int cRow = row * Stride - Padding;
                            int cColumn = column * Stride - Padding;
                            int currentBNNumber = inputChannel * Size[1] * Size[2] + cRow * Size[2] + cColumn;
                            int currentENNumber = inputChannel * columnSteps * rowSteps + row * columnSteps + column;

                            Neuron currentENeuron = nextLayer.Neurons[currentENNumber];
                            int currentMNumber = (inputChannel * CountOfMasks + maskNumber) * MaskSize * MaskSize;
                            int weightPointer = currentMNumber;
                            //var connections = new Connection[mas];
                            for (int maskRow = 0; maskRow < MaskSize; maskRow++)
                            {
                                for (int maskColumn = 0; maskColumn < MaskSize; maskColumn++)
                                {
                                    if ((cRow + maskRow > -1) && (cColumn + maskColumn > -1) && (cRow + maskRow < Size[1]) && (cColumn + maskColumn < Size[2]))
                                    {
                                        int BNPointer = currentBNNumber + maskRow * Size[2] + maskColumn;
                                        Neuron currentBNeuron = Neurons[BNPointer];
                                        var connection = new Connection(currentBNeuron, currentENeuron, Weights[weightPointer]);
                                        currentBNeuron.AddOutputConnection(connection);
                                        currentENeuron.AddInputConnection(connection);
                                    }

                                    weightPointer++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }



    /// <summary>
    /// Provides methods for neural network
    /// </summary>
    public class Net
    {
        /// <summary>
        /// Learning rate --------------------------
        /// </summary>
        public decimal LearningRate;

        /// <summary>
        /// Current architeture of neural network
        /// </summary>
        public List<Layer> Layers;

        /// <summary>
        /// Create new instance of network
        /// </summary>
        /// <param name="layers">Layers of NN</param>
        public Net(params Layer[] layers)
        {
            Layers = new List<Layer>(layers);
            for (int i = 0; i < Layers.Count - 1; i++)
                Layers[i].InitializeConnections(Layers[i + 1]);
        }

        /// <summary>
        /// Activate net
        /// </summary>
        public void Activation()
        {
            foreach (Layer i in Layers) i.Activation();
        }

        /// <summary>
        /// Clear errors of net
        /// </summary>
        public void ClearErrors()
        {
            foreach (Layer i in Layers) i.ClearErrors();
        }

        /// <summary>
        /// Set inputs to first layer and clear rest of layers
        /// </summary>
        /// <param name="input">input value of neural network</param>
        private void SetInputs(decimal[] input)
        {
            try
            {
                Layers[0].SetInputs(input);
            }
            catch (Exception e)
            {
                throw new NNException("Can not handle this exception", e);
            }
            for (int i = 1; i < Layers.Count; i++)
            {
                Layers[i].ClearInputs();
            }
        }

        /// <summary>
        /// Set errors
        /// </summary>
        /// <param name="right">Expected outputs</param>
        public void SetErrors(decimal[] right)
        {
            try
            {
                Layers.Last().SetErrors(right);
            }
            catch (Exception e)
            {
                throw new NNException("Can not handle this exception", e);
            }

            for (int i = Layers.Count - 2; i > 0; i--)
            {
                Layers[i].SetErrors();
            }
        }

        /// <summary>
        /// Update weights
        /// </summary>
        public void UpdateWeights()
        {
            for (int i = Layers.Count - 1; i > 0; i--)
            {
                Layers[i].UpdateWeights(this);
            }
        }

        /// <summary>
        /// Return result of NN
        /// </summary>
        /// <param name="input">Input values</param>
        /// <returns>result of NN</returns>
        public decimal[] Output(decimal[] input)
        {
            try
            {
                SetInputs(input);
            }
            catch (Exception e)
            {
                throw new NNException("Can not handle this exception", e);
            }

            Activation();
            return Layers.Last().GetOutputs();
        }

        /// <summary>
        /// Train net
        /// </summary>
        /// <param name="input">input of NN</param>
        /// <param name="right">expected values</param>
        public void Train(decimal[] input, decimal[] right)
        {
            try
            {
                SetInputs(input);
            }
            catch (Exception e)
            {
                throw new NNException("Can not handle this exception", e);
            }

            Activation();
            ClearErrors();

            try
            {
                SetErrors(right);
            }
            catch (Exception e)
            {
                throw new NNException("Can not handle this exception", e);
            }

            UpdateWeights();
        }
    }

    public class MyNet
    {
        public static void Main()
        {
            string Value = "";
            var l1 = new ConvolutionLayer(6, 6, 1, 3, 1, 2, 1);
            var l2 = new LinearLayer(9, 2);
            var l3 = new LinearLayer(2, 2);
            var net = new Net(l1, l2, l3)
            {
                LearningRate = 1
            };
            l1.SetWeights([0.1M, 0.2M, 0.3M, 0.4M, 0.5M, 0.6M, 0.7M, 0.8M, 0.9M]);

            var input = new decimal[] {
            0.2M, 0.3M, 0.4M, 0.5M, 0.6M, 0.7M,
            0.3M, 0.4M, 0.7M, 0.1M, 0.9M, 0.8M,
            0.9M, 0.2M, 0.6M, 0.2M, 0.8M, 0.4M,
            0.1M, 0.8M, 0.2M, 0.1M, 0.6M, 0.8M,
            0.1M, 0.5M, 0.1M, 0.4M, 0.6M, 0.5M,
            0.9M, 0.2M, 0.4M, 0.9M, 0.6M, 0.9M
            };
            var right = new decimal[] { 0.5M, 0.9M };

            var stopw = new Stopwatch();
            stopw.Start();
            for (int i = 0; i < 10; i++)

                net.Train(input, right);

            stopw.Stop();
            Console.WriteLine(stopw.Elapsed.ToString());
            var output = net.Output(input);
            net.ClearErrors();
            net.SetErrors(right);

            for (int i = 0; i < output.Length; i++)
            {
                Value += ' ' + output[i].ToString();
            }

            Value += '?';

            for (int i = 0; i < l2.Neurons.Length; i++)
            {
                Value += ' ' + l2.Neurons[i].Input.ToString();
            }

            Value += '?';

            for (int i = 0; i < l1.Neurons[0].OutputConnections.Count; i++)
            {
                Value += ' ' + l1.Neurons[0].OutputConnections[i].Weight.Value.ToString();
            }
            Console.WriteLine(Value);
            
        }

    }

    public class MyNet_
    {

        public static void Main111()
        {
            string Value = "";
            var l1 = new LinearLayer(2, 2);
            var l2 = new LinearLayer(2, 2);
            var l3 = new LinearLayer(2, 2);
            var net = new Net(l1, l2, l3)
            {
                LearningRate = 1
            };
            l1.SetWeights([0.5M, 0.7M, 0.4M, 0.6M]);
            l2.SetWeights([0.3M, 0.6M, 0.4M, 0.5M]);

            var input = new decimal[] { 1M, 0.5M };
            var right = new decimal[] { 0.5M, 0.9M };

            for (int i = 0; i < 100; i++)

                net.Train(input, right);

            var output = net.Output(input);
            net.ClearErrors();
            net.SetErrors(right);
            for (int i = 0; i < output.Length; i++)
            {
                Value += ' ' + output[i].ToString();
            }

            Value += '?';

            for (int i = 0; i < l2.Neurons.Length; i++)
            {
                Value += ' ' + l2.Neurons[i].Error.ToString();
            }

            for (int i = 0; i < l2.Neurons[0].OutputConnections.Count; i++)
            {
                Value += ' ' + l2.Neurons[0].OutputConnections[i].Weight.ToString();
            }
            Console.WriteLine(Value);
        }
    }
}
namespace NNFramework
{
    using System.Data.Common;
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
    /// Provides methods for logistical function
    /// </summary>
    public class Activation
    {
        /// <summary>
        /// Return value calculated from input argument
        /// </summary>
        /// <param name="input">argument of function</param>
        /// <returns></returns>
        public virtual decimal Calculate(decimal input)
        {
            return 0;
        }

        /// <summary>
        /// Return value of derived function in point
        /// </summary>
        /// <param name="point">point of function</param>
        /// <returns></returns>
        public virtual decimal Derived(decimal point)
        {
            return 0;
        }
    }

    /// <summary>
    /// Sigmoid activation function
    /// </summary>
    public class SigmoidActivation : Activation
    {
        public override decimal Calculate(decimal input) =>
            (decimal)(1 / (1 + Math.Exp(decimal.ToDouble(-1 * input))));

        /// <summary>
        /// Return value of derived function in point by current value of function
        /// </summary>
        /// <param name="CurrentValue">current value of function</param>
        /// <returns></returns>
        public override decimal Derived(decimal CurrentValue) =>
            CurrentValue * (1 - CurrentValue);
    }

    /// <summary>
    /// Leaky ReLU activation function
    /// </summary>
    public class LeakyReLUActivation : Activation
    {
        /// <summary>
        /// Value of negative rate
        /// </summary>
        private decimal _value;

        /// <summary>
        /// Create new instance of function with negative rate
        /// </summary>
        /// <param name="value">negative rate</param>
        public LeakyReLUActivation(decimal value) => _value = value;

        public override decimal Calculate(decimal input)
        {
            if (input > 0) return input;
            else return _value * input;
        }  
           
        public override decimal Derived(decimal point)
        {
            if (point > 0) return 1;
            else return _value;
        }
    }

    /// <summary>
    /// ReLU activation function
    /// </summary>
    public class ReLUActivation : Activation
    {
        public override decimal Calculate(decimal input)
        {
            if (input > 0) return input;
            else return 0;
        }

        public override decimal Derived(decimal point)
        {
            if (point > 0) return 1;
            else return 0;
        }
    }

    /// <summary>
    /// Provides learning of NN
    /// </summary>
    public class Optimizer
    {
        /// <summary>
        /// Learning rate
        /// </summary>
        public decimal LearningRate;

        /// <summary>
        /// Calculate error of prediction
        /// </summary>
        /// <param name="output">output value of neuron</param>
        /// <param name="right">right value</param>
        /// <returns></returns>
        public virtual decimal Error(decimal output, decimal right)
        {
            return 0;
        }

        /// <summary>
        /// Calculate criterion value of prediction
        /// </summary>
        /// <param name="output">output value of neuron</param>
        /// <param name="right">right value</param>
        /// <returns></returns>
        public virtual decimal CriterionValue(decimal output, decimal right)
        {
            return 0;
        }

        /// <summary>
        /// Calculate criterion value of prediction
        /// </summary>
        /// <param name="error">error of neuron</param>
        /// <returns></returns>
        public virtual decimal Derivative(decimal error)
        {
            return 0;
        }

        /// <summary>
        /// Create instance of optimizer
        /// </summary>
        public Optimizer(decimal lr) => LearningRate = lr;
    }

    /// <summary>
    /// Optimize by least square's method
    /// </summary>
    public class LeastSquareOptimizer : Optimizer
    {
        public override decimal Error(decimal output, decimal right) => output - right;

        public override decimal CriterionValue(decimal output, decimal right)
        {
            var error = output - right;
            error *= error;
            error /= 2;
            return error;
        }

        public override decimal Derivative(decimal error) => error;

        /// <summary>
        /// Create instance of optimizer
        /// </summary>
        public LeastSquareOptimizer(decimal lr) : base(lr)
        {

        }
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
        /// The net in witch this neuron is located
        /// </summary>
        public Net OwnerNet;

        /// <summary>
        /// Logistical function of neuron
        /// </summary>
        public Activation ActivationFunction;

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
        /// Create instance ot neuron 
        /// </summary>
        /// <param name="function">Logistic function</param>
        public Neuron(Activation function) =>
            ActivationFunction = function;

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
                Output = ActivationFunction.Calculate(Input);
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
        public virtual void SetDerivative() => Derivative = OwnerNet.Optimizer.Derivative(Error) * ActivationFunction.Derived(Output);

        /// <summary>
        /// Update weights
        /// </summary>
        /// <param name="net">Current NN which possess current Neuron</param>
        public virtual void UpdateWeights()
        {
            SetDerivative();
            var lr = OwnerNet.Optimizer.LearningRate;
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
        /// <summary>
        /// Create instance ot neuron 
        /// </summary>
        /// <param name="function">Logistic function</param>
        public ConvolutionalNeuron(Activation function) : base(function)
        {
        }

        public override void UpdateWeights()
        {
            SetDerivative();
            var lr = OwnerNet.Optimizer.LearningRate / InputConnections.Count;
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
        /// Logistical function of neuron
        /// </summary>
        public Activation ActivationFunction;

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
                throw new NNException("Weights' size is not equal count of Layer.Weights.Length");

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
            foreach (Neuron i in Neurons) i.UpdateWeights();
        }

        public virtual void SetOwnerNet(Net ownerNet)
        {
            foreach (Neuron neuron in Neurons) neuron.OwnerNet = ownerNet;
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
        /// <param name="function">activation function of all neurons of layer</param>
        /// <param name="inputSize">count of neurons</param>
        /// <param name="outputSize">count of neurons in next layer</param>
        /// <param name="function">logistic function</param>
        /// <exception cref="NNException"></exception>
        public LinearLayer(Activation function, int inputSize, int outputSize)
        {
            ArgumentNullException.ThrowIfNull(function, nameof(function));
            if (inputSize < 1)
                throw new NNException("Input size of neural network can not be less than 1");
            if (outputSize < 1)
                throw new NNException("Output size of neural network can not be less than 1");

            Size = [inputSize];
            Outputs = new Variable[outputSize];
            for (int i = 0; i < outputSize; i++) Outputs[i] = new Variable(0);
            Neurons = new Neuron[inputSize];
            for (int i = 0; i < inputSize; i++) Neurons[i] = new Neuron(function);
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
    /// Provides methods for convolutional layer
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
        /// <param name="function">activation function of all neurons of layer</param>
        /// <param name="countOfRows">count of rows</param>
        /// <param name="countOfColumns">count of columns</param>
        /// <param name="countOfChannels">count of input channels</param>
        /// <param name="maskSize">size of convolutional mask</param>
        /// <param name="countOfMasks">count of mask per channel</param>
        /// <param name="stride">stride of convolution</param>
        /// <param name="padding">zero padding of edges</param>
        public ConvolutionLayer(Activation function, int countOfRows, int countOfColumns, int countOfChannels = 1, int maskSize = 3, int countOfMasks = 3, int stride = 1, int padding = 0)
        {
            ArgumentNullException.ThrowIfNull(function, nameof(function));
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
                Neurons[i] = new ConvolutionalNeuron(function);
        }

        public override void InitializeConnections(Layer nextLayer)
        {
            ArgumentNullException.ThrowIfNull(nextLayer, nameof(nextLayer));

            ///[countOfChannels, countOfRows, countOfColumns]
            var weightsCount = MaskSize * MaskSize * Size[0] * CountOfMasks;
            Weights = new Variable[weightsCount];
            int rowSteps = (int)Math.Ceiling((decimal)((2 * Padding + Size[1] - MaskSize + 1) / Stride));
            int columnSteps = (int)Math.Ceiling((decimal)((2 * Padding + Size[2] - MaskSize + 1) / Stride));
            for (int i = 0; i < weightsCount; i++) Weights[i] = new Variable();

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
                            int currentBNNumber = inputChannel * _countInChannel + cRow * Size[2] + cColumn;
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
    /// Provides methods for maximun pooling layer
    /// </summary>
    class MaxPoolingLayer : Layer
    {
        /// <summary>
        /// total count of neurons
        /// </summary>
        private int _size;

        /// <summary>
        /// Steps of iteration of row
        /// </summary>
        private int _rowSteps;

        /// <summary>
        /// Steps of iteration of column
        /// </summary>
        private int _columnSteps;

        /// <summary>
        /// Count neurons in channel
        /// </summary>
        private int _countInChannel;

        /// <summary>
        /// Next layer
        /// </summary>
        private Layer _nextLayer;

        /// <summary>
        /// Size of pooling area
        /// </summary>
        public int PoolingSize;

        /// <summary>
        /// Create instance of max pooling layer
        /// </summary>
        /// <param name="function">activation function of all neurons of layer</param>
        /// <param name="countOfRows">count of rows</param>
        /// <param name="countOfColumns">count of columns</param>
        /// <param name="countOfChannels">count of input channels</param>
        /// <param name="poolingArea">size of cide of pooling square</param>
        /// <exception cref="NNException"></exception>
        public MaxPoolingLayer(Activation function, int countOfRows, int countOfColumns, int countOfChannels = 1, int poolingArea = 2)
        {
            ArgumentNullException.ThrowIfNull(function, nameof(function));
            if (countOfChannels < 1) throw new NNException("'countOfChannels' can not be less than 1");
            if (countOfRows < 1) throw new NNException("'countOfRows' can not be less than 1");
            if (countOfColumns < 1) throw new NNException("'countOfColumns' can not be less than 1");
            if (poolingArea < 1) throw new NNException("'poolingArea' can not be less than 1");

            _countInChannel = countOfRows * countOfColumns;
            _columnSteps = (int)Math.Ceiling((decimal)(Size[2] / PoolingSize));
            _rowSteps = (int)Math.Ceiling((decimal)(Size[1] / PoolingSize));
            _size = countOfChannels * countOfRows * countOfColumns;
            Size = [countOfChannels, countOfRows, countOfColumns];
            PoolingSize = poolingArea;
            Neurons = new Neuron[_size];
            for (int i = 0; i < _size; i++)
                Neurons[i] = new Neuron(function);
        }

        public override void InitializeConnections(Layer nextLayer)
        {
            ArgumentNullException.ThrowIfNull(nextLayer, nameof(nextLayer));

            _nextLayer = nextLayer;
            var weightsCount = Size[0] * _rowSteps * _columnSteps;
            Weights = new Variable[weightsCount];
            for (int i = 0; i < weightsCount; i++) Weights[i] = new Variable();
        }

        public override void Activation()
        {
            int currentBNNumber = 0;
            int currentENNumber = 0;
            //[countOfChannels, countOfRows, countOfColumns]
            for (int inputChannel = 0; inputChannel < Size[0]; inputChannel++)
            {
                for (int row = 0; row < _rowSteps; row++)
                {
                    for (int column = 0; column < _columnSteps; column++)
                    {
                        currentBNNumber += (row * Size[2] + column) * PoolingSize;
                        Neuron maxValue = Neurons[currentBNNumber];
                        Neuron currentENeuron = _nextLayer.Neurons[currentENNumber];
                        //var connections = new Connection[mas];
                        for (int poolRow = 0; poolRow < PoolingSize; poolRow++)
                        {
                            for (int poolColumn = 0; poolColumn < PoolingSize; poolColumn++)
                            {
                                if ((row * PoolingSize + poolRow < Size[1]) && (column * PoolingSize + poolColumn < Size[2]))
                                {
                                    Neuron neuron = Neurons[currentBNNumber + poolRow * Size[2] + poolColumn];
                                    if (neuron.Output > maxValue.Output) maxValue = neuron;
                                }
                            }
                        }
                        currentENNumber += 1;
                    }
                }
                currentBNNumber = inputChannel * _countInChannel;
            }

            foreach (Neuron i in Neurons) i.Activation();
        }
    }


    /// <summary>
    /// Provides methods for neural network
    /// </summary>
    public class Net
    {
        /// <summary>
        /// Optimizer
        /// </summary>
        public Optimizer Optimizer;

        /// <summary>
        /// Current architeture of neural network
        /// </summary>
        public List<Layer> Layers;

        /// <summary>
        /// Create new instance of network
        /// </summary>
        /// <param name="layers">Layers of NN</param>
        public Net(Optimizer optimizer, params Layer[] layers)
        {
            ArgumentNullException.ThrowIfNull(optimizer, nameof(optimizer));

            this.Optimizer = optimizer;
            Layers = new List<Layer>(layers);
            foreach (Layer layer in layers) layer.SetOwnerNet(this);
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
            var sigm = new SigmoidActivation();
            var l1 = new ConvolutionLayer(sigm, 6, 6, 1, 3, 1, 2, 1);
            var l2 = new LinearLayer(sigm, 9, 2);
            var l3 = new LinearLayer(sigm, 2, 2);
            var opt = new LeastSquareOptimizer(1);
            var net = new Net(opt, l1, l2, l3);
            l1.SetWeights([0.1M, 0.2M, 0.3M, 0.4M, 0.5M, 0.6M, 0.7M, 0.8M, 0.9M]);
            l2.SetWeights(new decimal[] {0.2M, 0.3M, 0.4M, 0.5M, 0.6M, 0.7M, 0.3M, 0.4M, 0.7M, 0.1M, 0.9M, 0.8M, 0.9M, 0.2M, 0.6M, 0.2M, 0.8M, 0.4M} );
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
            var sigm = new SigmoidActivation();
            var l1 = new LinearLayer(sigm, 2, 2);
            var l2 = new LinearLayer(sigm, 2, 2);
            var l3 = new LinearLayer(sigm, 2, 2);
            var opt = new LeastSquareOptimizer(1);
            var net = new Net(opt, l1, l2, l3);
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
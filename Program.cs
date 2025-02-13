﻿using System;
using System.Diagnostics;
using System.Security.AccessControl;

namespace NNFramework
{
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
    /// Implenents methods for learning rate sheduler
    /// </summary>
    public class Sheduler
    {
        /// <summary>
        /// Optimizer with adjustable learning rate
        /// </summary>
        public Optimizer Optimizer;

        /// <summary>
        /// Current step in learning rate shedule
        /// </summary>
        public int CurrentStep = 0;

        /// <summary>
        /// Create new sheduler
        /// </summary>
        /// <param name="optimizer">optimizer with adjustable learning rate</param>
        public Sheduler(Optimizer optimizer)
        {
            ArgumentNullException.ThrowIfNull(optimizer, nameof(optimizer));
            Optimizer = optimizer; 
        }

        /// <summary>
        /// Made a one step in shedule
        /// </summary>
        public virtual void Step()
        {
        }
    }

    /// <summary>
    /// Implenents methods for exponential decay learning rate sheduler
    /// </summary>
    public class ExponentialSheduler : Sheduler
    {
        /// <summary>
        /// Divider of learning rate
        /// </summary>
        private decimal _gamma;

        /// <summary>
        /// Create new exponential sheduler dividing the learning rate of divider every step
        /// </summary>
        /// <param name="optimizer">optimizer with adjustable learning rate</param>
        /// <param name="gamma">divider of learning rate</param>
        public ExponentialSheduler(Optimizer optimizer, decimal gamma) : base (optimizer) =>
            _gamma = gamma;

        public override void Step()
        {
            CurrentStep++;
            Optimizer.LearningRate /= _gamma;
        }
    }

    /// <summary>
    /// Implenents methods for linear decay learning rate sheduler
    /// </summary>
    public class DiscreteSheduler : Sheduler
    {
        /// <summary>
        /// Shedule Dictionary<steps, learning rate>
        /// </summary>
        private Dictionary<int, decimal> _shedule;

        /// <summary>
        /// Shedule Dictionary<steps, learning rate>
        /// </summary>
        private int _currentNote = 0;

        /// <summary>
        /// Steps to change LR
        /// </summary>
        private int _targetStep = 0;

        /// <summary>
        /// Create new linear sheduler decaying the learning rate every step
        /// </summary>
        /// <param name="optimizer">optimizer with adjustable learning rate</param>
        /// <param name="shedule">shedule Dictionary<steps, learning rate></param>
        public DiscreteSheduler(Optimizer optimizer, Dictionary<int, decimal> shedule) : base(optimizer)
        {
            ArgumentNullException.ThrowIfNull(shedule, nameof(shedule));
            if (shedule.Count == 0)
                throw new NNException("Shedule must contain at least 1 note");

            _shedule = new Dictionary<int, decimal>(shedule.OrderBy(key => key.Key));
            _targetStep = _shedule.Keys.ElementAt(0);
        }

        public override void Step()
        {
            if (_currentNote < _shedule.Count)
            {
                if (CurrentStep >= _targetStep)
                {
                    Optimizer.LearningRate = _shedule.Values.ElementAt(_currentNote);
                    _currentNote++;
                    if (_currentNote < _shedule.Count)
                        _targetStep = _shedule.Keys.ElementAt(_currentNote);
                }
            }
            CurrentStep++;
        }
    }
    
    /// <summary>
    /// Implenents methods for linear decay learning rate sheduler
    /// </summary>
    public class LinearSheduler : Sheduler
    {
        /// <summary>
        /// Decay of learning rate
        /// </summary>
        private decimal _gamma;

        /// <summary>
        /// Create new linear sheduler decaying the learning rate every step
        /// </summary>
        /// <param name="optimizer">optimizer with adjustable learning rate</param>
        /// <param name="gamma">decay of learning rate</param>
        public LinearSheduler(Optimizer optimizer, decimal gamma) : base(optimizer) =>
            _gamma = gamma;

        public override void Step()
        {
            CurrentStep++;
            Optimizer.LearningRate -= _gamma;
        }
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

        /// <summary>
        /// Input neuron of current connection
        /// </summary>
        public Neuron Input;

        /// <summary>
        /// Output neuron of current connection
        /// </summary>
        public Neuron Output;


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
            Input = input;
            Output = output;
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
        public virtual void UpdateWeight(decimal lr, decimal derivative) =>
            Weight.Value -= lr * derivative * Output.Output;

        /// <summary>
        /// Add value to next neuron
        /// </summary>
        public void GoForward() => Output.Input += Input.Output * Weight.Value;

        /// <summary>
        /// Get error of output neuron
        /// </summary>
        public decimal GetError() => Weight.Value * Output.Error;
    }

    /// <summary>
    /// Provides methods for neural connections
    /// </summary>
    public class UnteachableConnection : Connection
    {
        /// <summary>
        /// Create new unteachable connection between neurons
        /// </summary>
        /// <param name="input">Input neuron</param>
        /// <param name="output">Output neuron</param>
        public UnteachableConnection(Neuron input, Neuron output) : base(input, output) { }


        /// <summary>
        /// Create new unteachable connection between neurons
        /// </summary>
        /// <param name="input">Input neuron</param>
        /// <param name="output">Output neuron</param>
        /// <param name="value">weight of connection</param>
        public UnteachableConnection(Neuron input, Neuron output, Variable value) : base(input, output, value) { }

        /// <summary>
        /// Create new unteachable connection between neurons
        /// </summary>
        /// <param name="input">Input neuron</param>
        /// <param name="output">Output neuron</param>
        /// <param name="value">weight of connection</param>
        public UnteachableConnection(Neuron input, Neuron output, decimal value) : base(input, output, new Variable(value)) { }

        public override void UpdateWeight(decimal lr, decimal derivative)
        {
        }
    }

    /// <summary>
    /// Provides methods for neurons
    /// </summary>
    public class Neuron
    {
        /// <summary>
        /// The layer in witch this neuron is located
        /// </summary>
        public Layer OwnerLayer;

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
            CalculateOutput();
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
            foreach (Connection i in OutputConnections)
                Error += i.GetError();
        }

        /// <summary>
        /// Set derivative
        /// </summary>
        /// <returns></returns>
        public virtual void SetDerivative() => Derivative = OwnerLayer.Optimizer.Derivative(Error) * ActivationFunction.Derived(Output);

        /// <summary>
        /// Update weights
        /// </summary>
        /// <param name="net">Current NN which possess current Neuron</param>
        public virtual void UpdateWeights()
        {
            SetDerivative();
            var lr = OwnerLayer.Optimizer.LearningRate;
            foreach (Connection i in InputConnections)
                i.UpdateWeight(lr, Derivative);
        }

        /// <summary>
        /// Calculate output value
        /// </summary>
        public virtual void CalculateOutput()
        {
            if (InputConnections.Count != 0)
            {
                Output = ActivationFunction.Calculate(Input);
            }
            else
            {
                Output = Input;
            }
        }
            
    }

    /// <summary>
    /// Provides methods for convolutional neurons
    /// </summary>
    public class ConvolutionalNeuron : Neuron
    {
        /// <summary>
        /// Devisioner of LR
        /// </summary>
        private int _devider;

        /// <summary>
        /// Create instance ot neuron 
        /// </summary>
        /// <param name="function">Logistic function</param>
        public ConvolutionalNeuron(Activation function, int devider) : base(function)
        {
            if (devider < 1) throw new NNException("Devider must be more then 1");
            _devider = devider;
        }

        public override void UpdateWeights()
        {
            SetDerivative();
            var lr = OwnerLayer.Optimizer.LearningRate / _devider;
            foreach (Connection i in InputConnections)
            {
                i.UpdateWeight(lr, Derivative);
            }
        }
    }

    /// <summary>
    /// Provides methods for max pooling neurons
    /// </summary>
    public class MaxPoolingNeuron : Neuron
    {
        /// <summary>
        /// Create instance ot neuron 
        /// </summary>
        /// <param name="function">Logistic function</param>
        public MaxPoolingNeuron(Activation function) : base(function)
        {
        }

        public override void Activation()
        {
            if (OutputConnections.Count != 0) GoForward();
        }
    }

    /// <summary>
    /// Provides methods for layer of fully connected neurons
    /// </summary>
    public class Layer
    {
        /// <summary>
        /// Count if output neurons
        /// </summary>
        public int CountOfOutputNeurons;

        /// <summary>
        /// Optimization algorithm for current layer
        /// </summary>
        public Optimizer Optimizer;

        /// <summary>
        /// Count if input neurons
        /// </summary>
        public int CountOfInputNeurons;

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
        /// Average loss
        /// </summary>
        public decimal Loss;

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
            if (Weights == null)
                throw new NNException("Layer has no weights");
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
                result[i] = Neurons[i].Output;
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
            Loss = 0M;
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].Error = Neurons[i].Output - right[i];
                Loss += Neurons[i].Error * Neurons[i].Error / 2;
            }
            Loss /= CountOfInputNeurons;
        }
        
        /// <summary>
        /// Set errors of neurons
        /// </summary>
        public virtual void SetErrors()
        {
            foreach (Neuron neuron in Neurons) neuron.SetErrors();
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
        /// <param name="net">Net which contains that layer</param>
        public virtual void UpdateWeights(Net net)
        {
            foreach (Neuron i in Neurons) i.UpdateWeights();
        }

        /// <summary>
        /// Set owner layer to neurons
        /// </summary>
        /// <param name="ownerLayer"></param>
        public virtual void SetOwnerLayer()
        {
            foreach (Neuron neuron in Neurons) neuron.OwnerLayer = this;
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
        /// <param name="optimizer">optimization algorithm of this layer</param>
        /// <param name="function">activation function of all neurons of layer</param>
        /// <param name="inputSize">count of neurons</param>
        /// <param name="outputSize">count of neurons in next layer</param>
        /// <param name="function">logistic function</param>
        /// <exception cref="NNException"></exception>
        public LinearLayer(Optimizer optimizer, Activation function, int inputSize, int outputSize)
        {
            ArgumentNullException.ThrowIfNull(optimizer, nameof(optimizer));
            ArgumentNullException.ThrowIfNull(function, nameof(function));
            if (inputSize < 1)
                throw new NNException("Input size of neural network can not be less than 1");
            if (outputSize < 1)
                throw new NNException("Output size of neural network can not be less than 1");

            Optimizer = optimizer;
            CountOfInputNeurons = inputSize;
            CountOfOutputNeurons = outputSize;
            Size = [inputSize];
            Outputs = new Variable[outputSize];
            for (int i = 0; i < outputSize; i++) Outputs[i] = new Variable(0);
            Neurons = new Neuron[inputSize];
            for (int i = 0; i < inputSize; i++) Neurons[i] = new Neuron(function);
        }

        public override void InitializeConnections(Layer nextLayer)
        {
            ArgumentNullException.ThrowIfNull(nextLayer, nameof(nextLayer));

            int countOfWeights = CountOfInputNeurons * CountOfOutputNeurons;
            Weights = new Variable[countOfWeights];
            for (int i = 0; i < countOfWeights; i++) Weights[i] = new Variable();

            for (int begin = 0; begin < Size[0]; begin++)
            {
                var beginNeuron = Neurons[begin];
                for (int end = 0; end < CountOfOutputNeurons; end++)
                {
                    var endNeuron = nextLayer.Neurons[end];
                    var connection = new Connection(beginNeuron, endNeuron, Weights[begin * CountOfOutputNeurons + end]);
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
        /// Count of output channels
        /// </summary>
        public int CountOfOutputChannels;

        /// <summary>
        /// Rows in output layer
        /// </summary>
        public int OutputRows;

        /// <summary>
        /// Columns in output layer
        /// </summary>
        public int OutputColumns;

        /// <summary>
        /// Count neurons in channel
        /// </summary>
        private int _countInChannel;

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
        /// <param name="optimizer">optimization algorithm of this layer</param>
        /// <param name="function">activation function of all neurons of layer</param>
        /// <param name="countOfRows">count of rows</param>
        /// <param name="countOfColumns">count of columns</param>
        /// <param name="countOfChannels">count of input channels</param>
        /// <param name="maskSize">size of convolutional mask</param>
        /// <param name="countOfMasks">count of mask per channel</param>
        /// <param name="stride">stride of convolution</param>
        /// <param name="padding">zero padding of edges</param>
        public ConvolutionLayer(Optimizer optimizer, Activation function, int countOfRows, int countOfColumns, int countOfChannels = 1, int maskSize = 3, int countOfMasks = 3, int stride = 1, int padding = 0)
        {
            ArgumentNullException.ThrowIfNull(optimizer, nameof(optimizer));
            ArgumentNullException.ThrowIfNull(function, nameof(function));

            Optimizer = optimizer;
            var sizes = GetOutputSizes(countOfRows, countOfColumns, countOfChannels, maskSize, countOfMasks, stride, padding);
            CountOfOutputChannels = sizes[0];
            OutputRows = sizes[1];
            OutputColumns = sizes[2];
            int countInOutputChannel = OutputRows * OutputColumns;
            CountOfMasks = countOfMasks;
            MaskSize = maskSize;
            Stride = stride;
            Padding = padding;
            CountOfInputNeurons = countOfChannels * countOfRows * countOfColumns;
            CountOfOutputNeurons = CountOfOutputChannels * OutputRows * OutputColumns;
            _countInChannel = countOfRows * countOfColumns;
            Size = [countOfChannels, countOfRows, countOfColumns];
            Neurons = new Neuron[CountOfInputNeurons];
            for (int i = 0; i < CountOfInputNeurons; i++)
                Neurons[i] = new ConvolutionalNeuron(function, countInOutputChannel);
        }

        /// <summary>
        /// Return output sharp of layer
        /// </summary>
        /// <param name="countOfRows">count of rows</param>
        /// <param name="countOfColumns">count of columns</param>
        /// <param name="countOfChannels">count of input channels</param>
        /// <param name="maskSize">size of convolutional mask</param>
        /// <param name="countOfMasks">count of mask per channel</param>
        /// <param name="stride">stride of convolution</param>
        /// <param name="padding">zero padding of edges</param>
        /// <returns>Array like [count of output channels, rows, columns]</returns>
        /// <exception cref="NNException"></exception>
        public static int[] GetOutputSizes(int countOfRows, int countOfColumns, int countOfChannels, int maskSize, int countOfMasks, int stride, int padding)
        {
            if (countOfChannels < 1) throw new NNException("'countOfChannels' can not be less than 1");
            if (countOfRows < 1) throw new NNException("'countOfRows' can not be less than 1");
            if (countOfColumns < 1) throw new NNException("'countOfColumns' can not be less than 1");
            if (maskSize < 1) throw new NNException("'maskSize' can not be less than 1");
            if (countOfMasks < 1) throw new NNException("'countOfMasks' can not be less than 1");
            if (stride < 1) throw new NNException("'stride' can not be less than 1");
            if (padding < 0) throw new NNException("'stride' can not be less than 0");

            int countOfOutputChannels = countOfChannels * countOfMasks;
            int rowSteps = (int)Math.Ceiling((decimal)((2 * padding + countOfRows - maskSize + 1) / stride));
            int columnSteps = (int)Math.Ceiling((decimal)((2 * padding + countOfColumns - maskSize + 1) / stride));
            return [countOfOutputChannels, rowSteps, columnSteps];
        }

        public override void InitializeConnections(Layer nextLayer)
        {
            ArgumentNullException.ThrowIfNull(nextLayer, nameof(nextLayer));

            ///[countOfChannels, countOfRows, countOfColumns]
            var weightsCount = MaskSize * MaskSize * Size[0] * CountOfMasks;
            Weights = new Variable[weightsCount];
            for (int i = 0; i < weightsCount; i++) Weights[i] = new Variable();

            for (int inputChannel = 0; inputChannel < Size[0]; inputChannel++)
            {
                for (int maskNumber = 0; maskNumber < CountOfMasks; maskNumber++)
                {
                    for (int row = 0; row < OutputRows; row++)
                    {
                        for (int column = 0; column < OutputColumns; column++)
                        {
                            int cRow = row * Stride - Padding;
                            int cColumn = column * Stride - Padding;
                            int currentBNNumber = inputChannel * _countInChannel + cRow * Size[2] + cColumn;
                            int currentENNumber = inputChannel * OutputColumns * OutputRows + row * OutputColumns + column;

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
    /// Provides methods for pooling layer
    /// </summary>
    class PoolingLayer : Layer
    {
        /// <summary>
        /// Rows in output layer
        /// </summary>
        public int OutputRows;

        /// <summary>
        /// Columns in output layer
        /// </summary>
        public int OutputColumns;

        /// <summary>
        /// Count neurons in channel
        /// </summary>
        public int CountInChannel;

        /// <summary>
        /// Next layer
        /// </summary>
        public Layer NextLayer;

        /// <summary>
        /// Size of pooling area
        /// </summary>
        public int PoolingSize;

        /// <summary>
        /// Create instance of max pooling layer
        /// </summary>
        /// <param name="optimizer">optimization algorithm of this layer</param>
        /// <param name="function">activation function of all neurons of layer</param>
        /// <param name="countOfRows">count of rows</param>
        /// <param name="countOfColumns">count of columns</param>
        /// <param name="countOfChannels">count of input channels</param>
        /// <param name="poolingArea">size of cide of pooling square</param>
        /// <exception cref="NNException"></exception>
        public PoolingLayer(Optimizer optimizer, Activation function, int countOfRows, int countOfColumns, int countOfChannels = 1, int poolingArea = 2)
        {
            ArgumentNullException.ThrowIfNull(optimizer, nameof(optimizer));
            ArgumentNullException.ThrowIfNull(function, nameof(function));
            var sizes = GetOutputSizes(countOfRows, countOfColumns, countOfChannels, poolingArea);

            Optimizer = optimizer;
            CountInChannel = countOfRows * countOfColumns;
            OutputRows = sizes[1];
            OutputColumns = sizes[2];
            CountOfInputNeurons = countOfChannels * countOfRows * countOfColumns;
            CountOfOutputNeurons = sizes[0] * OutputRows * OutputColumns;
            Size = [countOfChannels, countOfRows, countOfColumns];
            PoolingSize = poolingArea;
            Neurons = new Neuron[CountOfInputNeurons];
        }

        /// <summary>
        /// Return output sharp of layer and check output parameters
        /// </summary>
        /// <param name="countOfRows">count of rows</param>
        /// <param name="countOfColumns">count of columns</param>
        /// <param name="countOfChannels">count of input channels</param>
        /// <param name="poolingArea">size of cide of pooling square</param>
        /// <returns>Array like [count of output channels, rows, columns]</returns>
        /// <exception cref="NNException"></exception>
        public static int[] GetOutputSizes(int countOfRows, int countOfColumns, int countOfChannels, int poolingArea)
        {
            if (countOfChannels < 1) throw new NNException("'countOfChannels' can not be less than 1");
            if (countOfRows < 1) throw new NNException("'countOfRows' can not be less than 1");
            if (countOfColumns < 1) throw new NNException("'countOfColumns' can not be less than 1");
            if (poolingArea < 1) throw new NNException("'poolingArea' can not be less than 1");

            int countOfOutputChannels = countOfChannels;
            int columnSteps = (int)Math.Ceiling((decimal)countOfColumns / poolingArea);
            int rowSteps = (int)Math.Ceiling((decimal)countOfRows / poolingArea);
            return [countOfOutputChannels, rowSteps, columnSteps];
        }
    }

    /// <summary>
    /// Provides methods for maximun pooling layer
    /// </summary>
    class MaxPoolingLayer : PoolingLayer
    {
        /// <summary>
        /// Create instance of max pooling layer
        /// </summary>
        /// <param name="optimizer">optimization algorithm of this layer</param>
        /// <param name="function">activation function of all neurons of layer</param>
        /// <param name="countOfRows">count of rows</param>
        /// <param name="countOfColumns">count of columns</param>
        /// <param name="countOfChannels">count of input channels</param>
        /// <param name="poolingArea">size of cide of pooling square</param>
        /// <exception cref="NNException"></exception>
        public MaxPoolingLayer(Optimizer optimizer, Activation function, int countOfRows, int countOfColumns, int countOfChannels = 1, int poolingArea = 2) : base (optimizer, function, countOfRows, countOfColumns, countOfChannels, poolingArea)
        {
            for (int i = 0; i < CountOfInputNeurons; i++)
                Neurons[i] = new MaxPoolingNeuron(function);
        }

        public override void InitializeConnections(Layer nextLayer)
        {
            ArgumentNullException.ThrowIfNull(nextLayer, nameof(nextLayer));

            NextLayer = nextLayer;
            int cnHelp1 = 0;
            int currentENNumber = 0;
            //[countOfChannels, countOfRows, countOfColumns]
            for (int inputChannel = 0; inputChannel < Size[0]; inputChannel++)
            {
                var cnHelp2 = cnHelp1;
                for (int row = 0; row < OutputRows; row++)
                {
                    var currentBNNumber = cnHelp2;
                    for (int column = 0; column < OutputColumns; column++)
                    {
                        Neuron currentBNeuron = Neurons[currentBNNumber];
                        Neuron currentENeuron = NextLayer.Neurons[currentENNumber];
                        var connection = new UnteachableConnection(currentBNeuron, currentENeuron, 1);
                        currentBNeuron.AddOutputConnection(connection);
                        currentENeuron.AddInputConnection(connection);
                        currentBNNumber += PoolingSize;
                        currentENNumber++;
                    }
                    cnHelp2 += Size[2] * PoolingSize;
                }
                cnHelp1 += CountInChannel;
            }
        }

        public override void Activation()
        {
            int cnHelp1 = 0;
            int currentENNumber = 0;
            //[countOfChannels, countOfRows, countOfColumns]
            for (int inputChannel = 0; inputChannel < Size[0]; inputChannel++)
            {
                var rHelp = 0;
                var cnHelp2 = cnHelp1;
                for (int row = 0; row < OutputRows; row++)
                {
                    var cHelp = 0;
                    var currentBNNumber = cnHelp2;
                    for (int column = 0; column < OutputColumns; column++)
                    {
                        Neuron maxValue = Neurons[currentBNNumber];
                        Neuron currentENeuron = NextLayer.Neurons[currentENNumber];
                        for (int poolRow = 0; poolRow < PoolingSize; poolRow++)
                        {
                            for (int poolColumn = 0; poolColumn < PoolingSize; poolColumn++)
                            {
                                if ((rHelp + poolRow < Size[1]) && (cHelp + poolColumn < Size[2]))
                                {
                                    Neuron neuron = Neurons[currentBNNumber + poolRow * Size[2] + poolColumn];
                                    neuron.CalculateOutput();
                                    if (neuron.Output > maxValue.Output) maxValue = neuron;
                                }
                            }
                        }
                        var connection = currentENeuron.InputConnections[0];
                        connection.Input.OutputConnections.RemoveAt(0);
                        connection.Input = maxValue;
                        maxValue.OutputConnections.Add(connection);
                        maxValue.Activation();
                        currentBNNumber += PoolingSize;
                        currentENNumber++;
                        cHelp += PoolingSize;
                    }
                    cnHelp2 += Size[2] * PoolingSize;
                    rHelp += PoolingSize;
                }
                cnHelp1 += CountInChannel;
            }
        }
    }

    /// <summary>
    /// Provides methods for summation pooling layer
    /// </summary>
    class SumPoolingLayer : PoolingLayer
    {
        /// <summary>
        /// Create instance of summation pooling layer
        /// </summary>
        /// <param name="optimizer">optimization algorithm of this layer</param>
        /// <param name="function">activation function of all neurons of layer</param>
        /// <param name="countOfRows">count of rows</param>
        /// <param name="countOfColumns">count of columns</param>
        /// <param name="countOfChannels">count of input channels</param>
        /// <param name="poolingArea">size of cide of pooling square</param>
        /// <exception cref="NNException"></exception>
        public SumPoolingLayer(Optimizer optimizer, Activation function, int countOfRows, int countOfColumns, int countOfChannels = 1, int poolingArea = 2) : base(optimizer, function, countOfRows, countOfColumns, countOfChannels, poolingArea)
        {
            for (int i = 0; i < CountOfInputNeurons; i++)
                Neurons[i] = new Neuron(function);
        }

        /// <summary>
        /// Initialize neurons of layer
        /// </summary>
        /// <param name="nextLayer">next layer</param>
        public void InitConnections(Layer nextLayer)
        {
            ArgumentNullException.ThrowIfNull(nextLayer, nameof(nextLayer));

            NextLayer = nextLayer;
            int cnHelp1 = 0;
            int currentENNumber = 0;
            //[countOfChannels, countOfRows, countOfColumns]
            for (int inputChannel = 0; inputChannel < Size[0]; inputChannel++)
            {
                var rHelp = 0;
                var cnHelp2 = cnHelp1;
                for (int row = 0; row < OutputRows; row++)
                {
                    var cHelp = 0;
                    var currentBNNumber = cnHelp2;
                    for (int column = 0; column < OutputColumns; column++)
                    {
                        Neuron currentENeuron = NextLayer.Neurons[currentENNumber];
                        Variable weight = new(1);
                        for (int poolRow = 0; poolRow < PoolingSize; poolRow++)
                        {
                            for (int poolColumn = 0; poolColumn < PoolingSize; poolColumn++)
                            {
                                if ((rHelp + poolRow < Size[1]) && (cHelp + poolColumn < Size[2]))
                                {
                                    Neuron currentBNeuron = Neurons[currentBNNumber + poolRow * Size[2] + poolColumn];
                                    UnteachableConnection connection = new(currentBNeuron, currentENeuron, weight);
                                    currentBNeuron.AddOutputConnection(connection);
                                    currentENeuron.AddInputConnection(connection);
                                }
                            }
                        }
                        currentBNNumber += PoolingSize;
                        currentENNumber++;
                        cHelp += PoolingSize;
                    }
                    cnHelp2 += Size[2] * PoolingSize;
                    rHelp += PoolingSize;
                }
                cnHelp1 += CountInChannel;
            }
        }

        public override void InitializeConnections(Layer nextLayer)
        {
            InitConnections(nextLayer);
        }
    }

    /// <summary>
    /// Provides methods for average pooling layer
    /// </summary>
    class AveragePoolingLayer : SumPoolingLayer
    {
        /// <summary>
        /// Create instance of average pooling layer
        /// </summary>
        /// <param name="optimizer">optimization algorithm of this layer</param>
        /// <param name="function">activation function of all neurons of layer</param>
        /// <param name="countOfRows">count of rows</param>
        /// <param name="countOfColumns">count of columns</param>
        /// <param name="countOfChannels">count of input channels</param>
        /// <param name="poolingArea">size of cide of pooling square</param>
        /// <exception cref="NNException"></exception>
        public AveragePoolingLayer(Optimizer optimizer, Activation function, int countOfRows, int countOfColumns, int countOfChannels = 1, int poolingArea = 2) : base (optimizer, function, countOfRows, countOfColumns, countOfChannels, poolingArea)
        {
        }

        public override void InitializeConnections(Layer nextLayer)
        {
            InitConnections(nextLayer);
            foreach (Neuron neuron in NextLayer.Neurons)
                neuron.InputConnections[0].Weight.Value /= neuron.InputConnections.Count;
        }
    }

    /// <summary>
    /// Provides methods for neural network
    /// </summary>
    public class Net
    {
        /// <summary>
        /// Current architeture of neural network
        /// </summary>
        public List<Layer> Layers;

        /// <summary>
        /// Create new instance of network
        /// </summary>
        /// <param name="lastOptimizer">Optimization algorithm of output layer</param>
        /// <param name="lastActivation">Activation function of output layer</param>
        /// <param name="layers">Layers of NN</param>
        public Net(Optimizer lastOptimizer, Activation lastActivation, params Layer[] layers)
        {
            ArgumentNullException.ThrowIfNull(lastActivation, nameof(lastActivation));
            for (int begin = 0; begin < layers.Length - 1; begin++)
            {
                var bLayer = layers[begin];
                var eLayer = layers[begin + 1];
                if (bLayer == null)
                    throw new NNException("Layer with index " + begin.ToString() + " is null");
                if (eLayer == null)
                    throw new NNException("Layer with index " + (begin + 1).ToString() + " is null");
                var outputs = bLayer.CountOfOutputNeurons;
                var inputs = eLayer.CountOfInputNeurons;
                if (outputs != inputs)
                    throw new NNException("Layer with index " + begin.ToString() + " have " + outputs.ToString() + " output neurons but layer with index " + (begin + 1).ToString() + " have " + inputs.ToString());

                for (int end = begin + 1; end < layers.Length; end++)
                {
                    eLayer = layers[end];
                    if (bLayer == eLayer)
                        throw new NNException("Layers with indexes " + begin.ToString() + " and " + end.ToString() + " is equal");
                }
            }

            Layers = new List<Layer>(layers);
            for (int i = 0; i < Layers.Count - 1; i++)
                Layers[i].InitializeConnections(Layers[i + 1]);
            Layer lastLayer = Layers.Last();
            int lOutputs = lastLayer.CountOfOutputNeurons;
            LinearLayer newLast = new(lastOptimizer, lastActivation, lOutputs, lOutputs);
            lastLayer.InitializeConnections(newLast);
            Layers.Add(newLast);
            foreach (Layer layer in Layers) layer.SetOwnerLayer();
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
        public void SetInputs(decimal[] input)
        {
            Layers[0].SetInputs(input);
            for (int i = 1; i < Layers.Count; i++)
                Layers[i].ClearInputs();
        }

        /// <summary>
        /// Set errors
        /// </summary>
        /// <param name="right">Expected outputs</param>
        public void SetErrors(decimal[] right)
        {
            Layers.Last().SetErrors(right);
            for (int i = Layers.Count - 2; i > 0; i--)
                Layers[i].SetErrors();
        }

        /// <summary>
        /// Update weights
        /// </summary>
        public void UpdateWeights()
        {
            for (int i = Layers.Count - 1; i > 0; i--)
                Layers[i].UpdateWeights(this);
        }

        /// <summary>
        /// Return result of NN
        /// </summary>
        /// <param name="input">Input values</param>
        /// <returns>result of NN</returns>
        public decimal[] Output(decimal[] input)
        {
            SetInputs(input);
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
            SetInputs(input);
            Activation();
            ClearErrors();
            SetErrors(right);
            UpdateWeights();
        }
    }

    public class MyNet
    {
        public static void Main()
        {
            RandomGenerator.SetSeed(0);
            string Value = "";
            var act = new SigmoidActivation();
            var opt1 = new LeastSquareOptimizer(1M);
            var opt2 = new LeastSquareOptimizer(0.005M);
            var l1 = new ConvolutionLayer(opt1, act, 28, 28, 1, 3, 32, 1, 1);
            var l2 = new MaxPoolingLayer(opt1, act, l1.OutputRows, l1.OutputColumns, l1.CountOfOutputChannels, 2);
            var l3 = new ConvolutionLayer(opt1, act, l2.OutputRows, l2.OutputColumns, l1.CountOfOutputChannels, 3, 64, 1, 1);
            var l4 = new MaxPoolingLayer(opt1, act, l3.OutputRows, l3.OutputColumns, l3.CountOfOutputChannels, 2);
            //var l5 = new ConvolutionLayer(opt1, act, l4.OutputRows, l4.OutputColumns, l3.CountOfOutputChannels, 3, 3, 1, 1);
            //var l6 = new MaxPoolingLayer(opt1, act, l5.OutputRows, l5.OutputColumns, l5.CountOfOutputChannels, 2);
            var ln1 = new LinearLayer(opt2, act, 784, 784);
            var ln2 = new LinearLayer(opt2, act, 784, 441);
            var ln3 = new LinearLayer(opt2, act, 441, 10);
            var lin1 = new LinearLayer(opt2, act, l4.CountOfOutputNeurons, 64);
            Console.WriteLine(lin1.CountOfInputNeurons.ToString());
            
            var lin2 = new LinearLayer(opt2, act, lin1.CountOfOutputNeurons, 10);

            var sheduler = new ExponentialSheduler(opt2, 1M);
            //var net = new Net(opt2, act, l1, l2, l3, l4, lin1, lin2);
            var net = new Net(opt2, act, ln1, ln2, ln3);

            string[] lines = System.IO.File.ReadAllLines("D:\\source\\NNFramework\\mnist_test.csv");
            var testCount = lines.Length;
            var test_x = new decimal[testCount][];
            var test_y = new decimal[testCount][];
            var test_y_n = new int[testCount];
            Decimal down = 0.35M;
            Decimal up = 0.65M;
            for (int i = 0; i < testCount; i++)
            {
                var data = lines[i].Split(',');
                var y = int.Parse(data[0]);
                test_y_n[i] = y;
                decimal[] yD = [down, down, down, down, down, down, down, down, down, down];
                yD[y] = up;
                test_y[i] = yD;
                var data_x = new decimal[784];
                for (int j = 1; j < 785; j++)
                {
                    data_x[j - 1] = decimal.Parse(data[j]) / 255;
                }
                test_x[i] = data_x;
            }
            int dataSetLength = 10000;
            for (int epoch = 0; epoch < 100; epoch++)
            {
                var stopw = new Stopwatch();
                var score = 0;
                stopw.Start();
                decimal loss = 0;
                for (int i = 0; i < dataSetLength; i++)
                {
                    var output = net.Output(test_x[i]);
                    var outputNumber = 0;
                    var outputValue = 0M;
                    loss += net.Layers.Last().Loss;
                    for (int num = 0; num < 10; num++)
                    {
                        if (outputValue < output[num])
                        {
                            outputNumber = num;
                            outputValue = output[num];
                            
                        }
                    }

                    //Console.WriteLine(outputNumber.ToString() + " " + test_y_n[i]);
                    if (outputNumber == test_y_n[i])
                        score++;

                    net.ClearErrors();
                    net.SetErrors(test_y[i]);
                    net.UpdateWeights();
                    
                    //Console.WriteLine(i.ToString() + " " + opt.LearningRate.ToString());
                }
                loss /= dataSetLength;
                sheduler.Step();
                stopw.Stop();

                Console.WriteLine(epoch.ToString() + " time: "+ stopw.Elapsed.ToString() + " loss: " + loss + " score: " + score.ToString());
            }
        }
    }
}

﻿using System.Diagnostics;
using System.Threading;
using System.Linq;
using System.Runtime.InteropServices;
using DistributedWorkManager;
using static System.Runtime.InteropServices.JavaScript.JSType;
using static System.Formats.Asn1.AsnWriter;
using System.Net.NetworkInformation;

// Thanks to Lenard Gunda
// http://blog.rebuildall.net/2010/03/08/Running_NET_threads_on_selected_processor_cores
namespace DistributedWorkManager
{
    public class DistributedThread
    {
        [DllImport("kernel32.dll")]
        public static extern int GetCurrentThreadId();

        [DllImport("kernel32.dll")]
        public static extern int GetCurrentProcessorNumber();

        private ThreadStart threadStart;

        private ParameterizedThreadStart parameterizedThreadStart;

        private Thread thread;

        public int ProcessorAffinity { get; set; }

        public Thread ManagedThread
        {
            get
            {
                return thread;
            }
        }

        private DistributedThread()
        {
            thread = new Thread(DistributedThreadStart);
        }

        public DistributedThread(ThreadStart threadStart)
            : this()
        {
            this.threadStart = threadStart;
        }

        public DistributedThread(ParameterizedThreadStart threadStart)
            : this()
        {
            this.parameterizedThreadStart = threadStart;
        }

        public void Start()
        {
            if (this.threadStart == null) throw new InvalidOperationException();

            thread.Start(null);
        }

        public void Start(object parameter)
        {
            if (this.parameterizedThreadStart == null) throw new InvalidOperationException();

            thread.Start(parameter);
        }

        private void DistributedThreadStart(object parameter)
        {
            try
            {
                // fix to OS thread
                Thread.BeginThreadAffinity();

                // set affinity
                if (ProcessorAffinity != 0)
                {
                    CurrentThread.ProcessorAffinity = new IntPtr(ProcessorAffinity);
                }

                // call real thread
                if (this.threadStart != null)
                {
                    this.threadStart();
                }
                else if (this.parameterizedThreadStart != null)
                {
                    this.parameterizedThreadStart(parameter);
                }
                else
                {
                    throw new InvalidOperationException();
                }
            }
            finally
            {
                // reset affinity
                CurrentThread.ProcessorAffinity = new IntPtr(0xFFFF);
                Thread.EndThreadAffinity();
            }
        }

        private ProcessThread CurrentThread
        {
            get
            {
                int id = GetCurrentThreadId();
                return
                    (from ProcessThread th in Process.GetCurrentProcess().Threads
                     where th.Id == id
                     select th).Single();
            }
        }
    }
}

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
        /// Returns next pseudorandom number from 0 including to 1
        /// </summary>
        /// <returns>decimal from 0 including to 1</returns>
        public static decimal NextPositive() => (decimal)Random.NextDouble();

        /// <summary>
        /// Shuffle array of integer values
        /// </summary>
        /// <param name="values">array of values</param>
        public static void Shuffle(ref int[] values) => Random.Shuffle(values);

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
        /// Minimal count of neurons in thread
        /// </summary>
        public static int ThreadMinimum = 5000;

        /// <summary>
        /// Indicates whether the optimizer is attached
        /// </summary>
        private bool _isAttached;

        /// <summary>
        /// Owner layer
        /// </summary>
        public Layer OwnerLayer;

        /// <summary>
        /// Neurons
        /// </summary>
        public Neuron[] Neurons;

        /// <summary>
        /// Count of computational threads
        /// </summary>
        public int ThreadsCount = 1;

        /// <summary>
        /// Computational threads
        /// </summary>
        public Thread[] Threads;

        /// <summary>
        /// Bounds for computational threads
        /// </summary>
        public int[] TasksBounds;

        /// <summary>
        /// Task for thread
        /// </summary>
        public virtual void CalculateThread(int begin, int end)
        {
            for (int i = begin; i < end; i++)
            {
                Neuron neuron = Neurons[i];
                neuron.Output = neuron.Input + neuron.Bias;
            }
        }

        /// <summary>
        /// Calculate output values 
        /// </summary>
        /// 
        public virtual void Calculate()
        {
            int proc = 2;
            for (int i = 0; i < (ThreadsCount - 1); i++)
            {
                int t = i;

                DistributedThread thread = new DistributedThread(() => CalculateThread(TasksBounds[t], TasksBounds[t + 1]));
                thread.ProcessorAffinity = proc;
                thread.ManagedThread.Priority = ThreadPriority.AboveNormal;
                thread.ManagedThread.Name = "ThreadOnCPU1";
                thread.Start();
                proc *= 2;
                Threads[i] = thread.ManagedThread;
            }
            CalculateThread(0, TasksBounds[0]);

            foreach (Thread thread in Threads)
            {
                thread.Join();
                Thread.EndThreadAffinity();
            }
                
        }

        /// <summary>
        /// Task for thread
        /// </summary>
        public virtual void SetDerivativeeThread(int begin, int end)
        {
            for (int i = begin; i < end; i++)
            {
                Neuron neuron = Neurons[i];
                neuron.ActivationDerivative = 1;
            }
        }

        /// <summary>
        /// Return value of derived function in point
        /// </summary>
        public virtual void SetDerivative()
        {
            int proc = 2;
            for (int i = 0; i < (ThreadsCount - 1); i++)
            {
                int t = i;

                DistributedThread thread = new DistributedThread(() => SetDerivativeeThread(TasksBounds[t], TasksBounds[t + 1]));
                thread.ProcessorAffinity = proc;
                thread.ManagedThread.Priority = ThreadPriority.AboveNormal;
                thread.ManagedThread.Name = "ThreadOnCPU1";
                thread.Start();
                proc *= 2;
                Threads[i] = thread.ManagedThread;
            }
            SetDerivativeeThread(0, TasksBounds[0]);

            foreach (Thread thread in Threads)
            {
                thread.Join();
                Thread.EndThreadAffinity();
            }
        }

        /// <summary>
        /// Initialize computational threads
        /// </summary>
        /// <param name="countOfThreads">count of computational threads</param>
        /// <exception cref="NNException"></exception>
        private void initThreads(int countOfThreads)
        {
            if (countOfThreads < 1)
                throw new NNException("Count of computational threads must be more 0");

            ThreadsCount = countOfThreads;
            int alternativeCount = (int)Math.Ceiling((decimal)Neurons.Length / ThreadMinimum);
            if (alternativeCount < ThreadsCount)
                ThreadsCount = alternativeCount;
            TasksBounds = new int[ThreadsCount];
            for (int i = 0; i < ThreadsCount; i++)
            {
                TasksBounds[i] = Neurons.Length * (i + 1) / ThreadsCount;
            }
            Threads = new Thread[ThreadsCount - 1];
        }

        /// <summary>
        /// Attach this activation function to layer
        /// <param name="layer">Owner layer</param>
        /// <param name="countOfThreads">count of computational threads</param>
        /// </summary>
        public virtual void Attache(Layer layer, int countOfThreads = 2)
        {
            ArgumentNullException.ThrowIfNull(layer, nameof(layer));
            if (_isAttached)
                throw new NNException("This activation function is already attached");
            Neurons = layer.Neurons;
            if (Neurons == null)
                throw new NNException("Layer has no neurons");

            initThreads(countOfThreads);
            _isAttached = true;
            OwnerLayer = layer;
        }
    }

    /// <summary>
    /// Sigmoid activation function
    /// </summary>
    public class SigmoidActivation : Activation
    {
        public override void SetDerivativeeThread(int begin, int end)
        {
            for (int i = begin; i < end; i++)
            {
                Neuron neuron = Neurons[i];
                neuron.ActivationDerivative = neuron.Output * (1 - neuron.Output);
            }
        }

        public override void CalculateThread(int begin, int end)
        {
            for (int i = begin; i < end; i++)
            {
                Neuron neuron = Neurons[i];
                neuron.Output = (decimal)(1 / (1 + Math.Exp(decimal.ToDouble(-neuron.Input - neuron.Bias))));
            }
        }
        /*public override void Calculate()
        {
            foreach (Neuron neuron in Neurons)
                neuron.Output = (decimal)(1 / (1 + Math.Exp(decimal.ToDouble(- neuron.Input - neuron.Bias))));
        }

        public override void SetDerivative()
        {
            foreach (Neuron neuron in Neurons)
                neuron.ActivationDerivative = neuron.Output * (1 - neuron.Output);
        }*/
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

        public override void CalculateThread(int begin, int end)
        {
            for (int i = begin; i < end; i++)
            {
                Neuron neuron = Neurons[i];
                decimal input = neuron.Input + neuron.Bias;
                if (input > 0) neuron.Output = input;
                else neuron.Output = _value * input;
            }
        }

        public override void SetDerivative()
        {
            foreach (Neuron neuron in Neurons)
            {
                if (neuron.Input + neuron.Bias > 0) neuron.ActivationDerivative = 1;
                else neuron.ActivationDerivative = _value;
            }
        }
    }

    /// <summary>
    /// ReLU activation function
    /// </summary>
    public class ReLUActivation : Activation
    {
        public override void Calculate()
        {
            foreach (Neuron neuron in Neurons)
            {
                decimal input = neuron.Input + neuron.Bias;
                if (input > 0) neuron.Output = input;
                else neuron.Output = 0;
            }
        }

        public override void SetDerivative()
        {
            foreach (Neuron neuron in Neurons)
            {
                if (neuron.Input + neuron.Bias> 0) neuron.ActivationDerivative = 1;
                else neuron.ActivationDerivative = 0;
            }
        }
    }

    /// <summary>
    /// ReLU activation function
    /// </summary>
    public class SoftmaxActivation : Activation
    {
        /// <summary>
        /// Count of neurons in layer
        /// </summary>
        private int _countOfNeurons;

        /// <summary>
        /// Counter for calculations
        /// </summary>
        private int _counter;

        /// <summary>
        /// Dinominator for calculations
        /// </summary>
        private decimal _dinominator;

        /// <summary>
        /// Numenators for calculations
        /// </summary>
        private decimal[] _numenators;

        public override void Calculate()
        {
            _dinominator = 0;
            for (int i = 0; i < _countOfNeurons; i++)
            {
                decimal numenator = (decimal)(1 / (1 + Math.Exp(decimal.ToDouble(-Neurons[i].Input - Neurons[i].Bias))));
                _numenators[i] = numenator;
                _dinominator += numenator;
            }
            for (int i = 0; i < _countOfNeurons; i++)
                Neurons[i].Output = _numenators[i] / _dinominator;
        }

        public override void SetDerivative()
        {
            bool wasNoneZero = false;
            for (int i = 0; i < _countOfNeurons; i++)
            {
                Neuron neuron = Neurons[i];
                if (neuron.Error > 0)
                {
                    if (wasNoneZero)
                        throw new NNException("There was more than one category");
                    wasNoneZero = true;
                    neuron.ActivationDerivative = neuron.Output * (1 - neuron.Output);
                }
                else
                {
                    neuron.ActivationDerivative = - neuron.Output * _numenators[i] / _dinominator;
                }
            }
            if (!wasNoneZero)
                throw new NNException("There was less than one category");
        }

        public override void Attache(Layer layer, int threadsCount)
        {
            base.Attache(layer);
            _countOfNeurons = Neurons.Length;
            _numenators = new decimal[_countOfNeurons];
        }
    }

    /// <summary>
    /// Provides simple learning of NN
    /// </summary>
    public class Optimizer
    {
        /// <summary>
        /// SGD momentum
        /// </summary>
        public decimal Momentum;

        /// <summary>
        /// Indicates whether the optimizer is attached
        /// </summary>
        private bool _isAttached;

        /// <summary>
        /// Current neurons
        /// </summary>
        public Neuron[] Neurons;

        /// <summary>
        /// Owner layer
        /// </summary>
        public Layer OwnerLayer;

        /// <summary>
        /// Learning rate
        /// </summary>
        public decimal LearningRate;

        /// <summary>
        /// Set criterion value of prediction
        /// </summary>
        /// <param name="right">right values</param>
        public virtual void SetCriterionValue(decimal[] right)
        {

        }

        /// <summary>
        /// Calculate criterion value of prediction
        /// </summary>
        public virtual void SetDerivative()
        {

        }

        /// <summary>
        /// Create new optimizer
        /// </summary>
        /// <param name="lr">learning rate</param>
        public Optimizer(decimal lr) => LearningRate = lr;

        /// <summary>
        /// Attach this optimizer to layer
        /// </summary>
        /// <param name="layer">Owner layer</param>
        public virtual void Attache(Layer layer)
        {
            ArgumentNullException.ThrowIfNull(layer, nameof(layer));
            if (_isAttached)
                throw new NNException("This activation function is already attached");

            _isAttached = true;
            Neurons = layer.Neurons;
            OwnerLayer = layer;
        }
    }

    /// <summary>
    /// Provides least square methods
    /// </summary>
    public class LeastSquareOptimizer : Optimizer
    {
        public override void SetCriterionValue(decimal[] right)
        {
            var loss = 0M;
            for (int i = 0; i < Neurons.Length; i++)
            {
                var buffer = Neurons[i].Output - right[i];
                Neurons[i].Error = buffer;
                buffer *= buffer / 2;
                //Loss += Neurons[i].Error * Neurons[i].Error / 2;
                loss += buffer;
            }
            loss /= Neurons.Length;
            OwnerLayer.Loss = loss;
        }

        public override void SetDerivative()
        {
            foreach (Neuron neuron in Neurons)
                neuron.CriterionDerivative = neuron.Error;
        }
        
        /// <summary>
        /// Create instance of optimizer
        /// </summary>
        public LeastSquareOptimizer(decimal lr, decimal momentum) : base(lr)
        {
            Momentum = momentum;
        }
    }

    /// <summary>
    /// Provides cross entropy methods
    /// </summary>
    public class CrossEntropyOptimizer : Optimizer
    {
        private decimal[] _right;

        public override void SetCriterionValue(decimal[] right)
        {
            _right = right;
            var loss = 0M;
            for (int i = 0; i < Neurons.Length; i++)
            {
                var buffer = - right[i] * (decimal)Math.Log((double)Neurons[i].Output);
                Neurons[i].Error = buffer;
                loss += buffer;
            }
            OwnerLayer.Loss = loss;
        }

        public override void SetDerivative()
        {
            for (int i = 0; i < Neurons.Length; i++)
                Neurons[i].CriterionDerivative = -_right[i] / Neurons[i].Output;
        }

        /// <summary>
        /// Create instance of optimizer
        /// </summary>
        public CrossEntropyOptimizer(decimal lr) : base(lr)
        {

        }
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
        /// Past weight difference
        /// </summary>
        private decimal _deltaWeight = 0M;

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
        public virtual void UpdateWeight(decimal lr, decimal momentum, decimal derivative)
        {
            _deltaWeight = lr * derivative * Output.Output + _deltaWeight * momentum;
            Weight.Value -= _deltaWeight;
        }
            

        /// <summary>
        /// Add value to next neuron
        /// </summary>
        public void GoForward() => Output.Input += Input.Output * Weight.Value;

        /// <summary>
        /// Get error of output neuron
        /// </summary>
        public decimal GetError() => Weight.Value * Output.Derivative;
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

        public override void UpdateWeight(decimal lr, decimal momentum, decimal derivative)
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
        /// Current input value of neuron
        /// </summary>
        public decimal Input = 0;

        /// <summary>
        /// Current bias
        /// </summary>
        public decimal Bias;

        /// <summary>
        /// Current bias
        /// </summary>
        private decimal _deltaBias = 0;

        /// <summary>
        /// Current activation derivative of neuron
        /// </summary>
        public decimal ActivationDerivative = 0;

        /// <summary>
        /// Current criterion derivative of neuron
        /// </summary>
        public decimal CriterionDerivative = 0;

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
            foreach (Connection i in OutputConnections)
                i.GoForward();
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
        public virtual void SetDerivative() => 
            Derivative = ActivationDerivative * CriterionDerivative;

        /// <summary>
        /// Update weights
        /// </summary>
        /// <param name="net">Current NN which possess current Neuron</param>
        public virtual void UpdateWeights()
        {
            var lr = OwnerLayer.Optimizer.LearningRate;
            var mom = OwnerLayer.Optimizer.Momentum;
            _deltaBias = lr * Derivative + _deltaBias * mom;
            Bias -= _deltaBias;
            foreach (Connection i in InputConnections)
                i.UpdateWeight(lr, mom,  Derivative);
        }      

        /// <summary>
        /// Create new neuron
        /// </summary>
        /// <param name="setBias">indicates whether set bias</param>
        public Neuron(bool setBias) => Bias = setBias ? RandomGenerator.Next() : 0;
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
        public ConvolutionalNeuron(int devider, bool setBias) : base(setBias)
        {
            if (devider < 1) throw new NNException("Devider must be more then 1");
            _devider = devider;
        }

        public override void UpdateWeights()
        {
            var lr = OwnerLayer.Optimizer.LearningRate / _devider;
            var mom = OwnerLayer.Optimizer.Momentum;
            foreach (Connection i in InputConnections)
            {
                i.UpdateWeight(lr, mom, Derivative);
            }
        }
    }

    /// <summary>
    /// Provides methods for layer of fully connected neurons
    /// </summary>
    public class Layer
    {
        /// <summary>
        /// Indicates whether this layer is output
        /// </summary>
        public bool IsOutput;

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
        public virtual void InitializeConnections(Layer nextLayer)
        {
            if (IsOutput)
                throw new NNException("Cannot to initialize connections of output layer");
        }

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
            var result = new decimal[CountOfInputNeurons];
            for (int i = 0; i < CountOfInputNeurons; i++)
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
        /// Set derivatives to neurons
        /// </summary>
        private void setDerivatives()
        {
            ActivationFunction.SetDerivative();
            Optimizer.SetDerivative();
            foreach (Neuron neuron in Neurons)
                neuron.SetDerivative();
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

            Optimizer.SetCriterionValue(right);
            setDerivatives();
        }
        
        /// <summary>
        /// Set errors of neurons
        /// </summary>
        public virtual void SetErrors()
        {
            foreach (Neuron neuron in Neurons) neuron.SetErrors();
            setDerivatives();
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
            ActivationFunction.Calculate();
            if (!IsOutput)
            {
                foreach (Neuron i in Neurons) i.GoForward();
            }
        }

        /// <summary>
        /// Update weights
        /// </summary>
        public virtual void UpdateWeights()
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

        /// <summary>
        /// Create new layer
        /// </summary>
        /// <param name="isOutput">Indicates whether this layer is output</param>
        public Layer(bool isOutput) => IsOutput = isOutput;
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
        /// <exception cref="NNException"></exception>
        public LinearLayer(Optimizer optimizer, Activation function, int inputSize, int outputSize, bool isOutput = false, bool setBias = false) : base(isOutput)
        {
            ArgumentNullException.ThrowIfNull(optimizer, nameof(optimizer));
            ArgumentNullException.ThrowIfNull(function, nameof(function));
            if (inputSize < 1)
                throw new NNException("Input size of neural network can not be less than 1");
            if (outputSize < 1)
                throw new NNException("Output size of neural network can not be less than 1");

            Optimizer = optimizer;
            ActivationFunction = function;
            CountOfInputNeurons = inputSize;
            CountOfOutputNeurons = outputSize;
            Size = [inputSize];
            Outputs = new Variable[outputSize];
            for (int i = 0; i < outputSize; i++) Outputs[i] = new Variable(0);
            Neurons = new Neuron[inputSize];
            for (int i = 0; i < inputSize; i++) Neurons[i] = new Neuron(setBias);
            ActivationFunction.Attache(this);
            optimizer.Attache(this);
        }

        public override void InitializeConnections(Layer nextLayer)
        {
            ArgumentNullException.ThrowIfNull(nextLayer, nameof(nextLayer));

            base.InitializeConnections(nextLayer);
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
    /// Provides methods for dropout layer
    /// </summary>
    class DropoutLayer : Layer
    {
        /// <summary>
        /// Dropout probability
        /// </summary>
        public decimal DropoutProbability;

        /// <summary>
        /// Create new dropout layer
        /// </summary>
        /// <param name="optimizer">optimization algorithm of this layer</param>
        /// <param name="function">activation function of all neurons of layer</param>
        /// <param name="inputSize">count of neurons</param>
        /// <param name="probability">dropout probability</param>
        public DropoutLayer(Optimizer optimizer, Activation function, int inputSize, decimal probability = 0.2M, bool isOutput = false, bool setBias = false) : base(isOutput)
        {
            ArgumentNullException.ThrowIfNull(optimizer, nameof(optimizer));
            ArgumentNullException.ThrowIfNull(function, nameof(function));
            if (inputSize < 1)
                throw new NNException("Input size of neural network can not be less than 1");
            if (probability < 0M || probability > 1M)
                throw new NNException("'probability' must be in interval [0, 1]");

            DropoutProbability = probability;
            Optimizer = optimizer;
            CountOfInputNeurons = inputSize;
            CountOfOutputNeurons = inputSize;
            Size = [inputSize];

            Neurons = new Neuron[inputSize];
            for (int i = 0; i < inputSize; i++) Neurons[i] = new Neuron(setBias);
            ActivationFunction = function;
            ActivationFunction.Attache(this);
            optimizer.Attache(this);
        }
        
        public override void InitializeConnections(Layer nextLayer)
        {
            ArgumentNullException.ThrowIfNull(nextLayer, nameof(nextLayer));

            base.InitializeConnections(nextLayer);
            for (int begin = 0; begin < CountOfInputNeurons; begin++)
            {
                var beginNeuron = Neurons[begin];
                var endNeuron = nextLayer.Neurons[begin];
                var connection = new UnteachableConnection(beginNeuron, endNeuron, 1M);
                beginNeuron.AddOutputConnection(connection);
                endNeuron.AddInputConnection(connection);
            }
        }

        public override void Activation()
        {
            foreach (Neuron neuron in Neurons)
                neuron.OutputConnections.First().Weight.Value = 1;
            base.Activation();

            foreach (Neuron neuron in Neurons)
            {
                if (RandomGenerator.NextPositive() < DropoutProbability)
                    neuron.OutputConnections.First().Weight.Value = 0;
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
        public ConvolutionLayer(Optimizer optimizer, Activation function, int countOfRows, int countOfColumns, int countOfChannels = 1, int maskSize = 3, int countOfMasks = 3, int stride = 1, int padding = 0, bool isOutput = false, bool setBias = false) : base(isOutput)
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
                Neurons[i] = new ConvolutionalNeuron(countInOutputChannel, setBias);
            ActivationFunction = function;
            ActivationFunction.Attache(this);
            optimizer.Attache(this);
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

            base.InitializeConnections(nextLayer);
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
        public PoolingLayer(Optimizer optimizer, Activation function, int countOfRows, int countOfColumns, int countOfChannels = 1, int poolingArea = 2, bool isOutput = false, bool setBias = false) : base(isOutput)
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
        public MaxPoolingLayer(Optimizer optimizer, Activation function, int countOfRows, int countOfColumns, int countOfChannels = 1, int poolingArea = 2, bool isOutput = false, bool setBias = false) : base (optimizer, function, countOfRows, countOfColumns, countOfChannels, poolingArea, isOutput, setBias)
        {
            for (int i = 0; i < CountOfInputNeurons; i++)
                Neurons[i] = new Neuron(setBias);
            ActivationFunction = function;
            ActivationFunction.Attache(this);
            optimizer.Attache(this);
        }

        public override void InitializeConnections(Layer nextLayer)
        {
            ArgumentNullException.ThrowIfNull(nextLayer, nameof(nextLayer));

            base.InitializeConnections(nextLayer);
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
            ActivationFunction.Calculate();
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
                                    
                                    
                                    //----------------------neuron.CalculateOutput();


                                    if (neuron.Output > maxValue.Output) maxValue = neuron;
                                }
                            }
                        }
                        var connection = currentENeuron.InputConnections[0];
                        connection.Input.OutputConnections.RemoveAt(0);
                        connection.Input = maxValue;
                        maxValue.OutputConnections.Add(connection);


                        //-------------------maxValue.Activation(); 


                        currentBNNumber += PoolingSize;
                        currentENNumber++;
                        cHelp += PoolingSize;
                    }
                    cnHelp2 += Size[2] * PoolingSize;
                    rHelp += PoolingSize;
                }
                cnHelp1 += CountInChannel;
            }
            foreach (Neuron neuron in Neurons)
                neuron.GoForward();
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
        public SumPoolingLayer(Optimizer optimizer, Activation function, int countOfRows, int countOfColumns, int countOfChannels = 1, int poolingArea = 2, bool isOutput = false, bool setBias = false) : base(optimizer, function, countOfRows, countOfColumns, countOfChannels, poolingArea, isOutput, setBias)
        {
            for (int i = 0; i < CountOfInputNeurons; i++)
                Neurons[i] = new Neuron(setBias);
            ActivationFunction = function;
            ActivationFunction.Attache(this);
            optimizer.Attache(this);
        }

        /// <summary>
        /// Initialize neurons of layer
        /// </summary>
        /// <param name="nextLayer">next layer</param>
        public void InitConnections(Layer nextLayer)
        {
            ArgumentNullException.ThrowIfNull(nextLayer, nameof(nextLayer));


            base.InitializeConnections(nextLayer);
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
        public AveragePoolingLayer(Optimizer optimizer, Activation function, int countOfRows, int countOfColumns, int countOfChannels = 1, int poolingArea = 2, bool isOutput = false, bool setBias = false) : base (optimizer, function, countOfRows, countOfColumns, countOfChannels, poolingArea, isOutput, setBias)
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
        public Net(Optimizer lastOptimizer, Activation lastActivation, bool setBias, params Layer[] layers)
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
            LinearLayer newLast = new(lastOptimizer, lastActivation, lOutputs, lOutputs, true, setBias);
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
                Layers[i].UpdateWeights();
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

    /// <summary>
    /// Provides methods for data shuffle
    /// </summary>
    public class DataShuffler
    {
        /// <summary>
        /// Data length
        /// </summary>
        private int _length;

        /// <summary>
        /// Current index
        /// </summary>
        private int _index;

        /// <summary>
        /// Indexes queue
        /// </summary>
        private int[] _indexes;

        /// <summary>
        /// Create new data shuffler
        /// </summary>
        /// <param name="lenght">data length</param>
        /// <exception cref="NNException"></exception>
        public DataShuffler(int lenght)
        {
            if (lenght < 1)
                throw new NNException("Length must be more 0");

            _length = lenght;
            _indexes = new int[lenght];
            for (int i = 0; i < lenght;  i++)
                _indexes[i] = i;
            RandomGenerator.Shuffle(ref _indexes);
        }

        /// <summary>
        /// Return next index of data
        /// </summary>
        /// <returns>index of data</returns>
        public int Next()
        {
            if (_index < _length)
                return _indexes[_index++];
            else return -1;
        }

        /// <summary>
        /// Generate new queue of indexes and reset index to 0
        /// </summary>
        public void Update()
        {
            RandomGenerator.Shuffle(ref _indexes);
            _index = 0;
        }
    }

    public class MyNet
    {
        public static void Main_Testing_BattleField()
        {
            RandomGenerator.SetSeed(0);
            decimal lr2 = 0.3M;
            var ln1 = new LinearLayer(new LeastSquareOptimizer(lr2, 0), new Activation(), 2, 3);
            var ln2 = new LinearLayer(new LeastSquareOptimizer(lr2, 0), new SigmoidActivation(), 3, 2);
            var ln3 = new LinearLayer(new LeastSquareOptimizer(lr2, 0), new SigmoidActivation(), 2, 1);
            var net = new Net(new LeastSquareOptimizer(lr2, 0), new SigmoidActivation(), true, ln1, ln2, ln3);
            ln1.SetWeights([-1M, 2.5M, 1M, 1M, 0.4M, -1.5M]);
            ln2.SetWeights([2.2M, 0.34M, -1.4M, 1.05M, 0.56M, 3.1M]);
            ln3.SetWeights([0.75M, -0.22M]);

            for (int epoch = 0; epoch < 1; epoch++)
            {
                var output = net.Output([0.6M, 0.7M]);
                
                net.ClearErrors();
                net.SetErrors([0.9M]);
                net.UpdateWeights();
                foreach (Variable weight in ln1.Weights)
                {
                    Console.WriteLine(weight.Value.ToString() + " ");

                }
            }

        }
    }

    public class MyNet1
    {
        
        public static void SetLR(Net net, decimal lr)
        {
            foreach (Layer layer in net.Layers)
            {
                layer.Optimizer.LearningRate = lr;
            }
        }
        private static void circle()
        {
            //Thread.Sleep(1000);
            //var stopw = new Stopwatch();
            //stopw.Start();
            /*
            for (int x = 0; x < 10; x++)
            {
                for (int i = 0; i < 10; i++)
                {
                    var a = i - 1;
                    var b = a / (2 * i + 1);
                    var stw = Math.Exp((double)b);

                }
            }
            */
            //stopw.Stop();

            //Console.Write(stopw.Elapsed.ToString());
        }

        public static void Main()
        {
            /*
            for (int i = 0; i < 10; i++)
            {
                DistributedThread thread = new DistributedThread(Main__);
                thread.ProcessorAffinity = 1;
                thread.ManagedThread.Priority = ThreadPriority.Highest;
                thread.ManagedThread.Name = "ThreadOnCPU1";
                thread.Start();
                thread.ManagedThread.Join();

            }
            
            DistributedThread thread = new DistributedThread(Main1);
            thread.ProcessorAffinity = 1;
            
            thread.Start(); 
            */
            DistributedThread thread = new DistributedThread(Main1);
            thread.ProcessorAffinity = 1;
            thread.ManagedThread.Priority = ThreadPriority.Highest;
            thread.ManagedThread.Name = "ThreadOnCPU1";
            thread.Start();
        }

        public static void Main___()
        {
            for (int i = 0; i < 100; i++)
            {
                var StartTime = DateTime.Now;
                //DistributedThread thread = new DistributedThread(circle);
                //thread.ProcessorAffinity = 2;
                //thread.ManagedThread.Priority = ThreadPriority.Highest;
                //thread.ManagedThread.Name = "ThreadOnCPU2";
                //thread.Start();
                //thread.ManagedThread.Join();
                TimeSpan ts = DateTime.Now.Subtract(StartTime);
                string elapsedTime = string.Format("{0:00}:{1:00}:{2:00}.{3:000}",
                        ts.Hours, ts.Minutes, ts.Seconds,
                        ts.Milliseconds);
                Console.WriteLine(elapsedTime);

            }
            /*
            DistributedThread thread = new DistributedThread(Main1);
            thread.ProcessorAffinity = 1;
            
            thread.Start(); 
            */
            /*DistributedThread thread = new DistributedThread(Main1);
            thread.ProcessorAffinity = 1;
            thread.ManagedThread.Priority = ThreadPriority.Highest;
            thread.ManagedThread.Name = "ThreadOnCPU1";
            thread.Start();
            */
        }

        public static void Main1()
        {
            RandomGenerator.SetSeed(0);
            int bestLossEpoch = 0;
            decimal bestLoss = decimal.MaxValue;
            int bestScoreEpoch = 0;
            int bestScore = 0;

            decimal lr1 = 0.00001M;
            decimal lr2 = 0.1M;
            decimal momentum = 0.6M;
            int dataSetLength = 1000;
            int epochs = 120;
            DataShuffler dataShuffler = new(dataSetLength);

            

            var ln1 = new LinearLayer(new LeastSquareOptimizer(lr2, momentum), new Activation(), 784, 500, false, false);
            var ln2 = new LinearLayer(new LeastSquareOptimizer(lr2, momentum), new SigmoidActivation(), 500, 256, false, false);
            var ln3 = new LinearLayer(new LeastSquareOptimizer(lr2, momentum), new SigmoidActivation(), 256, 128, false, false);
            var ln4 = new LinearLayer(new LeastSquareOptimizer(lr2, momentum), new SigmoidActivation(), 128, 10, false, false);
            var net = new Net(new LeastSquareOptimizer(lr2, momentum), new SigmoidActivation(), false, ln1, ln2, ln3, ln4);

            var opt2 = new LeastSquareOptimizer(lr2, momentum);
            var l1 = new ConvolutionLayer(new LeastSquareOptimizer(lr1, momentum), new SigmoidActivation(), 28, 28, 1, 4, 16, 1, 1);
            var l2 = new MaxPoolingLayer(new LeastSquareOptimizer(lr1, momentum), new SigmoidActivation(), l1.OutputRows, l1.OutputColumns, l1.CountOfOutputChannels, 2);
            var l3 = new ConvolutionLayer(new LeastSquareOptimizer(lr1, momentum), new SigmoidActivation(), l2.OutputRows, l2.OutputColumns, l1.CountOfOutputChannels, 3, 8, 1, 1);
            var l4 = new MaxPoolingLayer(new LeastSquareOptimizer(lr1, momentum), new SigmoidActivation(), l3.OutputRows, l3.OutputColumns, l3.CountOfOutputChannels, 2);
            //var l5 = new ConvolutionLayer(opt1, act, l4.OutputRows, l4.OutputColumns, l3.CountOfOutputChannels, 3, 3, 1, 1);
            //var l6 = new MaxPoolingLayer(opt1, act, l5.OutputRows, l5.OutputColumns, l5.CountOfOutputChannels, 2);
            //var ln1 = new LinearLayer(opt2, act, 784, 784);
            //var ln2 = new LinearLayer(opt2, act, 784, 441);
            //var ln3 = new LinearLayer(opt2, act, 441, 10);
           
            //var drop1 = new DropoutLayer(new LeastSquareOptimizer(lr2, momentum), new SigmoidActivation(), 256, 0.05M);
            //var ln2 = new LinearLayer(new LeastSquareOptimizer(lr2, momentum), new SigmoidActivation(), 128, 10, false, false);
            //var drop2 = new DropoutLayer(new LeastSquareOptimizer(lr2, momentum), new SigmoidActivation(), 128, 0.01M);
            //var ln3 = new LinearLayer(new LeastSquareOptimizer(lr2), new SigmoidActivation(), 128, 64);
            //var ln4 = new LinearLayer(new LeastSquareOptimizer(lr2), new SigmoidActivation(), 64, 10);

            var drops1 = new DropoutLayer(new LeastSquareOptimizer(lr2, momentum), new SigmoidActivation(), l4.CountOfOutputNeurons, 0.05M);
            var lin1 = new LinearLayer(new LeastSquareOptimizer(lr2, momentum), new Activation(), l4.CountOfOutputNeurons, 64);
            Console.WriteLine(lin1.CountOfInputNeurons.ToString());

            var lin2 = new LinearLayer(new LeastSquareOptimizer(lr2, momentum), new SigmoidActivation(), lin1.CountOfOutputNeurons, 10);

            //var net = new Net(new CrossEntropyOptimizer(lr2), new SoftmaxActivation(), l1, l2, l3, l4, drops1, lin1, lin2);


            string[] lines = System.IO.File.ReadAllLines("D:\\source\\NNFramework\\mnist_test.csv");
            var testCount = lines.Length;
            var test_x = new decimal[testCount][];
            var test_y = new decimal[testCount][];
            var test_y_n = new int[testCount];

            

            decimal down = 0.01M;
            decimal up = 0.99M;
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
                    data_x[j - 1] = (decimal.Parse(data[j]) / 255);
                }
                test_x[i] = data_x;
            }

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                if (epoch == 0)
                {
                    SetLR(net, 0M);
                    net.Layers.Last().Optimizer.LearningRate = 0.05M;
                }
                if (epoch == 10)
                {
                    SetLR(net, 0M);
                    ln4.Optimizer.LearningRate = 0.05M;
                }
                if (epoch == 20)
                {
                    SetLR(net, 0M);
                    ln3.Optimizer.LearningRate = 0.05M;
                }
                if (epoch == 30)
                {
                    SetLR(net, 0M);
                    ln2.Optimizer.LearningRate = 0.05M;
                }

                if (epoch == 40)
                {
                    SetLR(net, 0M);
                    net.Layers.Last().Optimizer.LearningRate = 0.05M;
                }
                if (epoch == 50)
                {
                    SetLR(net, 0M);
                    ln4.Optimizer.LearningRate = 0.05M;
                }
                if (epoch == 60)
                {
                    SetLR(net, 0M);
                    ln3.Optimizer.LearningRate = 0.05M;
                }
                if (epoch == 70)
                {
                    SetLR(net, 0M);
                    ln2.Optimizer.LearningRate = 0.05M;
                }
                if (epoch == 80)
                {
                    SetLR(net, 0M);
                    net.Layers.Last().Optimizer.LearningRate = 0.05M;
                }
                if (epoch == 90)
                {
                    SetLR(net, 0M);
                    ln4.Optimizer.LearningRate = 0.05M;
                }
                if (epoch == 100)
                {
                    SetLR(net, 0M);
                    ln3.Optimizer.LearningRate = 0.05M;
                }
                if (epoch == 110)
                {
                    SetLR(net, 0M);
                    ln2.Optimizer.LearningRate = 0.05M;
                }


                //if (epoch == 1) SetLR(net, 0.05M);
                //if (epoch == 15) SetLR(net, 0.02M);
                //if (epoch == 30) SetLR(net, 0.01M);
                //if (epoch == 50) SetLR(net, 0.005M);
                //if (epoch == 60) SetLR(net, 0.002M);
                //if (epoch == 70) SetLR(net, 0.001M);
                DateTime StartTime = DateTime.Now;
                var score = 0;
                decimal loss = 0;
                for (int i11 = 0; i11 < dataSetLength; i11++)
                {
                    int i = dataShuffler.Next();
                    var output = net.Output(test_x[i]);
                    

                    //foreach (decimal iii in output)
                    //    Console.Write(iii.ToString() + " ");
                    //Console.WriteLine();
                    var outputNumber = 0;
                    var outputValue = 0M;
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
                    loss += net.Layers.Last().Loss;
                    net.UpdateWeights();


                    //Console.WriteLine(i.ToString() + " " + opt.LearningRate.ToString());
                }
                loss /= dataSetLength;
                TimeSpan ts = DateTime.Now.Subtract(StartTime);
                string elapsedTime = string.Format("{0:00}:{1:00}:{2:00}.{3:000}",
                        ts.Hours, ts.Minutes, ts.Seconds,
                        ts.Milliseconds);
                Console.Write(epoch.ToString() + " loss: " + loss + " score: " + score.ToString() + " ");
                Console.Write(elapsedTime, "RunTime");

                if (bestLoss > loss)
                {
                    bestLoss = loss;
                    bestLossEpoch = epoch;
                    Console.Write(" bestLoss --");
                }
                if (bestScore < score)
                {
                    bestScore = score;
                    bestScoreEpoch = epoch;
                    Console.Write(" bestScore ++");
                }
                Console.WriteLine();
                dataShuffler.Update();
            }
            
        }
    }
}

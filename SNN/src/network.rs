use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Mul};
use std::sync::Arc;
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::{read_npy, ReadNpyError};
use serde::{Serialize, Deserialize};
use rand::{Rng, thread_rng};
use num_traits::cast::FromPrimitive;
use tokio::task::JoinSet;


#[derive(Clone, Copy)]
pub enum FaultType {
    StuckAtZero(u8),
    StuckAtOne(u8),
    Transient(u8, i32),
}

impl Display for FaultType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FaultType::StuckAtZero(bit) => {
                write!(f, "Bit {} stuck at ZERO", bit)
            }
            FaultType::StuckAtOne(bit) => {
                write!(f, "Bit {} stuck at ONE", bit)
            }
            FaultType::Transient(bit, time) => {
                write!(f, "Bit {} FLIPPED at step {}", bit, time)
            }
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum OpSelector {
    FirstOperand,
    SecondOperand,
    Result,
}

#[derive(PartialEq, Clone)]
pub enum TestedUnit {
    //elaboration
    Adder(OpSelector, usize),
    Multiplier(OpSelector, usize),
    Comparator(OpSelector, usize),
    //memory
    SynapseWeight(usize, usize, bool, Option<Vec<Arc<Vec<Vec<f32>>>>>),
    RestPotential(usize),
    ThresholdPotential(usize),
    ResetPotential(usize),
    Potential(usize),
    //communication
    NeuronInput(usize, Option<Vec<Arc<Vec<Vec<f32>>>>>),
    NeuronOutput(usize, Option<Vec<Arc<Vec<Vec<f32>>>>>),
}

impl Display for TestedUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TestedUnit::Adder(op, neuron) => {
                write!(f, "Adder ({:?} affected) at neuron {}", op, neuron)
            }
            TestedUnit::Multiplier(op, neuron) => {
                write!(f, "Multiplier ({:?} affected) at neuron {}", op, neuron)
            }
            TestedUnit::Comparator(op, neuron) => {
                write!(f, "Comparator ({:?} affected) at neuron {}", op, neuron)
            }
            TestedUnit::SynapseWeight(neuron, edge, same_layer, _) => {
                write!(f, "Synapse weight n°{} from neuron {} to {} layer ", edge, neuron, if *same_layer { "same" } else { "next" })
            }
            TestedUnit::RestPotential(neuron) => {
                write!(f, "Rest Potential at neuron {}", neuron)
            }
            TestedUnit::ThresholdPotential(neuron) => {
                write!(f, "Threshold Potential at neuron {}", neuron)
            }
            TestedUnit::ResetPotential(neuron) => {
                write!(f, "Reset Potential at neuron {}", neuron)
            }
            TestedUnit::Potential(neuron) => {
                write!(f, "Potential at neuron {}", neuron)
            }
            TestedUnit::NeuronInput(neuron, _) => {
                write!(f, "Input of neuron {}", neuron)
            }
            TestedUnit::NeuronOutput(neuron, _) => {
                write!(f, "Output of neuron {}", neuron)
            }
        }
    }
}

#[allow(dead_code)]
pub enum Fault {
    StuckAtZero,
    StuckAtOne,
    Transient,
}

#[allow(dead_code)]
pub enum Unit {
    //elaboration
    Adder,
    Multiplier,
    Comparator,
    //memory
    SynapseWeight,
    RestPotential,
    ThresholdPotential,
    ResetPotential,
    Potential,
    //communication
    NeuronInput,
    NeuronOutput,
}

enum UnitInputGenerator {
    SynapseWeight(usize, usize, bool),
    NeuronInput(usize),
    NeuronOutput(usize),
}

/// Neuron configuration for Snn
/// # Fields
/// * rest_potential : rest potential of the neuron
/// * threshold_potential : threshold potential of the neuron
/// * reset_potential : reset potential of the neuron
/// * time_constant : time constant of the neuron membrane
pub struct NeuronConfig {
    pub rest_potential: f32,
    pub threshold_potential: f32,
    pub reset_potential: f32,
    pub time_constant: f32,
}

impl NeuronConfig{
    /// Create Neuron configuration for Snn
    ///
    /// # Arguments
    /// * rest_potential : rest potential of the neuron
    /// * threshold_potential : threshold potential of the neuron
    /// * reset_potential : reset potential of the neuron
    /// * time_constant : time constant of the neuron membrane
    pub fn new(rest_potential: f32, threshold_potential: f32, reset_potential: f32, time_constant: f32) ->Self{
        Self{
            rest_potential,
            threshold_potential,
            reset_potential,
            time_constant,
        }
    }
}
///Input reader to manage fault injection
pub struct InputReader {
    values: Arc<Array3<f32>>,
    limit: Option<usize>,
}

impl InputReader {
    ///Create a new instance of InputReader
    /// # Arguments
    /// * path : path to numpy file
    /// * limit : add optional limit to number of input retrieved from the file
    pub fn from_numpy(path: &str, limit: Option<usize>) -> Result<Self, ReadNpyError> {
        Ok(Self {
            values: Arc::new(read_npy(path)?),
            limit,
        })
    }

    fn inject_fault(&self, snn: Arc<Snn>, input_matrices: &Vec<Arc<Vec<Vec<f32>>>>, fault: (FaultType, UnitInputGenerator)) -> Vec<Arc<Vec<Vec<f32>>>> {
        let mut ret = Vec::with_capacity(input_matrices.len());

        for (input_n, input) in input_matrices.iter().enumerate() {
            let mut faulted_input = (**input).clone();
            match fault.1 {
                UnitInputGenerator::SynapseWeight(neuron, edge, _ ) => {
                    let diff = -snn.neurons[neuron].next_layer_synapses[edge].weight
                        + Snn::add_fault_to_float(snn.neurons[neuron].next_layer_synapses[edge].weight, &fault.0);

                    match &fault.0 {
                        FaultType::Transient(_, time) => {
                            if *self.values.get((*time as usize, input_n, neuron)).unwrap() != 0.0 {
                                for i in faulted_input[*time as usize].iter_mut() {
                                    *i += diff;
                                }
                            }
                        }
                        _ => {
                            for time in 0..self.values.shape()[0] {
                                if *self.values.get((time, input_n, neuron)).unwrap() != 0.0 {
                                    for i in faulted_input[time as usize].iter_mut() {
                                        *i += diff;
                                    }
                                }
                            }
                        }
                    }
                }
                // each neuron has a boolean output (triggered or not), but, since we are in the first
                // layer, here each neuron also have a boolean input from the input matrix, so neuron input and
                // neuron output faults act the same
                UnitInputGenerator::NeuronOutput(neuron) | UnitInputGenerator::NeuronInput(neuron)=> {
                    match fault.0 {
                        FaultType::StuckAtZero(_) => {
                            for time in 0..self.values.shape()[0] {
                                if *self.values.get((time, input_n, neuron)).unwrap() != 0.0 {
                                    for synapse in &snn.neurons[neuron].next_layer_synapses {
                                        faulted_input[time][synapse.to - snn.layers[0].neurons.len()] -= synapse.weight;
                                    }
                                }
                            }
                        }
                        FaultType::StuckAtOne(_) => {
                            for time in 0..self.values.shape()[0] {
                                if *self.values.get((time, input_n, neuron)).unwrap() == 0.0 {
                                    for synapse in &snn.neurons[neuron].next_layer_synapses {
                                        faulted_input[time][synapse.to - snn.layers[0].neurons.len()] += synapse.weight;
                                    }
                                }
                            }
                        }
                        FaultType::Transient(_, time) => {
                            if *self.values.get((time as usize, input_n, neuron)).unwrap() == 0.0 {
                                for synapse in &snn.neurons[neuron].next_layer_synapses {
                                    faulted_input[time as usize][synapse.to - snn.layers[0].neurons.len()] += synapse.weight;
                                }
                            } else {
                                for synapse in &snn.neurons[neuron].next_layer_synapses {
                                    faulted_input[time as usize][synapse.to - snn.layers[0].neurons.len()] -= synapse.weight;
                                }
                            }
                        }
                    }
                }
            }

            ret.push(Arc::new(faulted_input));
        }
        ret
    }

    async fn generate_inputs(&self, snn: Arc<Snn>) -> Vec<Arc<Vec<Vec<f32>>>> {
        let total_inputs = self.values.shape()[1];
        let total_signals = self.values.shape()[0];
        let total_neurons = self.values.shape()[2];

        let mut inputs = BTreeMap::<usize, Vec<Vec<f32>>>::new();
        let mut handles = JoinSet::<(usize, Vec<Vec<f32>>)>::new();

        for input_n in 0..total_inputs {
            if let Some(x) = self.limit {
                if input_n == x { break; }
            }

            handles.spawn(Self::worker(snn.clone(), total_signals, total_neurons, self.values.clone(), input_n));
        }

        while let Some(res) = handles.join_next().await {
            if let Ok((input_n, input)) = res{
                match inputs.get_mut(&input_n){
                    None => {
                        inputs.insert(input_n, input);
                    }
                    Some(_) => {}
                }
            }

        }

        inputs.into_iter().map(|x| Arc::new(x.1)).collect()
    }

    async fn worker<'a>(snn: Arc<Snn>, total_signals: usize, total_neurons: usize, inputs_raw: Arc<Array3<f32>>, input_n: usize) -> (usize, Vec<Vec<f32>>) {
        let mut input = Vec::with_capacity(total_signals);
        for signal_n in 0..total_signals {
            let mut signal = vec![0.0; snn.layers[1].neurons.len()];
            for neuron_n in 0..total_neurons {
                if *inputs_raw.get((signal_n, input_n, neuron_n)).unwrap() == 0.0 { continue; }
                for synapse in &snn.neurons[neuron_n].next_layer_synapses {
                    signal[synapse.to - snn.layers[0].neurons.len()] += synapse.weight;
                }
            }
            input.push(signal);
        }
        (input_n, input)
    }
}

///Spiral Neural Network object
pub struct Snn {
    layers: Vec<Layer>,
    pub(crate) neurons: Vec<Neuron>,
    neuron_config: NeuronConfig,
}

impl Snn {
    /// Create new empty Spiral Neural Network
    pub fn new(neuron_config: NeuronConfig) -> Snn {
        Snn {
            layers: Vec::<Layer>::new(),
            neurons: Vec::<Neuron>::new(),
            neuron_config,
        }
    }
    ///
    /// Create Spiking neural network from vector of numpy files and neuron parameters
    ///
    /// # Arguments:
    /// * layer_paths: vector of paths to numpy files, each one representing one layer. Files must be ordered
    /// * time_constant : time constant of the neuron membrane
    /// * threshold_potential : threshold potential of the neuron
    /// * rest_potential : rest potential of the neuron
    /// * reset_potential : reset potential of the neuron
    pub fn from_numpy(layer_paths: Vec<&str>, neuron_config: NeuronConfig) -> Result<Snn, ReadNpyError> {
        let mut snn = Self::new(neuron_config);

        let input_layer = snn.new_layer();

        let mut neurons_added: usize = 0;
        for (n_layer, layer_path) in layer_paths.iter().enumerate() {
            let weights: Array2<f32> = read_npy(layer_path)?;
            let layer = snn.new_layer();

            if n_layer == 0 {
                for _ in 0..weights.shape()[1] {
                    snn.new_neuron(input_layer);
                }
            }

            for _ in 0..weights.shape()[0] {
                snn.new_neuron(layer);
            }
            for j in 0..weights.shape()[1] {
                for i in 0..weights.shape()[0] {
                    snn.new_synapse(j + neurons_added, i + neurons_added + weights.shape()[1], *weights.get((i, j)).unwrap());
                }
            }
            neurons_added += weights.shape()[1];
        }

        Ok(snn)
    }


    /// Create new layer in SNN
    pub fn new_layer(&mut self) -> usize {
        let ret = self.layers.len();
        self.layers.push(Layer {
            id: self.layers.len(),
            neurons: Vec::<usize>::new(),
        });
        ret
    }


    /// Create new neuron in SNN
    /// # Arguments
    /// * layer_id : layer identifier obtained from new_layer()
    ///
    pub fn new_neuron(&mut self, layer_id: usize) -> usize {
        let ret = self.neurons.len();
        self.neurons.push(Neuron {
            id: ret,
            layer: layer_id,
            next_layer_synapses: Vec::<Synapses>::new(),
            same_layer_synapses: Vec::<Synapses>::new(),
        });
        self.layers[layer_id].add_neuron(ret);
        ret
    }

    /// Create new synapse in SNN
    /// # Arguments
    /// * neuron_from : neuron identifier of the neuron where the synapse starts
    /// * neuron_to : neuron identifier of the neuron where the synapse ends
    /// * weight : synapse weight
    pub fn new_synapse(&mut self, neuron_from: usize, neuron_to: usize, weight: f32) {
        if self.layers[self.neurons[neuron_from].layer].neurons.contains(&neuron_to) {
            self.neurons[neuron_from].same_layer_synapses.push(Synapses {
                to: neuron_to,
                weight,
            });
        } else if self.layers[self.neurons[neuron_from].layer + 1].neurons.contains(&neuron_to) {
            self.neurons[neuron_from].next_layer_synapses.push(Synapses {
                to: neuron_to,
                weight,
            });
        }
    }

    fn add_fault_to_float(value: f32, fault_type: &FaultType) -> f32 {
        match fault_type {
            FaultType::StuckAtZero(bit) => {
                let mask = 1 << bit;
                f32::from_bits(value.to_bits() & !mask)
            }
            FaultType::StuckAtOne(bit) => {
                let mask = 1 << bit;
                f32::from_bits(value.to_bits() | mask)
            }
            FaultType::Transient(bit, _) => {
                let mask = 1 << bit;
                if value.to_bits() & mask == 0 {
                    f32::from_bits(value.to_bits() | mask)
                } else {
                    f32::from_bits(value.to_bits() & !mask)
                }
            }
        }
    }


    ///
    /// Test the SNN over a given array of possible faults
    /// # Arguments
    ///
    /// * `snn` : Arc smart pointer encapsulating the Snn struct representing the spiking neural network
    /// * `input_reader`: input reader to access inputs
    /// * `faults_to_add`: vector of possible faults to be added to the test
    /// * `n_inferences`: number of inferences of the input matrices each one randomly adding one of the faults in the fault_to_add array
    /// * `delta`: time measure of a time step
    /// * `f`: activation function of the neuron
    /// * `expected_outputs` : outputs expected
    ///
    /// returns a vector of Outputs, containing for each input the output with no fault added and the list of outputs
    /// with the faults added
    ///
    /// # Activation Function
    ///
    /// The activation function accept:
    /// * neuron_config: &NeuronConfig : reference to the SNN Neuron configuration
    /// * neuron_params: &NeuronInferenceParams: reference to neuron variable parameters
    /// * current_step : i32 : the current step of the execution
    /// * delta : f32 : the time measure of the time step
    /// * testing_add : Box<dyn Fn(f32, f32) -> f32> : adding function between f32 modified to accept faults
    /// * * example: a+b => a.testing_add(b)
    /// * testing_mul : Box<dyn Fn(f32, f32) -> f32> : multiplier function between f32 modified to accept faults
    /// * * example: a*b => a.testing_mul(b)
    /// * testing_cmp : Box<dyn Fn(f32, f32) -> Ordering> : comparator function between f32 modified to accept faults
    /// * * example: a == b => a.testing_cmp(b) == Ordering::Eq
    /// * * example: a > b => a.testing_cmp(b) == Ordering::Less
    /// * * example: a < b => a.testing_cmp(b) == Ordering::Greater
    ///
    /// returns pair of f32 as new calculated potential and bool as if the neuron was triggered
    ///
    /// Testing functions should be used instead of the +, * and <=> between f32 operations in order to allow testing of the adding, multiplier and comparator units.
    /// Anyways, if testing on those units is not required, the use of testing functions can be avoided.
    ///
    /// # Example
    ///
    /// Implementation of the Leaky Integrate and Fire model:
    ///
    /// ```
    /// let f = |neuron_config: &NeuronConfig, neuron_params: &NeuronInferenceParams, current_step: i32, delta: f32,
    /// testing_add: Box<dyn Fn(f32, f32) -> f32>,
    /// testing_mul: Box<dyn Fn(f32, f32) -> f32>,
    /// testing_cmp: Box<dyn Fn(f32, f32) -> Ordering>| {
    ///
    /// let a = testing_add(neuron_params.potential, -neuron_config.rest_potential);
    ///
    /// let b = testing_mul(testing_add(current_step as f32, -neuron_params.last_activity as f32), delta);
    ///
    /// let c = (-b / neuron_config.time_constant).exp();
    ///
    /// let mut new_potential = testing_add(testing_add(neuron_config.rest_potential, testing_mul(a, c)), neuron_params.input_signal);
    ///
    /// let triggered = testing_cmp(new_potential, neuron_config.threshold_potential) == Ordering::Greater;
    ///
    /// if triggered { new_potential = neuron_config.reset_potential };
    ///
    /// (new_potential, triggered)
    /// };
    /// ```
    pub async fn test<'a>(snn: Arc<Snn>, input_reader: InputReader, faults_to_add: Vec<(Fault, Unit)>, n_inferences: usize, delta: f32, f: fn(&NeuronConfig, &NeuronInferenceParams, i32, f32, Box<dyn Fn(f32, f32) -> f32>, Box<dyn Fn(f32, f32) -> f32>, Box<dyn Fn(f32, f32) -> Ordering>) -> (f32, bool), expected_outputs: Vec<ExpectedOutput>) -> Vec<Output> {
        let mut tree = BTreeMap::<usize, Output>::new();

        let none_arc = Arc::new(None);

        println!("Loading inputs...");

        // let start = Instant::now();
        let input_matrices = input_reader.generate_inputs(snn.clone()).await;
        // let elapsed = start.elapsed();
        // println!("Input Loading time: {:?}", elapsed);

        println!("Starting...");
        let mut join_set = JoinSet::<(usize, Vec<Vec<bool>>, Arc<Option<(FaultType, TestedUnit)>>)>::new();

        for _ in 0..n_inferences {
            if faults_to_add.len() == 0 {
                break;
            }
            let n_fault = thread_rng().gen_range(0..faults_to_add.len());
            let (ft, u) = &faults_to_add[n_fault];
            let broken_bit: u8 = thread_rng().gen_range(0..32);
            let mut broken_neuron: usize = thread_rng().gen_range(0..snn.neurons.len().clone());
            let broken_neuron_elaboration: usize = thread_rng().gen_range(snn.layers[1].neurons[0]..snn.neurons.len().clone());
            let time: i32 = thread_rng().gen_range(0..input_matrices[0].len()) as i32;
            let op = match thread_rng().gen_range(0..3) {
                0 => { OpSelector::FirstOperand }
                1 => { OpSelector::SecondOperand }
                _ => { OpSelector::Result }
            };
            let fault = match u {
                Unit::Adder => {
                    match ft {
                        Fault::StuckAtZero => { (FaultType::StuckAtZero(broken_bit), TestedUnit::Adder(op, broken_neuron_elaboration)) }
                        Fault::StuckAtOne => { (FaultType::StuckAtOne(broken_bit), TestedUnit::Adder(op, broken_neuron_elaboration)) }
                        Fault::Transient => { (FaultType::Transient(broken_bit, time), TestedUnit::Adder(op, broken_neuron_elaboration)) }
                    }
                }
                Unit::Multiplier => {
                    match ft {
                        Fault::StuckAtZero => { (FaultType::StuckAtZero(broken_bit), TestedUnit::Multiplier(op, broken_neuron_elaboration)) }
                        Fault::StuckAtOne => { (FaultType::StuckAtOne(broken_bit), TestedUnit::Multiplier(op, broken_neuron_elaboration)) }
                        Fault::Transient => { (FaultType::Transient(broken_bit, time), TestedUnit::Multiplier(op, broken_neuron_elaboration)) }
                    }
                }
                Unit::Comparator => {
                    match ft {
                        Fault::StuckAtZero => { (FaultType::StuckAtZero(broken_bit), TestedUnit::Comparator(op, broken_neuron_elaboration)) }
                        Fault::StuckAtOne => { (FaultType::StuckAtOne(broken_bit), TestedUnit::Comparator(op, broken_neuron_elaboration)) }
                        Fault::Transient => { (FaultType::Transient(broken_bit, time), TestedUnit::Comparator(op, broken_neuron_elaboration)) }
                    }
                }
                Unit::SynapseWeight => {
                    let mut same_layer = thread_rng().gen::<bool>();
                    let same_layer_len = snn.neurons[broken_neuron].same_layer_synapses.len();
                    let next_layer_len = snn.neurons[broken_neuron].next_layer_synapses.len();
                    let broken_edge;

                    if same_layer && same_layer_len != 0 {
                        broken_edge = thread_rng().gen_range(0..same_layer_len);
                    } else if next_layer_len != 0 {
                        same_layer = false;
                        broken_edge = thread_rng().gen_range(0..next_layer_len);
                    } else {
                        broken_neuron = 0;
                        broken_edge = 0
                    }


                    let faulted_inputs = if snn.neurons[broken_neuron].layer == 0 {
                        match ft {
                            Fault::StuckAtZero => Some(input_reader.inject_fault(snn.clone(), &input_matrices, (FaultType::StuckAtZero(broken_bit), UnitInputGenerator::SynapseWeight(broken_neuron, broken_edge, same_layer)))),
                            Fault::StuckAtOne => Some(input_reader.inject_fault(snn.clone(), &input_matrices, (FaultType::StuckAtOne(broken_bit), UnitInputGenerator::SynapseWeight(broken_neuron, broken_edge, same_layer)))),
                            Fault::Transient => Some(input_reader.inject_fault(snn.clone(), &input_matrices, (FaultType::Transient(broken_bit, time), UnitInputGenerator::SynapseWeight(broken_neuron, broken_edge, same_layer)))),
                        }
                    } else { None };

                    match ft {
                        Fault::StuckAtZero => { (FaultType::StuckAtZero(broken_bit), TestedUnit::SynapseWeight(broken_neuron, broken_edge, same_layer, faulted_inputs)) }
                        Fault::StuckAtOne => { (FaultType::StuckAtOne(broken_bit), TestedUnit::SynapseWeight(broken_neuron, broken_edge, same_layer, faulted_inputs)) }
                        Fault::Transient => { (FaultType::Transient(broken_bit, time), TestedUnit::SynapseWeight(broken_neuron, broken_edge, same_layer, faulted_inputs)) }
                    }
                }
                Unit::RestPotential => {
                    match ft {
                        Fault::StuckAtZero => { (FaultType::StuckAtZero(broken_bit), TestedUnit::RestPotential(broken_neuron_elaboration)) }
                        Fault::StuckAtOne => { (FaultType::StuckAtOne(broken_bit), TestedUnit::RestPotential(broken_neuron_elaboration)) }
                        Fault::Transient => { (FaultType::Transient(broken_bit, time), TestedUnit::RestPotential(broken_neuron_elaboration)) }
                    }
                }
                Unit::ThresholdPotential => {
                    match ft {
                        Fault::StuckAtZero => { (FaultType::StuckAtZero(broken_bit), TestedUnit::ThresholdPotential(broken_neuron_elaboration)) }
                        Fault::StuckAtOne => { (FaultType::StuckAtOne(broken_bit), TestedUnit::ThresholdPotential(broken_neuron_elaboration)) }
                        Fault::Transient => { (FaultType::Transient(broken_bit, time), TestedUnit::ThresholdPotential(broken_neuron_elaboration)) }
                    }
                }
                Unit::ResetPotential => {
                    match ft {
                        Fault::StuckAtZero => { (FaultType::StuckAtZero(broken_bit), TestedUnit::ResetPotential(broken_neuron_elaboration)) }
                        Fault::StuckAtOne => { (FaultType::StuckAtOne(broken_bit), TestedUnit::ResetPotential(broken_neuron_elaboration)) }
                        Fault::Transient => { (FaultType::Transient(broken_bit, time), TestedUnit::ResetPotential(broken_neuron_elaboration)) }
                    }
                }
                Unit::Potential => {
                    match ft {
                        Fault::StuckAtZero => { (FaultType::StuckAtZero(broken_bit), TestedUnit::Potential(broken_neuron_elaboration)) }
                        Fault::StuckAtOne => { (FaultType::StuckAtOne(broken_bit), TestedUnit::Potential(broken_neuron_elaboration)) }
                        Fault::Transient => { (FaultType::Transient(broken_bit, time), TestedUnit::Potential(broken_neuron_elaboration)) }
                    }
                }
                Unit::NeuronInput => {
                    let faulted_inputs = if snn.neurons[broken_neuron].layer == 0 {
                        match ft {
                            Fault::StuckAtZero => Some(input_reader.inject_fault(snn.clone(), &input_matrices, (FaultType::StuckAtZero(broken_bit), UnitInputGenerator::NeuronInput(broken_neuron)))),
                            Fault::StuckAtOne => Some(input_reader.inject_fault(snn.clone(), &input_matrices, (FaultType::StuckAtOne(broken_bit), UnitInputGenerator::NeuronInput(broken_neuron)))),
                            Fault::Transient => Some(input_reader.inject_fault(snn.clone(), &input_matrices, (FaultType::Transient(broken_bit, time), UnitInputGenerator::NeuronInput(broken_neuron)))),
                        }
                    } else { None };
                    match ft {
                        Fault::StuckAtZero => { (FaultType::StuckAtZero(broken_bit), TestedUnit::NeuronInput(broken_neuron, faulted_inputs)) }
                        Fault::StuckAtOne => { (FaultType::StuckAtOne(broken_bit), TestedUnit::NeuronInput(broken_neuron, faulted_inputs)) }
                        Fault::Transient => { (FaultType::Transient(broken_bit, time), TestedUnit::NeuronInput(broken_neuron, faulted_inputs)) }
                    }
                }
                Unit::NeuronOutput => {
                    let faulted_inputs = if snn.neurons[broken_neuron].layer == 0 {
                        match ft {
                            Fault::StuckAtZero => Some(input_reader.inject_fault(snn.clone(), &input_matrices, (FaultType::StuckAtZero(broken_bit), UnitInputGenerator::NeuronOutput(broken_neuron)))),
                            Fault::StuckAtOne => Some(input_reader.inject_fault(snn.clone(), &input_matrices, (FaultType::StuckAtOne(broken_bit), UnitInputGenerator::NeuronOutput(broken_neuron)))),
                            Fault::Transient => Some(input_reader.inject_fault(snn.clone(), &input_matrices, (FaultType::Transient(broken_bit, time), UnitInputGenerator::NeuronOutput(broken_neuron)))),
                        }
                    } else { None };
                    match ft {
                        Fault::StuckAtZero => { (FaultType::StuckAtZero(broken_bit), TestedUnit::NeuronOutput(broken_neuron, faulted_inputs)) }
                        Fault::StuckAtOne => { (FaultType::StuckAtOne(broken_bit), TestedUnit::NeuronOutput(broken_neuron, faulted_inputs)) }
                        Fault::Transient => { (FaultType::Transient(broken_bit, time), TestedUnit::NeuronOutput(broken_neuron, faulted_inputs)) }
                    }
                }
            };
            let arc_fault = Arc::new(Some(fault));
            for (input_n, input) in input_matrices.iter().enumerate() {
                join_set.spawn(Self::run(input_n, snn.clone(), delta, input.clone(), arc_fault.clone(), f));
            }
        }

        for (input_n, input) in input_matrices.iter().enumerate() {
            join_set.spawn(Self::run(input_n, snn.clone(), delta, input.clone(), none_arc.clone(), f));
        }

        while let Some(res) = join_set.join_next().await {
            if let Ok((input_id, output, fault)) = res {
                match &*fault {
                    None => {
                        match tree.get_mut(&input_id) {
                            None => {
                                tree.insert(input_id, Output {
                                    input_id,
                                    expected_output: expected_outputs[input_id],
                                    no_fault_output: output,
                                    with_fault_output: vec![],
                                });
                            }
                            Some(val) => {
                                val.no_fault_output = output;
                            }
                        }
                    }
                    Some(ft) => {
                        match tree.get_mut(&input_id) {
                            None => {
                                tree.insert(input_id, Output {
                                    input_id,
                                    expected_output: expected_outputs[input_id],
                                    no_fault_output: vec![],
                                    with_fault_output: vec![OutputFaulted {
                                        output,
                                        fault: (*ft).0.clone(),
                                        unit: match (*ft).1 {
                                            TestedUnit::SynapseWeight(a, b, c, _) => {
                                                TestedUnit::SynapseWeight(a, b, c, None)
                                            }
                                            TestedUnit::NeuronInput(a, _) => {
                                                TestedUnit::NeuronInput(a, None)
                                            }
                                            TestedUnit::NeuronOutput(a, _) => {
                                                TestedUnit::NeuronOutput(a, None)
                                            }
                                            _ => { (*ft).1.clone() }
                                        },
                                    }],
                                });
                            }
                            Some(val) => {
                                val.with_fault_output.push(OutputFaulted {
                                    output,
                                    fault: (*ft).0.clone(),
                                    unit: match (*ft).1 {
                                        TestedUnit::SynapseWeight(a, b, c, _) => {
                                            TestedUnit::SynapseWeight(a, b, c, None)
                                        }
                                        TestedUnit::NeuronInput(a, _) => {
                                            TestedUnit::NeuronInput(a, None)
                                        }
                                        TestedUnit::NeuronOutput(a, _) => {
                                            TestedUnit::NeuronOutput(a, None)
                                        }
                                        _ => { (*ft).1.clone() }
                                    },
                                })
                            }
                        }
                    }
                }
            }
        }
        tree.into_iter().map(|(_, v)| v).collect()
    }


    async fn run(input_id: usize, snn: Arc<Snn>, delta: f32, mut input_matrix: Arc<Vec<Vec<f32>>>, fault: Arc<Option<(FaultType, TestedUnit)>>, f: fn(&NeuronConfig, &NeuronInferenceParams, i32, f32, Box<dyn Fn(f32, f32) -> f32>, Box<dyn Fn(f32, f32) -> f32>, Box<dyn Fn(f32, f32) -> Ordering>) -> (f32, bool)) -> (usize, Vec<Vec<bool>>, Arc<Option<(FaultType, TestedUnit)>>) {

        let mut ret = Vec::new();
        let (mut neurons, offset) = NeuronInferenceParams::new(&*snn);
        let mut messages = Message::new(&*snn);

        let mut faulted_neuron_config = NeuronConfig{
            rest_potential: snn.neuron_config.rest_potential,
            threshold_potential: snn.neuron_config.threshold_potential,
            reset_potential: snn.neuron_config.reset_potential,
            time_constant: snn.neuron_config.time_constant,
        };
        if let Some(x) = &*fault {
            let (ft, u) = x;
            match u {
                TestedUnit::SynapseWeight(_, _, _, new_inputs) => {
                    if let Some(val) = new_inputs {
                        input_matrix = val[input_id].clone();
                    }
                }
                TestedUnit::RestPotential(_) => {
                    match ft {
                        FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                            faulted_neuron_config.rest_potential = Self::add_fault_to_float(faulted_neuron_config.rest_potential, ft);
                        }
                        _ => {}
                    }
                }
                TestedUnit::ThresholdPotential(_) => {
                    match ft {
                        FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                            faulted_neuron_config.threshold_potential = Self::add_fault_to_float(faulted_neuron_config.threshold_potential, ft);
                        }
                        _ => {}
                    }
                }
                TestedUnit::ResetPotential(_) => {
                    match ft {
                        FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                            faulted_neuron_config.reset_potential = Self::add_fault_to_float(faulted_neuron_config.rest_potential, ft);
                        }
                        _ => {}
                    }
                }
                TestedUnit::NeuronInput(_, new_inputs) => {
                    if let Some(val) = new_inputs{
                        input_matrix = val[input_id].clone();
                    }
                }
                TestedUnit::NeuronOutput(_, new_inputs) => {
                    if let Some(val) = new_inputs{
                        input_matrix = val[input_id].clone();
                    }
                }
                _ => {}
            }
        }

        let last_layer_first_neuron_id = snn.layers[snn.layers.len() - 1].neurons[0];

        let mut t = 0;
        let last_layer_n_neurons = snn.layers.last().unwrap().neurons.len();

        for input in input_matrix.iter() {
            ret.push(vec![false; last_layer_n_neurons]);
            t += 1;

            for (neuron_id, weight) in input.iter().enumerate() {
                let n = &mut neurons[neuron_id];
                n.input += weight; //NEURON OUTPUT
                n.triggered = true;
            }

            if let Some((ft, u)) = &*fault{
                match u {
                    TestedUnit::RestPotential(_) => {
                        if let FaultType::Transient(_, time) = ft{
                            if *time == t{
                                faulted_neuron_config.rest_potential = Self::add_fault_to_float(faulted_neuron_config.rest_potential, ft);
                            }
                        }
                    }
                    TestedUnit::ThresholdPotential(_) => {
                        if let FaultType::Transient(_, time) = ft{
                            if *time == t{
                                faulted_neuron_config.threshold_potential = Self::add_fault_to_float(faulted_neuron_config.threshold_potential, ft);
                            }
                        }
                    }
                    TestedUnit::ResetPotential(_) => {
                        if let FaultType::Transient(_, time) = ft{
                            if *time == t{
                                faulted_neuron_config.reset_potential = Self::add_fault_to_float(faulted_neuron_config.reset_potential, ft);
                            }
                        }
                    }
                    TestedUnit::Potential(n) => {
                        match ft {
                            FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                                neurons[n - offset].potential = Self::add_fault_to_float(neurons[n - offset].potential, ft);
                            }
                            FaultType::Transient(_, time) => {
                                if *time == t {
                                    neurons[n - offset].potential = Self::add_fault_to_float(neurons[n - offset].potential, ft);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            let mut starting_neuron = 0;

            for layer_id in 1..snn.layers.len(){
                for i in starting_neuron..starting_neuron + snn.layers[layer_id].neurons.len() {
                    if neurons[i].triggered {
                        if let Some((ft, TestedUnit::NeuronInput(neuron, new_input))) = &*fault {
                            if new_input.is_none() && i == neuron - offset {
                                match ft {
                                    FaultType::StuckAtZero(_) | FaultType::StuckAtOne(_) => {
                                        neurons[i].input = Self::add_fault_to_float(neurons[i].input, ft);
                                    }
                                    FaultType::Transient(_, time) => {
                                        if *time == t {
                                            neurons[i].input = Self::add_fault_to_float(neurons[i].input, ft);
                                        }
                                    }
                                }
                            }
                        }

                        let (new_potential, triggered) =
                            match &*fault {
                                None => f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                                Some(x) => {
                                    let (fault_type, unit) = x;

                                    let mut skip = false;
                                    if let FaultType::Transient(_, time) = fault_type {
                                        skip = t != *time;
                                    }
                                    if skip {
                                        f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b)))
                                    }
                                    else {
                                        match unit {
                                            TestedUnit::Adder(selector, n) => {
                                                if i+offset == *n {
                                                    let fault_type = fault_type.clone();
                                                    match selector {
                                                        OpSelector::FirstOperand => f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| Self::add_fault_to_float(a, &fault_type.clone()).add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                                                        OpSelector::SecondOperand => f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(Self::add_fault_to_float(b, &fault_type.clone()))), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                                                        OpSelector::Result => f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| Self::add_fault_to_float(a.add(b), &fault_type.clone())), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                                                    }
                                                }
                                                else { f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))) }
                                            }
                                            TestedUnit::Multiplier(selector, n) => {
                                                if i+offset == *n {
                                                    let fault_type = fault_type.clone();
                                                    match selector {
                                                        OpSelector::FirstOperand => f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| Self::add_fault_to_float(a, &fault_type.clone()).mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                                                        OpSelector::SecondOperand => f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(Self::add_fault_to_float(b, &fault_type.clone()))), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                                                        OpSelector::Result => f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| Self::add_fault_to_float(a.mul(b), &fault_type.clone())), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                                                    }
                                                }
                                                else { f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))) }
                                            }
                                            TestedUnit::Comparator(selector, n) => {
                                                let fault_type = fault_type.clone();
                                                if i+offset == *n {
                                                    match selector {
                                                        OpSelector::FirstOperand => f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| Self::add_fault_to_float(a, &fault_type.clone()).total_cmp(&b))),
                                                        OpSelector::SecondOperand => f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&Self::add_fault_to_float(b, &fault_type.clone())))),
                                                        OpSelector::Result => f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))),
                                                    }
                                                }
                                                else { f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))) }
                                            }
                                            TestedUnit::RestPotential(n) | TestedUnit::ResetPotential(n) | TestedUnit::ThresholdPotential(n) => {
                                                if i+offset == *n{
                                                    f(&faulted_neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b)))
                                                }
                                                else { f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b))) }
                                            }
                                            _ => f(&snn.neuron_config, &neurons[i], t, delta, Box::new(move |a: f32, b: f32| a.add(b)), Box::new(move |a: f32, b: f32| a.mul(b)), Box::new(move |a: f32, b: f32| a.total_cmp(&b)))
                                        }
                                    }

                                }
                            };

                        messages[i].message_received = true;
                        messages[i].new_potential = new_potential;
                        messages[i].triggered = triggered;

                    }
                    neurons[i].input = 0.0;
                    neurons[i].triggered = false;
                }

                let mut neuron_found = false;
                for i in starting_neuron..starting_neuron + snn.layers[layer_id].neurons.len() {

                    if !messages[i].message_received{
                        continue;
                    }
                    messages[i].message_received = false;

                    neurons[i].potential = messages[i].new_potential;
                    neurons[i].last_activity = t;

                    if messages[i].triggered {

                        //NEURON OUTPUT TEST
                        if let Some(x) = &*fault {
                            let (ft, u) = x;
                            match u {
                                TestedUnit::SynapseWeight(n, s, sl, new_input) => {
                                    if new_input.is_none() &&  i + offset == *n {
                                        let mut skip = false;
                                        if let FaultType::Transient(_, time) = ft{
                                            skip = *time != t;
                                        }

                                        if !skip {
                                            neurons[i].input += if *sl {
                                                -snn.neurons[*n].same_layer_synapses[*s].weight
                                                    +Self::add_fault_to_float(snn.neurons[*n].same_layer_synapses[*s].weight, ft)
                                            } else {
                                                -snn.neurons[*n].next_layer_synapses[*s].weight
                                                    +Self::add_fault_to_float(snn.neurons[*n].next_layer_synapses[*s].weight, ft)
                                            };
                                        }
                                    }
                                }
                                TestedUnit::NeuronOutput(n, new_input) => {
                                    if new_input.is_none() && i + offset == *n{
                                        match ft{
                                            FaultType::StuckAtZero(_) => {
                                                continue;
                                                //acting like the neuron was not triggered
                                            }
                                            FaultType::StuckAtOne(_) => {
                                                neuron_found = true;
                                                //the neuron was triggered, so later there's no need to act like it was triggered
                                            }
                                            FaultType::Transient(_, time) => {
                                                if *time == t {
                                                    neuron_found = true;
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }

                        for synapse in snn.neurons[i + offset].next_layer_synapses.iter() {
                            let n = &mut neurons[synapse.to - offset];
                            n.input += synapse.weight; //NEURON OUTPUT
                            n.triggered = true;
                        }
                        for synapse in snn.neurons[i + offset].same_layer_synapses.iter() {
                            let n = &mut neurons[synapse.to - offset];
                            n.input += synapse.weight; //NEURON OUTPUT
                            n.triggered = true;
                        }
                        if i + offset >= last_layer_first_neuron_id {
                            ret[t as usize - 1][i + offset - last_layer_first_neuron_id] = true;
                        }
                    }
                }

                if let Some((ft, TestedUnit::NeuronOutput(neuron, _))) = &*fault{

                    let mut trigger = false;

                    match ft{
                        FaultType::StuckAtZero(_) => {}
                        FaultType::StuckAtOne(_) => {
                            trigger = !neuron_found;
                        }
                        FaultType::Transient(_, time) => {
                            trigger = (*time == t) && !neuron_found
                        }
                    }

                    if trigger{
                        for synapse in snn.neurons[*neuron].next_layer_synapses.iter() {
                            let n = &mut neurons[synapse.to - offset];
                            n.input += synapse.weight; //NEURON OUTPUT
                            n.triggered = true;
                        }
                        for synapse in snn.neurons[*neuron].same_layer_synapses.iter() {
                            let n = &mut neurons[synapse.to - offset];
                            n.input += synapse.weight; //NEURON OUTPUT
                            n.triggered = true;
                        }
                        if *neuron >= last_layer_first_neuron_id {
                            ret[t as usize - 1][*neuron - last_layer_first_neuron_id] = true;
                        }
                    }
                }

                starting_neuron += snn.layers[layer_id].neurons.len()
            }


        }

        (input_id, ret, fault.clone())
    }
}
#[derive(Serialize, Deserialize)]
struct Layer {
    id: usize,
    neurons: Vec<usize>,
}

impl Layer {
    fn add_neuron(&mut self, neuron_id: usize) {
        self.neurons.push(neuron_id);
    }
}

/// Neuron parameters
///
/// * threshold_potential: neuron threshold potential value
/// * rest_potential: neuron rest potential value
/// * reset_potential: neuron reset potential value
/// * time_constant: neuron time constant value
#[derive(Serialize, Deserialize)]
pub struct Neuron {
    id: usize,
    layer: usize,
    next_layer_synapses: Vec<Synapses>,
    same_layer_synapses: Vec<Synapses>,
}


///Neuron variable parameters
///
/// * id : neuron identifier
/// * potential : neuron potential value
/// * input : the input signal of the neuron in the step (sum of the activated synapses' weights)
/// * last_activity : last step when  the neuron potential was recalculated
///   (in the activation function this value can be used with the current_step to calculate the steps passed)
///
pub struct NeuronInferenceParams {
    pub id: usize,
    pub potential: f32,
    pub last_activity: i32,
    pub input: f32,
    triggered: bool,
}

impl NeuronInferenceParams {
    fn new(snn: &Snn) -> (Vec<NeuronInferenceParams>, usize) {
        let offset = snn.layers[0].neurons.len();
        let mut vec = Vec::with_capacity(snn.neurons.len() - offset);
        for i in offset..snn.neurons.len() {
            vec.push(Self {
                id: i,
                potential: 0.0,
                last_activity: 0,
                input: 0.0,
                triggered: false,
            })
        }
        (vec, offset)
    }
}


#[derive(Serialize, Deserialize)]
pub struct Synapses {
    to: usize,
    weight: f32,
}

#[derive(Clone)]
struct Message {
    message_received : bool,
    new_potential: f32,
    triggered: bool,
}

impl Message{
    fn new(snn: &Snn)-> Vec<Message>{
        let ret = vec![Message{
            message_received: false,
            new_potential: 0.0,
            triggered: false
        }; snn.neurons.len()-snn.layers[0].neurons.len()];
        ret
    }
}

/// Output of a test function
///
/// * input_id: input identifier
/// * expected_output: output value expected
/// * no_fault_output: output matrix with no fault added
/// * with_fault_added: list of outputs with fault added
///
pub struct Output {
    pub input_id: usize,
    pub expected_output: ExpectedOutput,
    pub no_fault_output: Vec<Vec<bool>>,
    pub with_fault_output: Vec<OutputFaulted>,
}


/// Output with fault added
///
/// * output: output matrix obtained after the fault were added
/// * fault: type of fault added
/// * unit: unit whom fault was added
///
pub struct OutputFaulted {
    pub output: Vec<Vec<bool>>,
    pub fault: FaultType,
    pub unit: TestedUnit,
}

///
/// Pretty print of test function result
///
/// # Arguments
///
/// v : list of output object
///
pub fn print_output(v: &Vec<Output>) {
    for r in v.iter() {
        println!("\nInput #{} (n°{} expected)", r.input_id, r.expected_output.value);
        println!("\n#################\n\nNO FAULT OUTPUT:\n");
        let mut vv = vec![0; r.no_fault_output[0].len()];
        for a in &r.no_fault_output {
            for (bb, b) in a.iter().enumerate() {
                if *b{ vv[bb] += 1}
            }
        }
        println!("{:?}", vv);
        let mut m = 0;
        for ii in 0..vv.len() {
            if vv[ii] > vv[m] { m = ii; };
        }
        println!("\nVALUE: {}", m);
        if m != r.expected_output.value as usize {
            println!("\nMISMATCHED RESULTS");
        }
        println!("\n#################\n\nFAULTED OUTPUTS:");
        for t in &r.with_fault_output {
            println!("\nadding fault: {} at unit: {}: ", t.fault, t.unit);
            let mut vv = vec![0; r.no_fault_output[0].len()];
            for a in &t.output {
                for (bb, b) in a.iter().enumerate() {
                    if *b{ vv[bb] += 1}
                }
            }
            println!("{:?}", vv);
            let mut m2 = 0;
            for ii in 0..vv.len() {
                if vv[ii] > vv[m2] { m2 = ii; };
            }
            println!("\nVALUE: {}", m2);

            if m != m2 { println!("\nERR"); }
        }
    }
}

#[derive(Copy, Clone)]
pub struct ExpectedOutput {
    pub(crate) value: u8,
}

impl ExpectedOutput {
    /// Create new expected output vector for the Spiral Neural Network from numpy file
    /// # Arguments:
    /// * path : path to the numpy file
    pub fn from_numpy(path: &str) -> Result<Vec<ExpectedOutput>, ReadNpyError> {
        let outputs_raw: Array1<i64> = read_npy(path)?;
        let ret = outputs_raw.iter().map(|x| ExpectedOutput { value: u8::from_i64(*x).unwrap() }).collect();
        Ok(ret)
    }
    /// Create a dummy expected output vector
    /// # Arguments:
    /// * n_inputs : number of inputs to be used
    #[allow(dead_code)]
    pub fn dummy(n_inputs: usize) -> Vec<ExpectedOutput> { vec![ExpectedOutput { value: 0 }; n_inputs] }
}



use std::cmp::Ordering;
use std::sync::Arc;

mod renderer;
mod network;
use crate::network::{Snn, ExpectedOutput, Fault, Unit, NeuronConfig, NeuronInferenceParams, InputReader};


#[tokio::main()]
async fn main() {

    println!("Loading network...");

    let snn = Arc::new(
        Snn::from_numpy(
            vec!["./src/network_params/weights1.npy", "./src/network_params/weights2.npy"],
            NeuronConfig::new(0.0, 1.0, 0.0, 0.9375)
        ).unwrap()
    );

    let expected_outputs = ExpectedOutput::from_numpy("./src/network_params/outputs.npy").unwrap();

    let f = |neuron_config: &NeuronConfig, neuron_params: &NeuronInferenceParams, current_step: i32, delta: f32, testing_add: Box<dyn Fn(f32, f32) -> f32>, testing_mul: Box<dyn Fn(f32, f32) -> f32>, testing_cmp: Box<dyn Fn(f32, f32) -> Ordering>| {

        let a = testing_add(neuron_params.potential, -neuron_config.rest_potential);
        let b = testing_mul(testing_add(current_step as f32, -neuron_params.last_activity as f32), delta);
        let c = (-b / neuron_config.time_constant).exp();
        let mut new_potential = testing_add(testing_add(neuron_config.rest_potential, testing_mul(a, c)), neuron_params.input);
        let triggered = testing_cmp(new_potential, neuron_config.threshold_potential) == Ordering::Greater;

       if triggered { new_potential = neuron_config.reset_potential};   //document implementation
        // if triggered {
        //     new_potential = testing_add(neuron_params.potential, -neuron_config.threshold_potential) }; //snn torch implementation


        (new_potential, triggered)
    };

    let faults_to_add = vec![
         (Fault::StuckAtOne, Unit::Adder),
         (Fault::StuckAtOne, Unit::Multiplier),
    ];

    let input_reader = InputReader::from_numpy("./src/network_params/inputs.npy", None).unwrap();

    // let start = Instant::now();
    let result = Snn::test(snn.clone(), input_reader, faults_to_add, 25, 1.0, f, expected_outputs).await;
    // let elapsed = start.elapsed();
    // println!("Elapsed time: {:?}", elapsed);

    println!("Generating html output...");
    renderer::render_to_html(&result,"./src/templates",false).unwrap();
    network::print_output(&result);

}

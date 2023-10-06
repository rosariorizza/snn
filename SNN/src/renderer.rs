use std::fs;
use std::path::Path;
use handlebars::{Handlebars, RenderError};
use crate::network::Output;
use serde::{Serialize, Deserialize};
use text_io::read;

const AVG_INFERENCE_STORAGE: f32 = 0.00612;
const MAX_STORAGE_LIMIT: f32 = 200.0;
const WARNING_STORAGE_LIMIT: f32 = 20.0;
const SINGLE_PAGE_STORAGE_LIMIT: f32 = 2.0;



#[derive(Serialize, Deserialize)]
struct OutputWithSumsJson{
    input_id: usize,
    expected_output: u8,
    no_fault_sum: Vec<i32>,
    faulted_sum: Vec<OutputFaultedWithSumsJson>,
    signals: Vec<OutputJson>
}

#[derive(Serialize, Deserialize)]
struct OutputFaultedWithSumsJson{
    different: bool,
    fault: String,
    values: Vec<i32>
}

#[derive(Serialize, Deserialize)]
struct OutputJson {
    no_fault: Vec<bool>,
    faulted: Vec<OutputFaultedJson>,
}

#[derive(Serialize, Deserialize)]
struct OutputFaultedJson {
    different: bool,
    fault: String,
    values: Vec<bool>,
}

fn to_json(output: &Output) -> OutputWithSumsJson {
    let mut ret = OutputWithSumsJson{
        input_id: output.input_id,
        expected_output: output.expected_output.value,
        no_fault_sum: vec![0; output.no_fault_output[0].len()],
        faulted_sum: vec![],
        signals: vec![],
    };
    for nf in output.no_fault_output.iter(){
        for (bb,b) in nf.iter().enumerate(){
            if *b {
                ret.no_fault_sum[bb]+=1;
            }
        }
    }
    for wf_outer in output.with_fault_output.iter(){
        let mut tmp = OutputFaultedWithSumsJson{ different: false, fault:  format!("{} at unit: {}", wf_outer.fault, wf_outer.unit).to_string(), values: vec![0; output.no_fault_output[0].len()] };
        for wf in wf_outer.output.iter(){
            for (bb,b) in wf.iter().enumerate(){
                if *b {
                    tmp.values[bb]+=1;
                }
            }
        }
        for bb in 0..ret.no_fault_sum.len(){
            if ret.no_fault_sum[bb] != tmp.values[bb]{
                tmp.different = true;
                break;
            }
        }
        ret.faulted_sum.push(tmp);
    }
    for i in 0..output.no_fault_output.len() {
        let no_fault = output.no_fault_output[i].clone();
        let mut faulted = vec![];
        for wf in output.with_fault_output.iter() {
            let mut different = false;
            for j in 0..no_fault.len() {
                if wf.output[i][j] != no_fault[j] { different = true; }
            }
            faulted.push(
                OutputFaultedJson { different, fault: format!("{} at unit: {}", wf.fault, wf.unit).to_string(), values: wf.output[i].clone() }
            );
        }
        ret.signals.push(OutputJson { no_fault, faulted });
    }
    ret
}

/// Create new html data visualization
///
/// NOTE : If more than 200MB of storage is required the files won't be generated
/// # Arguments:
/// * data : vector of outputs to display
/// * output_path : output HTML file path
/// * index_file : use single page instead of directory with index page (if more than 2MB of storage is required, directory structure will be anyway used)
pub fn render_to_html(data: &Vec<Output>, output_path: &str, single_page: bool)->Result<(), RenderError>{
    let mut reg = Handlebars::new();

    let storage_needed = AVG_INFERENCE_STORAGE*data.len() as f32*data[0].with_fault_output.len() as f32;
    if storage_needed > MAX_STORAGE_LIMIT{
        println!("Storage required to generate files ({}MB) exceeded maximum storage limit ({}MB)", storage_needed, MAX_STORAGE_LIMIT)

    }
    if storage_needed > WARNING_STORAGE_LIMIT{
        println!("About {}MB of storage are required. Proceed?\n [Y]/[N]", storage_needed);
        let line: String = read!("{}\n");
        if line != "y" && line != "Y"{
            println!("Exiting...");
            return Ok(());
        }
        println!("Proceeding...");
    }

    let data: Vec<OutputWithSumsJson> = data.iter().map(|x|to_json(x)).collect();

    if !single_page || storage_needed>SINGLE_PAGE_STORAGE_LIMIT{
        reg.register_template_file("input", "./src/templates/template_input.hbs")?;
        reg.register_template_file("index", "./src/templates/template_index.hbs")?;

        match reg.render("index", &data) {
            Ok(x) => {
                let mut path = output_path.to_owned() + "/output";
                let mut c = 0;

                while Path::new(&format!("{}{}", path, c.to_string())).is_dir() {
                    c+=1;
                }

                path.push_str(&c.to_string());
                fs::create_dir(&path)?;

                fs::write(format!("{}/index.html", &path), x)?;
                for i in data{
                    match reg.render("input", &i) {
                        Ok(y) => {
                            fs::write(format!("{}/input{}.html", path, i.input_id), y)?;}
                        Err(y) =>  {return Err(y) }
                    }
                }
            }
            Err(x) => { return Err(x) }
        }
            }
    else {
        reg.register_template_file("single_file", "./src/templates/template_single_page.hbs")?;
        match reg.render("single_file", &data) {
            Ok(x) => {
                fs::write(output_path.to_owned()+"/output.html", x)?;
            }
            Err(x) => { return Err(x) }
        }
    }
    Ok(())
}

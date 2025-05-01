use serde::{Deserialize, Serialize};

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Welcome to Aquarium, {}! Your neural network journey begins here.", name)
}

#[derive(Debug, Serialize, Deserialize)]
struct LayerParams {
    shape: Option<Vec<i32>>,
    filters: Option<i32>,
    kernel_size: Option<Vec<i32>>,
    pool_size: Option<Vec<i32>>,
    units: Option<i32>,
    padding: Option<String>,
    activation: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Layer {
    id: String,
    name: String,
    layer_type: String,
    params: LayerParams,
}

#[derive(Debug, Serialize, Deserialize)]
struct ShapeResult {
    input_shape: Option<Vec<i32>>,
    output_shape: Vec<i32>,
    parameters: i64,
    memory_bytes: i64,
}

/// Calculate the output shape for a layer based on its input shape and parameters
#[tauri::command]
fn propagate_shape(layer: Layer, input_shape: Option<Vec<i32>>) -> Result<ShapeResult, String> {
    let output_shape: Vec<i32>;
    let parameters: i64;

    match layer.layer_type.as_str() {
        "input" => {
            if let Some(shape) = layer.params.shape {
                // Add batch dimension
                let mut full_shape = vec![0]; // 0 represents None/null in batch dimension
                full_shape.extend(shape.iter().cloned());
                output_shape = full_shape;
                parameters = 0;
            } else {
                return Err("Input layer requires shape parameter".to_string());
            }
        },
        "conv2d" => {
            if input_shape.is_none() {
                return Err("Conv2D layer requires input shape".to_string());
            }

            let input = input_shape.unwrap();
            if input.len() != 4 {
                return Err("Conv2D requires 4D input (batch, height, width, channels)".to_string());
            }

            let filters = layer.params.filters.unwrap_or(1);
            let kernel_size = layer.params.kernel_size.unwrap_or(vec![3, 3]);
            let padding = layer.params.padding.unwrap_or("valid".to_string());

            let batch = input[0];
            let height = input[1];
            let width = input[2];
            let channels = input[3];

            let output_height;
            let output_width;

            if padding == "same" {
                output_height = height;
                output_width = width;
            } else { // "valid"
                output_height = height - kernel_size[0] + 1;
                output_width = width - kernel_size[1] + 1;
            }

            output_shape = vec![batch, output_height, output_width, filters];
            parameters = (kernel_size[0] * kernel_size[1] * channels * filters + filters) as i64;
        },
        "maxpool" => {
            if input_shape.is_none() {
                return Err("MaxPooling2D layer requires input shape".to_string());
            }

            let input = input_shape.unwrap();
            if input.len() != 4 {
                return Err("MaxPooling2D requires 4D input (batch, height, width, channels)".to_string());
            }

            let pool_size = layer.params.pool_size.unwrap_or(vec![2, 2]);

            let batch = input[0];
            let height = input[1];
            let width = input[2];
            let channels = input[3];

            let output_height = height / pool_size[0];
            let output_width = width / pool_size[1];

            output_shape = vec![batch, output_height, output_width, channels];
            parameters = 0;
        },
        "flatten" => {
            if input_shape.is_none() {
                return Err("Flatten layer requires input shape".to_string());
            }

            let input = input_shape.unwrap();
            if input.len() < 2 {
                return Err("Flatten requires at least 2D input".to_string());
            }

            let batch = input[0];
            let mut flattened_size = 1;
            for &dim in input.iter().skip(1) {
                flattened_size *= dim;
            }

            output_shape = vec![batch, flattened_size];
            parameters = 0;
        },
        "dense" | "output" => {
            if input_shape.is_none() {
                return Err("Dense layer requires input shape".to_string());
            }

            let input = input_shape.unwrap();
            if input.len() != 2 {
                return Err("Dense requires 2D input (batch, features)".to_string());
            }

            let units = layer.params.units.unwrap_or(1);

            let batch = input[0];
            let features = input[1];

            output_shape = vec![batch, units];
            parameters = (features * units + units) as i64;
        },
        _ => {
            return Err(format!("Unknown layer type: {}", layer.layer_type));
        }
    }

    // Calculate memory usage (assuming float32 - 4 bytes per value)
    let total_elements: i64 = output_shape.iter().skip(1).map(|&x| x as i64).product();
    let memory_bytes = total_elements * 4; // 4 bytes per float32

    Ok(ShapeResult {
        input_shape,
        output_shape,
        parameters,
        memory_bytes,
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, propagate_shape])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

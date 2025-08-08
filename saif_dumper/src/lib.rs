#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::collections::HashMap;
use serde_json::Value;
use std::fmt;
use std::fs::File;
use std::io::{Write, Read};
use chrono::Utc;
use pyo3::prelude::*;
use lazy_static::lazy_static;
use regex::Regex;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// SAIF structure for storing parsed SAIF file information
pub struct SaifStruct {
 /// Timescale from the SAIF file header (normalized to 1 ps)
 pub timescale: f64,
 /// Duration of the test from the SAIF file header
 pub duration: f64,
 /// Dictionary containing SAIF data (string keys with 4-integer values: T0, T1, TX, TC)
 pub SAIFdict: HashMap<String, (i64, i64, i64, i32)>,
}

impl SaifStruct {
 /// Create a new SaifStruct with default values
 pub fn new() -> Self {
  Self {
   timescale: 0.0,
   duration: 0.0,
   SAIFdict: HashMap::new(),
  }
 }
}

lazy_static! {
 static ref SAIF_ENTRY_REGEX: Regex = Regex::new(r#"(?s)\(\s*(\S+)\s*\n\s*\(T0\s+(\d+(?:\.\d+)?)\s*\)\s*\(T1\s+(\d+(?:\.\d+)?)\s*\)(?:\s*\(TZ\s+\d+\s*\))?(?:\s*\(TX\s+(\d+(?:\.\d+)?)\s*\))?\s*\n\s*\(TC\s+(\d+)\s*\)(?:\s*\(IG\s+\d+\s*\))?\s*\)"#).unwrap();
 static ref STARTS_WITH_BACKSLASH_REGEX: Regex = Regex::new(r"^\\\d").unwrap();
}

/// Escape string for SAIF format according to specific rules
fn saifEscapedString(string: &str) -> String {
 let mut result = string.to_string();
 
 // Check if string starts with backslash followed by digit
 if STARTS_WITH_BACKSLASH_REGEX.is_match(string) {
  // Replace leading backslash with double backslash
  if result.starts_with('\\') {
   result = format!("\\\\{}", &result[1..]);
  }
 } else {
  // Remove leading backslash if not followed by digit
  if result.starts_with('\\') {
   result = result[1..].to_string();
  }
 }
 
 // Replace special characters
 result = result.replace('[', "\\[").replace(']', "\\]").replace('/', "\\/");
 
 result
}

 /// Parse a SAIF file and extract header information
 pub fn parseSaifFile(file_path: &str) -> Result<SaifStruct, std::io::Error> {
  clilog::init_stderr_color_debug();
  clilog::enable_timer("parse 1 SAIF file");
  clilog::enable_timer("");
  
  let time_parse_saif = clilog::stimer!("parse 1 SAIF file");
  
  let mut file = File::open(file_path)?;
  let mut contents = String::new();
  file.read_to_string(&mut contents)?;
  
  // Find the end of the header (first occurrence of "(INSTANCE")
  let header_end = contents.find("(INSTANCE")
   .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Could not find (INSTANCE in SAIF file"))?;
  
  let header = &contents[..header_end];
  
  // Extract timescale
  let timescale_start = header.find("(TIMESCALE ")
   .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Could not find TIMESCALE in SAIF file"))?;
  let timescale_start = timescale_start + "(TIMESCALE ".len();
  let timescale_end = header[timescale_start..].find(')')
   .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Could not find closing parenthesis for TIMESCALE"))?;
  let timescale_str = header[timescale_start..timescale_start + timescale_end].trim();
  
  // Parse timescale: split into number and unit
  let parts: Vec<&str> = timescale_str.split_whitespace().collect();
  if parts.len() != 2 {
   return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
    format!("Invalid TIMESCALE format: expected 'number unit', got '{}'", timescale_str)));
  }
  
  let time_unit_factor = parts[0].parse::<f64>()
   .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, 
    format!("Could not parse time unit factor '{}' as float", parts[0])))?;
  
  let time_unit_string = parts[1];
  let time_unit = match time_unit_string {
   "ms" => 1e-3,
   "us" => 1e-6,
   "ns" => 1e-9,
   "ps" => 1e-12,
   "fs" => 1e-15,
   _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
    format!("'{}' is not a valid time unit!", time_unit_string))),
  };
  
  // Calculate normalized timescale (normalized to 1 ps)
  let timescale = (time_unit * time_unit_factor) / 1e-12; // 1e-12 is 1 ps
  
  // Extract duration
  let duration_start = header.find("(DURATION ")
   .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Could not find DURATION in SAIF file"))?;
  let duration_start = duration_start + "(DURATION ".len();
  let duration_end = header[duration_start..].find(')')
   .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Could not find closing parenthesis for DURATION"))?;
  let duration_str = header[duration_start..duration_start + duration_end].trim();
  let duration_raw = duration_str.parse::<f64>()
   .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Could not parse DURATION as float"))?;
  
  // Calculate normalized duration (duration * timescale)
  let duration = duration_raw * timescale;
  
  // Parse NET entries from the rest of the file in parallel
  let net_section = &contents[header_end..];
  
  // Collect all regex captures first
  let captures: Vec<_> = SAIF_ENTRY_REGEX.captures_iter(net_section).collect();
  
  // Process captures in parallel
  let saif_dict: HashMap<String, (i64, i64, i64, i32)> = captures.par_iter()
   .map(|cap| {
    let net_name = cap[1].to_string();
    
    // Parse T0
    let t0_raw = cap[2].parse::<f64>()
     .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, 
      format!("Could not parse T0 value '{}' as float", cap[2].to_string())))?;
    
    // Parse T1
    let t1_raw = cap[3].parse::<f64>()
     .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, 
      format!("Could not parse T1 value '{}' as float", cap[3].to_string())))?;
    
    // Parse TX (optional - default to 0 if not present)
    let tx_raw = if cap.get(4).is_some() {
     cap[4].parse::<f64>()
      .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, 
       format!("Could not parse TX value '{}' as float", cap[4].to_string())))?
    } else {
     0.0 // Default TX time if not present
    };
    
    // Parse TC
    let tc_raw = cap[5].parse::<i32>()
     .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, 
      format!("Could not parse TC value '{}' as integer", cap[5].to_string())))?;
    
    // Apply timescale normalization to T0, T1, and TX
    let this_t0 = (t0_raw * timescale) as i64;
    let this_t1 = (t1_raw * timescale) as i64;
    let this_tx = (tx_raw * timescale) as i64;
    let this_tc = tc_raw;
    
    Ok((net_name, (this_t0, this_t1, this_tx, this_tc)))
   })
   .collect::<Result<Vec<_>, std::io::Error>>()?
   .into_iter()
   .collect();
  
  let result = SaifStruct {
   timescale,
   duration,
   SAIFdict: saif_dict,
  };
  
  clilog::finish!(time_parse_saif);
  Ok(result)
 }

 /// Compare two SAIF files and report if they match
 pub fn compareSaifFiles(file1_path: &str, file2_path: &str, tolerance: u32, TCtolerance: u32) -> Result<(), Box<dyn std::error::Error>> {
  clilog::init_stderr_color_debug();
  clilog::enable_timer("SAIF file comparison");
  clilog::enable_timer("parallel SAIFdict comparison");
  clilog::enable_timer("");
  
  let time_saif_comparison = clilog::stimer!("SAIF file comparison");
  println!("Comparing SAIF files:");
  println!("  Golden: {}", file1_path);
  println!("  Compare: {}", file2_path);
  println!("  Tolerance: {}", tolerance);
  println!("  TC Tolerance: {}", TCtolerance);
  // Parse both SAIF files
  let golden1 = parseSaifFile(file1_path)?;
  let compare2 = parseSaifFile(file2_path)?;
  
  // Get available threads for parallel processing
  let _max_threads = rayon::current_num_threads();
  println!("Rayon detects {} threads available", _max_threads);
  
  // Compare duration fields
  let duration_diff = (golden1.duration - compare2.duration).abs();
  if duration_diff > 1.0 {
   panic!("Duration mismatch: golden1={}, compare2={}, difference={}", 
    golden1.duration, compare2.duration, duration_diff);
  }
  println!("Duration comparison: PASS (difference: {})", duration_diff);
  // Compare SAIFdict entries in parallel
  let golden_dict = &golden1.SAIFdict;
  let compare_dict = &compare2.SAIFdict;
  
  // Collect all keys from golden1 for parallel processing
  let keys: Vec<&String> = golden_dict.keys().collect();
  
  // Process keys in parallel using rayon with timing
  let time_parallel_comparison = clilog::stimer!("parallel SAIFdict comparison");
  
  // Process keys in parallel using rayon with timing
  keys.par_iter().for_each(|key| {
   // Check if key exists in compare2
   match compare_dict.get(*key) {
    Some(compare_value) => {
     // Compare the values
     let (golden_t0, golden_t1, golden_tx, golden_tc) = golden_dict.get(*key).unwrap();
     let (compare_t0, compare_t1, compare_tx, compare_tc) = compare_value;
     
     // TC comparison logic based on TCtolerance
     let tc_match = if TCtolerance == 0 {
      // Exact match when TCtolerance is 0
      golden_tc == compare_tc
     } else {
      // Within TCtolerance absolute difference when TCtolerance > 0
      let tc_diff = (golden_tc - compare_tc).abs() as u32;
      tc_diff <= TCtolerance
     };
     
     if !tc_match {
      panic!("TC mismatch for key '{}': golden_tc={}, compare_tc={}, TCtolerance={}", 
       key, golden_tc, compare_tc, TCtolerance);
     }
     
     // T0/T1 comparison logic based on tolerance
     let t0_t1_match = if tolerance == 0 {
      // Exact match when tolerance is 0
      let t0_match = golden_t0 == compare_t0;
      let t1_match = golden_t1 == compare_t1;
      t0_match || t1_match
     } else {
      // Within tolerance amount when tolerance > 0
      let t0_diff = (golden_t0 - compare_t0).abs() as u32;
      let t1_diff = (golden_t1 - compare_t1).abs() as u32;
      t0_diff <= tolerance || t1_diff <= tolerance
     };
     
     if !t0_t1_match {
      panic!("T0/T1 mismatch for key '{}': golden=(T0:{}, T1:{}, TX:{}), compare=(T0:{}, T1:{}, TX:{}), tolerance={}", 
       key, golden_t0, golden_t1, golden_tx, compare_t0, compare_t1, compare_tx, tolerance);
     }
    }
    None => {
     panic!("Missing key '{}' in compare2 SAIF file", key);
    }
   }
  });
  clilog::finish!(time_parallel_comparison);
  
  println!("✅ SUCCESS: SAIF files match!");
  clilog::finish!(time_saif_comparison);
  Ok(())
 }

/// SAIF Dumper structure for handling SAIF (Switching Activity Interchange Format) data
pub struct SaifDumper {
 /// Internal storage for the data
 data: HashMap<String, Value>,
}

impl SaifDumper {
 /// Create a new SAIF Dumper instance
 pub fn new() -> Self {
  Self {
   data: HashMap::new(),
  }
 }

 /// Load data from a Python dictionary (represented as JSON-like structure)
 /// This function takes a HashMap that represents the Python dictionary
 pub fn load_from_dict(&mut self, dict: HashMap<String, Value>) {
  self.data = dict;
 }

 /// Create a SAIF file with the specified parameters
 pub fn create_saif_file(&self, block_name: &str, test_name: &str, test_duration: u64, instance_name: &str, 
                        tc_master: &[u32], t0s_master: &[u64]) -> Result<(), std::io::Error> {
  clilog::init_stderr_color_debug();
  clilog::enable_timer("create SAIF file");
  clilog::enable_timer("");
  
  let time_create_saif = clilog::stimer!("create SAIF file");
  
  // Create filename
  let filename = format!("{}_{}_{}ps.saif", block_name, test_name, test_duration);
  
  // Get current date
  let current_date = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
  
  // Create file and write SAIF header
  let mut file = File::create(&filename)?;
  
  writeln!(file, "(SAIFILE")?;
  writeln!(file, "(SAIFVERSION \"2.0\")")?;
  writeln!(file, "(DIRECTION \"backward\")")?;
  writeln!(file, "(DESIGN )")?;
  writeln!(file, "(DATE \"{}\")", current_date)?;
  writeln!(file, "(VENDOR \"GATSPI Open Source\")")?;
  writeln!(file, "(PROGRAM_NAME \"GATSPI\")")?;
  writeln!(file, "(VERSION \"Open Source\")")?;
  writeln!(file, "(DIVIDER / )")?;
  writeln!(file, "(TIMESCALE 1 ps)")?;
  writeln!(file, "(DURATION {})", test_duration)?;
  writeln!(file, "(INSTANCE {}", instance_name)?;
  writeln!(file, "  (INSTANCE dut")?;
  writeln!(file, "    (NET")?;
  
  // Get available threads for parallel processing
  let max_threads = rayon::current_num_threads();
  println!("Rayon detects {} threads available for SAIF file creation", max_threads);
  
  // Collect all keys for parallel processing
  let keys: Vec<&String> = self.data.keys().collect();
  
  // Process keys in parallel and generate SAIF net entries
  let net_entries: Vec<String> = keys.par_iter()
   .map(|key| {
    let value = self.data.get(*key).unwrap();
    
    // Parse the key as usize to index into the arrays
    let key_index = key.parse::<usize>()
     .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid key format"))?;
    
    // Check bounds for arrays
    if key_index >= tc_master.len() || key_index >= t0s_master.len() {
     return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Array index out of bounds"));
    }
    
    // Extract netName from the value (2nd element of tuple)
    let net_name = if let Value::Array(arr) = value {
     if arr.len() == 2 {
      arr[1].as_str()
       .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, 
        format!("Second element of tuple for key {} is not a string", key)))?
     } else {
      return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
       format!("Value for key {} is not a 2-element tuple (found {} elements)", key, arr.len())));
     }
    } else {
     return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, 
      format!("Value for key {} is not an array/tuple", key)));
    };
    
    // Get T0 and TC values
    let this_t0 = t0s_master[key_index];
    let this_tc = tc_master[key_index];
    
    // Generate SAIF net entry as a single string
    let escaped_net_name = saifEscapedString(net_name);
    
    // Skip printing if escaped_net_name is "1'b0" or "1'b1"
    if escaped_net_name == "1'b0" || escaped_net_name == "1'b1" {
     return Ok(String::new()); // Return empty string to skip this entry
    }
    
    let net_entry = format!("      ({}\n        (T0 {}) (T1 {}) (TX 0)\n        (TC {}) (IG 0)\n      )", 
     escaped_net_name, this_t0, test_duration - this_t0, this_tc);
    
    Ok(net_entry)
   })
   .collect::<Result<Vec<_>, std::io::Error>>()?;
  
  // Use binary reduction to combine all net entries into one string
  let combined_net_entries = net_entries.par_iter()
   .fold(|| String::new(), |mut acc, net_entry| {
    acc.push_str(net_entry);
    acc.push('\n');
    acc
   })
   .reduce(|| String::new(), |mut acc, other| {
    acc.push_str(&other);
    acc
   });
  
  // Write the combined string to file in one operation
  write!(file, "{}", combined_net_entries)?;
  
  // Close the SAIF file structure
  writeln!(file, "    )")?;
  writeln!(file, "  )")?;
  writeln!(file, ")")?;
  writeln!(file, ")")?;
  
  clilog::finish!(time_create_saif);
  Ok(())
 }



 /// Get the number of top-level entries in the dictionary
 pub fn entry_count(&self) -> usize {
  self.data.len()
 }

 /// Check if the dictionary is empty
 pub fn is_empty(&self) -> bool {
  self.data.is_empty()
 }

 /// Get a specific value by key
 pub fn get_value(&self, key: &str) -> Option<&Value> {
  self.data.get(key)
 }

 /// Get all keys as a vector
 pub fn get_keys(&self) -> Vec<&String> {
  self.data.keys().collect()
 }

 /// Process dictionary data in parallel using Rayon (placeholder for future implementation)
 pub fn process_parallel(&self) {
  // TODO: Implement parallel processing with Rayon
  println!("Parallel processing not yet implemented");
 }
}

impl fmt::Display for SaifDumper {
 fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
  writeln!(f, "SAIF Dumper with {} entries:", self.data.len())?;
  for (key, value) in &self.data {
   writeln!(f, "  {}: {}", key, value)?;
  }
  Ok(())
 }
}

impl Default for SaifDumper {
 fn default() -> Self {
  Self::new()
 }
}

// Python module definition
#[pymodule]
fn saif_dumper(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
 m.add_class::<PySaifDumper>()?;
 Ok(())
}

// Python wrapper for SaifDumper
#[pyclass]
pub struct PySaifDumper {
 inner: SaifDumper,
}

#[pymethods]
impl PySaifDumper {
 #[new]
 fn new() -> Self {
  Self {
   inner: SaifDumper::new(),
  }
 }

 /// Load data from a Python dictionary
 fn load_from_dict(&mut self, dict: &Bound<'_, PyAny>) -> PyResult<()> {
  // Convert Python dict to JSON string using json.dumps equivalent
  Python::with_gil(|py| {
   let json_module = py.import_bound("json")?;
   let json_str = json_module.call_method1("dumps", (dict,))?.str()?.to_string();
   let rust_dict: HashMap<String, Value> = serde_json::from_str(&json_str)
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
   self.inner.load_from_dict(rust_dict);
   Ok(())
  })
 }

 /// Create a SAIF file with the specified parameters
 fn create_saif_file(&self, block_name: &str, test_name: &str, test_duration: u64, instance_name: &str,
                     tc_master: &Bound<'_, PyAny>, t0s_master: &Bound<'_, PyAny>) -> PyResult<()> {
  // Convert Python objects to Rust vectors
  let tc_vec: Vec<u32> = tc_master.extract()
   .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("tc_master must be a list or array of uint32 values"))?;
  let t0s_vec: Vec<u64> = t0s_master.extract()
   .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("t0s_master must be a list or array of uint64 values"))?;
  
  self.inner.create_saif_file(block_name, test_name, test_duration, instance_name, &tc_vec, &t0s_vec)
   .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
 }

 /// Get the number of entries
 fn entry_count(&self) -> usize {
  self.inner.entry_count()
 }

 /// Check if empty
 fn is_empty(&self) -> bool {
  self.inner.is_empty()
 }

 /// Get a value by key (simplified - returns string representation)
 fn get_value(&self, key: &str) -> PyResult<Option<String>> {
  if let Some(value) = self.inner.get_value(key) {
   Ok(Some(value.to_string()))
  } else {
   Ok(None)
  }
 }

 /// Get all keys
 fn get_keys(&self) -> Vec<String> {
  self.inner.get_keys().iter().map(|s| (*s).clone()).collect()
 }

 /// Process in parallel (placeholder)
 fn process_parallel(&self) {
  self.inner.process_parallel();
 }

 fn __str__(&self) -> PyResult<String> {
  Ok(format!("{}", self.inner))
 }
}

 
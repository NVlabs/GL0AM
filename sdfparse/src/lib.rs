#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use regex::bytes::Regex;
use lazy_static::lazy_static;
use rayon::prelude::*;
use memchr::memmem;
lazy_static::lazy_static! {
    static ref SEQ_REGEX: regex::Regex = regex::Regex::new(r".*(_DF|_LATCH_|_CLKGATE_|_RAMS_|_SYNC).*").unwrap();
}

#[derive(Debug,serde::Serialize)]
pub struct sdfParse {
 pub designName: String,
 pub timeUnit: f32,
 pub parseInterconnectResults: Vec<Option<(String, String, u32)>>,
 pub parseAllCellsResults: Vec<Option<(String, String, Vec<(Option<Vec<(String, u8)>>, String, String, u32)>)>>
}

pub fn print_type<T>(_: &T) { 
    println!("{:?}", std::any::type_name::<T>());
}

fn find_delimiters_parallel(data: &[u8], delimiter: &[u8]) -> Vec<usize> {
 let max_threads = rayon::current_num_threads(); println!("Rayon detects {} threads available", max_threads);
 let chunk_size = (data.len() + max_threads - 1) / max_threads;
 let delimiter_len = delimiter.len();
 // Calculate how many threads we actually need, ensuring start < data.len()
 let mut num_threads = (data.len() + chunk_size - 1) / chunk_size;
 num_threads = num_threads.min(max_threads);
 // Parallel search for delimiters
 let mut positions: Vec<usize> = (0..num_threads).into_par_iter().flat_map(|i| {
  let start = i * chunk_size;
  let mut end = ((i + 1) * chunk_size).min(data.len());
  // extend to avoid splitting a delimiter
  if end + delimiter_len < data.len() {
   end += delimiter_len;
  }
  // Ensure end doesn't exceed data bounds
  let end = end.min(data.len());
  let slice = &data[start..end];
  memmem::find_iter(slice, delimiter).map(|pos| start + pos).collect::<Vec<_>>()
 }).collect();
 // Results may be out of order due to thread execution order
 positions.sort_unstable();
 positions
}

fn rfind_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
 if needle.is_empty() || haystack.len() < needle.len() {
  return None;
 }
 let mut i = haystack.len() - needle.len();
 loop {
  if &haystack[i..i + needle.len()] == needle {
   return Some(i);
  }
  if i == 0 { break; }
  i -= 1;
 }
 None
}

lazy_static! {
    static ref DESIGN_NAME_REGEX: Regex = Regex::new(r#"\(\s*DESIGN\s+"(.*?)""#).unwrap();
    static ref TIMESCALE_REGEX: Regex = Regex::new(r#"\(\s*TIMESCALE\s+(\d+)([munpf]s)"#).unwrap();
    static ref INTERCONNECT_BLOCKMODULE_REGEX: Regex = Regex::new(r#"\(INSTANCE "#).unwrap();
    static ref INTERCONNECT_REGEX: Regex = Regex::new(r#"\(\s*INTERCONNECT\s+(.*?)\s+(.*?)\s+(\(.*\))\s*\)"#).unwrap();
    static ref CELL_REGEX: Regex = Regex::new(r#"(?s)\(CELLTYPE\s+"(.*?)"\s*\)\s+\(INSTANCE\s+(.*?)\s*\)\s+\(DELAY\s+(?:\(PATHPULSEPERCENT\s+\(\d+\s*\)\s*\)\s*)?\(ABSOLUTE\s+(.*?)\n\s+\)\n\s+\)"#).unwrap();
    static ref COND_DELAY_REGEX: regex::Regex = regex::Regex::new(r#"(?i)\s*\(COND\s+(.*?)\s+\(IOPATH\s+(.*?)\s*(\(\s*[0-9\)].*)"#).unwrap();
    static ref IOPATH_DELAY_REGEX: regex::Regex = regex::Regex::new(r#"(?i)\s*\(IOPATH\s+(.*?)\s*(\(\s*[0-9\)].*)"#).unwrap();
    static ref FLOAT_GROUP_REGEX: regex::Regex = regex::Regex::new(r#"\([^)]*\)"#).unwrap();
}
const DEFAULT_TIMESCALE: f32 = 1e-12  ;//ps, picoseconds

impl sdfParse {

 pub fn readin_file(path: impl AsRef<std::path::Path>) -> Self {
  clilog::init_stderr_color_debug();
  clilog::enable_timer("read SDF");
  clilog::enable_timer("");
  let time_parse_interconnect = clilog::stimer!("parse INTERCONNECT");
  let s = std::fs::read(&path).expect("No such SDF file found!");
  let (designName, remainderIndex, timeUnit) = sdfParse::parseHeader(&s);
  let newlinePositions = find_delimiters_parallel(&s[0..remainderIndex], b"\n");
  let parseInterconnectResults = sdfParse::parseInterconnectDelay(&s, newlinePositions, timeUnit);
  clilog::finish!(time_parse_interconnect);
  let time_parse_cells = clilog::stimer!("parse CELL");
  let cellPositions = find_delimiters_parallel(&s[remainderIndex..], b"(CELL\n");
  let parseAllCellsResults = sdfParse::parseAllCellDelays(&s, cellPositions, remainderIndex, timeUnit);
  clilog::finish!(time_parse_cells);
  // Some simple debug info upon completion of parsing
  let firstInterconnectResult = parseInterconnectResults.iter()
    .filter_map(|result| result.as_ref())
    .next();
  let firstCellResult = parseAllCellsResults.iter()
    .filter_map(|result| result.as_ref())
    .next();
  let firstCellResultWithCond = parseAllCellsResults.iter()
    .filter_map(|result| result.as_ref())
    .find(|(_, _, parsed_results)| {
      parsed_results.iter().any(|(cond, _, _, _)| cond.is_some())
    });
  let firstCellResultWithEdge = parseAllCellsResults.iter()
    .filter_map(|result| result.as_ref())
    .find(|(_, _, parsed_results)| {
      parsed_results.iter().any(|(_, input_pin, _, _)| input_pin.contains("edge"))
    });
  println!("Sample INTERCONNECT delay parsing result: {:?}", firstInterconnectResult);
  println!("Sample INSTANCE delay parsing result: {:?}", firstCellResult);
  println!("Sample INSTANCE delay parsing result with COND: {:?}", firstCellResultWithCond);
  println!("Sample INSTANCE delay parsing result with 'edge' in input pin: {:?}", firstCellResultWithEdge);
  println!("Total interconnect results: {}", parseInterconnectResults.iter().filter_map(|r| r.as_ref()).count());
  println!("Total cell results: {}", parseAllCellsResults.iter().filter_map(|r| r.as_ref()).count());
  println!("Filtered out sequential cells (None results): {}", parseAllCellsResults.iter().filter(|r| r.is_none()).count());
  Self { 
   designName: designName, timeUnit: timeUnit, parseInterconnectResults: parseInterconnectResults, parseAllCellsResults: parseAllCellsResults,
  }
 }
 
 pub fn parseHeader<'a>(s: &'a [u8]) -> (String, usize, f32) {
  let mut captured = DESIGN_NAME_REGEX.captures(&s).expect("Something wrong with SDF file syntax, no DESIGN_NAME in header!");
  let designName = String::from_utf8_lossy(captured.get(1).unwrap().as_bytes()).to_string();
  let mut remainderIndex = captured.get(0).unwrap().end();
  captured = TIMESCALE_REGEX.captures(&s[remainderIndex..]).expect("Something wrong with SDF file syntax, no valid TIMESCALE statement found in header!");
  remainderIndex = captured.get(0).unwrap().end() + remainderIndex;
  let timeUnitString = std::str::from_utf8(captured.get(2).unwrap().as_bytes()).unwrap(); 
  let mut timeUnit = match timeUnitString { "ms" => 1e-3, "us" => 1e-6, "ns" => 1e-9, "ps" => 1e-12, "fs" => 1e-15, 
                              _ => panic!("{} is not a valid time unit!", timeUnitString) , } ; 
  let timeUnitFactor = std::str::from_utf8(captured.get(1).unwrap().as_bytes()).unwrap().parse::<f32>().unwrap(); 
  timeUnit = (timeUnit * timeUnitFactor) / DEFAULT_TIMESCALE ; println!("Using time unit factor of {:e} for all delays to conform to default timescale of {:e}", timeUnit, DEFAULT_TIMESCALE);
  captured = INTERCONNECT_BLOCKMODULE_REGEX.captures(&s[remainderIndex..]).expect("There are no INTERCONNECT statements in this top level SDF. Check it or add a dummy statement.");
  remainderIndex = captured.get(0).unwrap().end() + remainderIndex;
  if let Some(pos) = rfind_subslice(&s[0..remainderIndex], b"(CELL\n") { remainderIndex = pos; } else { panic!("Something wrong with SDF format around first leaf CELL instance!"); }
  (designName, remainderIndex, timeUnit)
 }
 
 pub fn parseDelayGroups(delay_str: &str, timeUnit: f32) -> u32 {
  // Count float groups (patterns matching "(.*)") within delay_str
  let float_groups: Vec<_> = FLOAT_GROUP_REGEX.find_iter(delay_str).collect();
  let float_group_count = float_groups.len();
  // Error handling for float group count
  match float_group_count {
   0 => panic!("Not a valid SDF file line! : {}", delay_str),
   1 => {
    // Single float group logic
    let float_group = &float_groups[0];
    let group_content = float_group.as_str();
    let floats: Vec<f64> = group_content
      .chars()
      .filter(|&c| c.is_digit(10) || c == '.' || c == ':' || c == '-')
      .collect::<String>()
      .split(':')
      .filter(|s| !s.trim().is_empty())
      .filter_map(|s| s.trim().parse::<f64>().ok())
      .collect();
    let float_count = floats.len();
    let intermediate_result = match float_count {
     0 => return 4_294_967_295u32, // Return max u32 value
     1 => (floats[0] * timeUnit as f64) as u32,
     2..=3 => (floats[1] * timeUnit as f64) as u32, // Pick the 2nd float
     _ => return 4_294_967_295u32, // Default case
    };
    // Check for overflow
    if intermediate_result > 65534 {
     println!("Warning: The delay value {} has overflowed. Check your application if it's ok? Group content: {}", intermediate_result, group_content);
    }
    // Create rf_delay with lower and higher 16 bits both equal to intermediate_result
    (intermediate_result << 16) | intermediate_result
   },
   2 => {
    // Two float groups logic
    let mut intermediate_result_r = 65535u32;
    let mut intermediate_result_f = 65535u32;
    // Process first float group
    let first_group = &float_groups[0];
    let first_group_content = first_group.as_str();
    let first_floats: Vec<f64> = first_group_content
      .chars()
      .filter(|&c| c.is_digit(10) || c == '.' || c == ':' || c == '-')
      .collect::<String>()
      .split(':')
      .filter(|s| !s.trim().is_empty())
      .filter_map(|s| s.trim().parse::<f64>().ok())
      .collect();
    let first_float_count = first_floats.len();
    if first_float_count > 0 {
     let first_intermediate = match first_float_count {
      1 => (first_floats[0] * timeUnit as f64) as u32,
      2..=3 => (first_floats[1] * timeUnit as f64) as u32, // Pick the 2nd float
      _ => 65535u32,
     };
     intermediate_result_r = first_intermediate;
    }
    // Process second float group
    let second_group = &float_groups[1];
    let second_group_content = second_group.as_str();
    let second_floats: Vec<f64> = second_group_content
      .chars()
      .filter(|&c| c.is_digit(10) || c == '.' || c == ':' || c == '-')
      .collect::<String>()
      .split(':')
      .filter(|s| !s.trim().is_empty())
      .filter_map(|s| s.trim().parse::<f64>().ok())
      .collect();
    let second_float_count = second_floats.len();
    if second_float_count > 0 {
     let second_intermediate = match second_float_count {
      1 => (second_floats[0] * timeUnit as f64) as u32,
      2..=3 => (second_floats[1] * timeUnit as f64) as u32, // Pick the 2nd float
      _ => 65535u32,
     };
     intermediate_result_f = second_intermediate;
    }
    // Check for overflow
    if (intermediate_result_r > 65534 && first_float_count > 0) || (intermediate_result_f > 65534 && second_float_count > 0) {
     println!("Warning: The delay value {} {} has overflowed. Check your application if it's ok? First group content: {}, Second group content: {}", intermediate_result_r, intermediate_result_f, first_group_content, second_group_content);
    }
    // Create rf_delay with higher 16 bits = intermediate_result_r, lower 16 bits = intermediate_result_f
    (intermediate_result_r << 16) | intermediate_result_f
   },
   _ => panic!("This program is not capable of parsing more than 2 float groups per line! : {}", delay_str),
  }
 }
  
 pub fn parseConditionalContent<'a>(cond_str: &'a str) -> Vec<(String, u8)> {
  let mut result = Vec::new();
  // Split by '&&' delimiter
  let cond_parts: Vec<&str> = cond_str.split("&&").collect();
  for part in cond_parts {
   // Trim whitespace from the part
   let trimmed_part = part.trim();
   // Split by "===1'b"
   let split_result = trimmed_part.split_once("===1'b");
   if let Some((before, after)) = split_result {
    // Clean up the before part: strip whitespace and remove '(' and ')'
    let cleaned_before = before.trim().trim_matches('(').trim_matches(')').to_string();
    // Parse the after part as u8
    if let Ok(value) = after.parse::<u8>() {
     if value <= 1 { // Only accept 0 and 1
      result.push((cleaned_before, value));
     }
    }
   }
  }
  result
 }
  
 pub fn parseInterconnectDelay<'a>(s: &'a [u8], newlinePositions: Vec<usize>, timeUnit: f32) -> Vec<Option<(String, String, u32)>> { 
  let mut chunks: Vec<&[u8]> = Vec::with_capacity(newlinePositions.len());
  for i in 0..newlinePositions.len() {
   if i == 0 { chunks.push(&s[0..newlinePositions[0]]); } else { chunks.push(&s[newlinePositions[i - 1]..newlinePositions[i]]); }
  }
  let parseInterconnectResults: Vec<_> = chunks
   .par_iter()  // Rayon parallel iterator
   .map(|chunk| {
    if let Some(captured) = INTERCONNECT_REGEX.captures(chunk) {
     //Strict: must have all 3 capture groups
     let driver = captured.get(1)? ; let load = captured.get(2)?; let delay = captured.get(3)?;
     let driver_str = std::str::from_utf8(driver.as_bytes()).ok()?.to_string(); 
     let load_str = std::str::from_utf8(load.as_bytes()).ok()?.to_string();	
     let delay_str = std::str::from_utf8(delay.as_bytes()).ok()?;
     let delay_value = sdfParse::parseDelayGroups(delay_str, timeUnit);
     return Some((driver_str, load_str, delay_value));
	}
	None
   })
  .collect();
  parseInterconnectResults
 }

 pub fn parseCell<'a>(chunk: &'a [u8], timeUnit: f32) -> Option<(String, String, Vec<(Option<Vec<(String, u8)>>, String, String, u32)>)> { 
  if let Some(captured) = CELL_REGEX.captures(chunk) {
   let celltype = captured.get(1)? ; let instance = captured.get(2)?; let delay = captured.get(3)?;
   let celltype_str = std::str::from_utf8(celltype.as_bytes()).ok()?.to_string(); 
   
   // Check if celltype_str matches SEQ_REGEX, return None if it does
   if SEQ_REGEX.is_match(&celltype_str) {
    return None;
   }
   
   let instance_str = std::str::from_utf8(instance.as_bytes()).ok()?.to_string();
   let delay_str = std::str::from_utf8(delay.as_bytes()).ok()?;
   // Parse delay_str line by line
   let lines: Vec<&str> = delay_str.split('\n').collect();
   let mut parsed_results = Vec::new();
   for line in lines {
    if line.contains("(COND ") {
     // Use COND_DELAY_REGEX for lines with conditional content
     if let Some(line_captured) = COND_DELAY_REGEX.captures(line) {
      let cond_str = line_captured.get(1).map(|m| m.as_str()).unwrap_or(""); // First capture group (COND)
      let iopath_content = line_captured.get(2).map(|m| m.as_str()).unwrap_or(""); // Second capture group (IOPATH content)
      let delays = line_captured.get(3).map(|m| m.as_str()).unwrap_or(""); // Third capture group (delays including the opening delimiter)
      // Parse conditional content
      let cond = Some(sdfParse::parseConditionalContent(cond_str));
      // Split iopath_content into input_pin and output_pin
      let parts: Vec<&str> = iopath_content.split_whitespace().collect();
      let (input_pin, output_pin) = if parts.len() >= 2 {
       let last_part = parts.last().unwrap();
       // Find the position of the last whitespace to split the string
       let last_whitespace_pos = iopath_content.rfind(' ').unwrap();
       let input_pin = &iopath_content[..last_whitespace_pos];
       (input_pin.to_string(), last_part.to_string())
      } else {
       (iopath_content.to_string(), "".to_string()) // If only one part, treat it as input_pin
      };
      let delay_value = sdfParse::parseDelayGroups(delays, timeUnit);
      parsed_results.push((cond, input_pin, output_pin, delay_value));
     } else {
      panic!("Line with COND does not match the expected pattern: {}", line);
     }
         } else {
      // Use IOPATH_DELAY_REGEX for lines without conditional content
      if let Some(line_captured) = IOPATH_DELAY_REGEX.captures(line) {
       let iopath_content = line_captured.get(1).map(|m| m.as_str()).unwrap_or(""); // First capture group (IOPATH content)
       let delays = line_captured.get(2).map(|m| m.as_str()).unwrap_or(""); // Second capture group (delays including the opening delimiter)
       // Split iopath_content into input_pin and output_pin
       let parts: Vec<&str> = iopath_content.split_whitespace().collect();
       let (input_pin, output_pin) = if parts.len() >= 2 {
        let last_part = parts.last().unwrap();
        // Find the position of the last whitespace to split the string
        let last_whitespace_pos = iopath_content.rfind(' ').unwrap();
        let input_pin = &iopath_content[..last_whitespace_pos];
        (input_pin.to_string(), last_part.to_string())
       } else {
        (iopath_content.to_string(), "".to_string()) // If only one part, treat it as input_pin
       };
       let delay_value = sdfParse::parseDelayGroups(delays, timeUnit);
       parsed_results.push((None, input_pin, output_pin, delay_value)); // None for cond since no COND
      } else {
       panic!("Line without COND does not match the expected pattern: {}", line);
      }
     }
   }
   return Some((celltype_str, instance_str, parsed_results));
  }
  None
 }

 pub fn parseAllCellDelays<'a>(s: &'a [u8], cellPositions: Vec<usize>, remainderIndex: usize, timeUnit: f32) -> Vec<Option<(String, String, Vec<(Option<Vec<(String, u8)>>, String, String, u32)>)>> { 
  let mut chunks: Vec<&[u8]> = Vec::with_capacity(cellPositions.len());
  for i in 0..cellPositions.len()-1 {
   chunks.push(&s[remainderIndex+cellPositions[i]..remainderIndex+cellPositions[i+1]]);
  }
  chunks.push(&s[remainderIndex+cellPositions[cellPositions.len()-1]..]);
  let parseAllCellsResults: Vec<_> = chunks
   .par_iter()  // Rayon parallel iterator
   .map(|chunk| { sdfParse::parseCell(chunk, timeUnit) } )
  .collect();
  parseAllCellsResults
 }
}

#![allow(non_snake_case)]

use netlistdb::*;
use ulib::UVec;
use ulib::Device;
use itertools::Itertools;
use std::iter;
use rayon::prelude::*;
use std::sync::atomic::{AtomicPtr, Ordering};
use sdfparse::sdfParse;
use clilog;

use std::collections::HashMap;
use std::path::Path;


pub mod stdlib_attributes;
pub use stdlib_attributes::*;

#[derive(Debug, Clone,serde::Serialize)]
pub struct GATSPIGraph {
 pub gatspi_cells: Vec<usize>,
 pub gatspi_celltypes: Vec<u16>,
 pub start: Vec<usize>,
 pub items: Vec<usize>,
 pub values: Vec<u8>,
 pub pinid2gatspiid: Vec<usize>,
 pub num_of_gatspi_cells: usize,
 pub SDFLUT: Option<Vec<u32>>,
 pub SDFPointerStart: Option<Vec<usize>>,
 pub SDFPointerEnd: Option<Vec<usize>>,
 pub interconnectDelays: Option<Vec<u32>>,
 //some debug stuff
 pub num_of_top_ports: usize,
 pub gatspi_cellname_index: Vec<(String,String)>,
 pub gatspi_port_index: HashMap<usize, String>,
}

impl GATSPIGraph {

/// Helper function to assign arcDelay to SDFLUT with proper rising/falling delay comparison
/// arcDelay and SDFLUT[i] both contain rising delay in upper 16 bits and falling delay in lower 16 bits
#[allow(dead_code)]
fn assignArcDelayToSDFLUT(sdflut: &mut [u32], index: usize, arc_delay: u32) {
    let current_value = sdflut[index];
    
    // Extract rising delays (upper 16 bits)
    let current_rising = (current_value >> 16) as u16;
    let arc_rising = (arc_delay >> 16) as u16;
    
    // Extract falling delays (lower 16 bits)
    let current_falling = (current_value & 0xFFFF) as u16;
    let arc_falling = (arc_delay & 0xFFFF) as u16;
    
    // Take minimum of rising and falling delays separately
    let min_rising = current_rising.min(arc_rising);
    let min_falling = current_falling.min(arc_falling);
    
    // Combine back into 32-bit value
    sdflut[index] = ((min_rising as u32) << 16) | (min_falling as u32);
}

/// Helper function to assign arcDelay to SDFLUT using raw pointer for parallel processing
/// arcDelay and SDFLUT[i] both contain rising delay in upper 16 bits and falling delay in lower 16 bits
unsafe fn assignArcDelayToSDFLUT_ptr(sdflut_ptr: *mut u32, index: usize, arc_delay: u32) {
    unsafe {
        let current_value = *sdflut_ptr.add(index);
        
        // Extract rising delays (upper 16 bits)
        let current_rising = (current_value >> 16) as u16;
        let arc_rising = (arc_delay >> 16) as u16;
        
        // Extract falling delays (lower 16 bits)
        let current_falling = (current_value & 0xFFFF) as u16;
        let arc_falling = (arc_delay & 0xFFFF) as u16;
        
        // Take minimum of rising and falling delays separately
        let min_rising = current_rising.min(arc_rising);
        let min_falling = current_falling.min(arc_falling);
        
        // Combine back into 32-bit value and write to pointer
        *sdflut_ptr.add(index) = ((min_rising as u32) << 16) | (min_falling as u32);
    }
}

/// Find CSR indices for edges from a driver to multiple loads with specific edge IDs
/// Returns a vector of (load_index, csr_index) pairs where csr_index is the index in items/values arrays
fn findCSRIndices4Loads(
    driver_gatspi_id: usize,
    loads: &[usize],
    edge_ids: &[u8],
    start: &[usize],
    items: &[usize],
    values: &[u8]
) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    
    // Ensure loads and edge_ids have the same length
    if loads.len() != edge_ids.len() {
        return result;
    }
    
    // Get CSR range for the driver
    let start_idx = start[driver_gatspi_id];
    let end_idx = start[driver_gatspi_id + 1];
    
    // Pairwise matching: loads[i] should match with edge_ids[i]
    for (load_idx, &load_node) in loads.iter().enumerate() {
        let edge_id = edge_ids[load_idx];
        
        // Find the CSR index that matches this specific load node and edge ID
        for i in start_idx..end_idx {
            let dest_node = items[i];
            let edge_value = values[i];
            
            if dest_node == load_node && edge_value == edge_id {
                result.push((load_idx, i));
                break; // Found the match for this load, move to next
            }
        }
    }
    
    result
}

/// Parse a pin name string from SDF to find the corresponding pin ID in the netlist database
fn parsePinname2id(db: &NetlistDB, pin_name_str: &str) -> Option<usize> {
 // Parse the pin name string to extract hierarchical name, pin name, and bus index
 // Expected format: "hierarchical/path/pin_name[bus_index]" or "pin_name[bus_index]" or "hierarchical/path/pin_name" or "pin_name"
 // Find the last '/' to separate hierarchical path from pin name
 let last_slash_pos = pin_name_str.rfind('/');
 let (hier_part, pin_part) = match last_slash_pos {
  Some(pos) => {let hier = &pin_name_str[..pos]; let pin = &pin_name_str[pos + 1..]; (hier, pin) }
  None => ("", pin_name_str)
 };
 // Parse bus index from pin part
 let (pin_name, bus_index) = if let Some(bracket_start) = pin_part.find('[') {
  if let Some(bracket_end) = pin_part.find(']') {
   let name = &pin_part[..bracket_start]; let index_str = &pin_part[bracket_start + 1..bracket_end]; let index: isize = index_str.parse().ok()?; (name, Some(index))
  } else {
   return None; // Unmatched bracket
  }
 } else {
  (pin_part, None)
 };
 // Create hierarchical name , Parse hierarchical path and create HierName using the proper API
 let hier_name = if hier_part.is_empty() { netlistdb::HierName::empty() } else { 
  let path_parts: Vec<&str> = hier_part.split('/').filter(|&part| !part.is_empty()).collect();
  netlistdb::HierName::from_topdown_hier_iter(path_parts.iter().map(|&s| s))
 };
 // Look up in pinname2id HashMap
 let pin_key = (hier_name, compact_str::CompactString::new_inline(pin_name), bus_index);
 db.pinname2id.get(&pin_key).copied()
}

 pub fn build_graph(db: &NetlistDB, stdlib_info: &(impl StandardCellTypeAttribute + std::marker::Sync), sdf_path: Option<&Path> ) -> Self {
    let num_ports = db.cell2pin.start[1]; 

  let max_threads = rayon::current_num_threads(); println!("Rayon detects {} threads available", max_threads);
  let range_size = db.num_pins - num_ports; let parallel_stride = (range_size + max_threads - 1) / max_threads;
  //find all the output pin ids (eda infra world)
  let mut these_opins: Vec<usize> = (0..max_threads).into_par_iter().flat_map(|thread_id| {
   let start = num_ports + thread_id * parallel_stride;
   let end = if thread_id == max_threads - 1 { db.num_pins } else { (num_ports + (thread_id + 1) * parallel_stride).min(db.num_pins) };
   (start..end).filter(|&index| db.pindirect[index] == Direction::O).collect::<Vec<usize>>()
  }).collect();
  
  //find all the top ports (input ports) that aren't assigned to a constant value (eda infra world)
  let legit_top_ports = Vec::from_iter( (0..num_ports).filter( |pin| (Some(db.pin2net[*pin]) != db.net_zero) && (Some(db.pin2net[*pin]) != db.net_one) && (db.pindirect[*pin] == Direction::O) ) );    
  these_opins.splice(0..0, legit_top_ports.clone()); 
  let these_opins2 = Vec::from(these_opins);
 
  
  let stride = (these_opins2.len() + max_threads - 1) / max_threads;
  //find all the net ids (eda infra world)
  let mut these_nets : Vec<usize> = (0..max_threads).into_par_iter().flat_map(|thread_id| {
   let start = thread_id * stride; let end = ((thread_id + 1) * stride).min(these_opins2.len());
   (start..end).map(|i| db.pin2net[these_opins2[i]]).collect::<Vec<usize>>()
  }).collect();
  
  // Append net_zero and net_one to these_nets
  if let Some(net_zero) = db.net_zero {
    these_nets.push(net_zero);
  }
  if let Some(net_one) = db.net_one {
    these_nets.push(net_one);
  }
  
  let stride2 = (these_nets.len() + max_threads - 1) / max_threads;
  let mut these_number_of_pins : Vec<usize> = (0..max_threads).into_par_iter().flat_map(|thread_id| {
   let start = thread_id * stride2; let end = ((thread_id + 1) * stride2).min(these_nets.len());
   (start..end).map(|i| db.net2pin.items[db.net2pin.start[these_nets[i]]..db.net2pin.start[these_nets[i]+1]].into_iter().filter(
    |&pin| (db.pindirect[*pin] == Direction::I) && (db.pin2cell[*pin] != 0) ).map( |&pin|  db.cell2noutputs[db.pin2cell[pin]] ).sum()
   ).collect::<Vec<usize>>()
  }).collect();
  these_number_of_pins = these_number_of_pins.into_iter().scan(0, |acc, x : usize| { *acc +=x; Some(*acc) }).collect();
  these_number_of_pins.splice(0..0, [0]);
  //the offsets of how many input pins each output pin drives
  let these_start = Vec::from(these_number_of_pins);

  let mut items: UVec<usize> = UVec::new_zeroed(these_start[these_start.len()-1], Device::CPU); // the GATSPI cell ids that each output pin drives
  let mut values: UVec<u8> = UVec::new_zeroed(these_start[these_start.len()-1], Device::CPU); //the edge id (input pin id) of each driver-load connection
  let max_opin = these_opins2.iter().max().unwrap_or(&0);
  let mut pinid2gatspiid = UVec::new_filled(max_opin+1, max_opin+1, Device::CPU); //translation vector indexed by pin id of eda infra world that returns node id in GATSPI world

  let num_of_gatspi_cells = these_opins2.len();

  let pinid2gatspiid_ptr = AtomicPtr::new(pinid2gatspiid.as_mut_ptr());
  (0..num_of_gatspi_cells).into_par_iter().for_each(|i| {
   unsafe {
    let ptr = pinid2gatspiid_ptr.load(Ordering::Relaxed);
    *ptr.add(these_opins2[i]) = i;
   }
  });

  let items_ptr = AtomicPtr::new(items.as_mut_ptr());
  let values_ptr = AtomicPtr::new(values.as_mut_ptr());
  (0..these_nets.len()).into_par_iter().for_each(|i| {
   let temp_items : Vec<usize> = db.net2pin.items[db.net2pin.start[these_nets[i]]..db.net2pin.start[these_nets[i]+1]].into_iter().filter(|&pin| (db.pindirect[*pin] == Direction::I) && (db.pin2cell[*pin] != 0) ).map(|&pin| db.pin2cell[pin]).
    map(|cell| Vec::from_iter((&db.cell2pin.items[db.cell2pin.start[cell]..db.cell2pin.start[cell+1]]).into_iter().filter(|&index| db.pindirect[*index] == Direction::O).map(|&index| index)) ).concat();
   let temp_edgetypes: Vec<u8>  = db.net2pin.items[db.net2pin.start[these_nets[i]]..db.net2pin.start[these_nets[i]+1]].into_iter().filter(|&pin| (db.pindirect[*pin] == Direction::I) && (db.pin2cell[*pin] != 0) ).
    map(|&pin| iter::repeat_n(stdlib_info.get_pintype(&db.celltypes[db.pin2cell[pin]], &db.pinnames[pin].1), db.cell2noutputs[db.pin2cell[pin]]).collect::<Vec<_>>() )
    .concat();
   unsafe {
    let items_ptr = items_ptr.load(Ordering::Relaxed);
    let values_ptr = values_ptr.load(Ordering::Relaxed);
    for j in 0..temp_items.len() {
     *items_ptr.add(these_start[i] + j) = pinid2gatspiid[temp_items[j]];
     *values_ptr.add(these_start[i] + j) = temp_edgetypes[j];
    }
   }
  });
  
 let mut these_celltypes: Vec<u16> = (0..max_threads).into_par_iter().flat_map(|thread_id| {
  let start = thread_id * stride;
  let end = ((thread_id + 1) * stride).min(these_opins2.len());
  (start..end).map(|i| stdlib_info.get_celltype( &db.celltypes[db.pin2cell[these_opins2[i]]], &db.pinnames[these_opins2[i]].1, &db.name )).collect::<Vec<u16>>()
  }).collect();
 

  
 // Parse SDF file if provided and populate SDFLUT
 let (SDFLUT, SDFPointerStart, SDFPointerEnd, interconnectDelays) = if let Some(sdf_file_path) = sdf_path {
  println!("Parsing SDF file: {:?}", sdf_file_path);
  
  // Enable timer for SDF processing
  clilog::enable_timer("SDF processing");
  let time_sdf_processing = clilog::stimer!("SDF processing");
  
  let SDFLUTSize: usize = these_celltypes.iter().filter(|&&celltype| celltype != 999).map(|&celltype| {let n = ipinsInCelltype[celltype as usize] as u32;
   n * (1 << (n + 1))}).sum::<u32>() as usize;
  println!("SDFLUTSize: {}", SDFLUTSize);
  let mut SDFLUT: Vec<u32> = vec![u32::MAX; SDFLUTSize]; //this houses all the SDF entries with a pre-calculated size   
  let SDFPointerEnd: Vec<usize> = items.iter().map(|&cell_id| these_celltypes[cell_id]).map(|celltype| ipinsInCelltype[celltype as usize] as usize)
   .map(|n| 1 << (n+1)).scan(0, |acc, x| {*acc += x; Some(*acc)}).collect(); //this means we can pre-calculate the pointers for each edge. 
  let mut SDFPointerStart: Vec<usize> = vec![0]; //the pointers point to where in the SDFLUT the delays reside for this arc. 
  SDFPointerStart.extend_from_slice(&SDFPointerEnd[..SDFPointerEnd.len()-1]); //the pointers are graph edge attributes
  assert!(SDFPointerEnd.last().unwrap() <= &SDFLUT.len());
  if SDFPointerEnd.last().unwrap() != &SDFLUT.len() {println!("Warning: The SDF parser reports some arcs are not annotated in the SDF file");}
  let sdf_data = sdfParse::readin_file(sdf_file_path); // process the SDF file. sdf_data is still mostyl string data, but organized and parsed.
  let mut interconnectDelays: Vec<u32> = vec![0; items.len()]; // default is 0 wire delays. each wire delay's rise/fall time is bit sliced into 1 u32 representation (u16/u16 for r/f delay time)
  
  // Enable timer for interconnect delays processing
  clilog::enable_timer("interconnect delays processing");
  let time_interconnect = clilog::stimer!("interconnect delays processing");
  
  // Process interconnect delays from SDF first using Rayon parallelization with direct updates
  let interconnect_delays_ptr = AtomicPtr::new(interconnectDelays.as_mut_ptr());
  sdf_data.parseInterconnectResults
    .par_iter()
    .for_each(|interconnect_result| {
      if let Some((driverString, loadString, delay)) = interconnect_result {
        // Convert driverString to GATSPI ID, Parse driverString to extract hierarchical name, pin name, and bus index
        if let Some(driver_pin_id) = Self::parsePinname2id(&db, driverString) {
          let driver_gatspi_id = pinid2gatspiid[driver_pin_id];  // Next, Process loadString
          let last_slash_pos = loadString.rfind('/');
          let (instance_name, thisInputPin) = match last_slash_pos {
            Some(pos) => { let instance = &loadString[..pos]; let pin = &loadString[pos + 1..]; (instance, pin) }
            None => ("", loadString.as_str())
          };
          let instance_hier_name = if instance_name.is_empty() {netlistdb::HierName::empty()} else {netlistdb::HierName::from_topdown_hier_iter(std::iter::once(instance_name))}; 
          // Translate instance name to eda-infra world cell id                
          if let Some(&cell_id) = db.cellname2id.get(&instance_hier_name) {
            // Find all output pin ids associated with the instance
            let output_pin_ids: Vec<usize> = db.cell2pin.items[db.cell2pin.start[cell_id]..db.cell2pin.start[cell_id+1]].iter()
              .filter(|&&pin_id| db.pindirect[pin_id] == Direction::O).copied().collect();
            let interconnectLoads: Vec<usize> = output_pin_ids.iter().map(|&pin_id| pinid2gatspiid[pin_id]).collect(); // Translate to gatspi ids
            let load_celltypes: Vec<u16> = interconnectLoads.iter().map(|&gatspi_id| these_celltypes[gatspi_id]).collect(); // Find celltypes of interconnectLoads by indexing these_celltypes
            let load_celltype_strings: Vec<&str> = load_celltypes.iter().filter_map(|&celltype| {stdlib_attributes::celltypeHash.iter()
              .find(|&(_, &value)| value == celltype).map(|(key, _)| *key)}).collect(); // Reverse search celltypeHash to get string celltypes
            let edge_ids: Vec<u8> = load_celltype_strings.iter().filter_map(|&celltype_str| {
              let pintype_key = format!("{}/{}", celltype_str, thisInputPin); stdlib_attributes::pintypeHash.get(pintype_key.as_str()).copied()
            }).collect(); // Concatenate celltype strings with thisInputPin and look up edge IDs
            // Find CSR indices for the interconnect delays
            let csr_indices = Self::findCSRIndices4Loads(
                driver_gatspi_id,
                &interconnectLoads,
                &edge_ids,
                &these_start,
                &items,
                &values
            );
            
            // Directly update interconnectDelays for each CSR index using atomic pointer
            unsafe {
              let ptr = interconnect_delays_ptr.load(Ordering::Relaxed);
              for (_, csr_index) in csr_indices {
                *ptr.add(csr_index) = *delay;
              }
            }
          }
        }
      }
    });
  
  // Finish interconnect delays processing timer
  clilog::finish!(time_interconnect);
  
  // Enable timer for cell delays processing
  clilog::enable_timer("cell delays processing");
  let time_cell_delays = clilog::stimer!("cell delays processing");
  
  // Process cell delays from SDF using Rayon parallelization with direct SDFLUT updates
  let sdflut_ptr = AtomicPtr::new(SDFLUT.as_mut_ptr());
  sdf_data.parseAllCellsResults
    .par_iter()
    .for_each(|cell_result| {
      if let Some((fullCelltype_str, instance_str, parsed_results)) = cell_result {
        // Process the fullCelltype_str using GL0AMStdLib's get_celltype
        let celltype_compact = compact_str::CompactString::new_inline(fullCelltype_str);
        
        // Get the output pin name from the first tuple in parsed_results
        let (_, _, output_pin, _, _) = parsed_results.first().unwrap();
        let pin_name_compact = compact_str::CompactString::new_inline(output_pin);
        
        let top_name_compact = compact_str::CompactString::new_inline(&db.name);
        
        let celltype_number = stdlib_info.get_celltype(&celltype_compact, &pin_name_compact, &top_name_compact);
        let numIPins = ipinsInCelltype[celltype_number as usize];
        //println!("DEBUG: fullCelltype_str = '{}', celltype_number = {}, numIPins = {}", fullCelltype_str, celltype_number, numIPins);

        // Process each tuple in parsed_results
        for parsed_tuple in parsed_results {
          let (CONDs, arcInputPin, arcOutputPin, arcDelay, sdf_line) = parsed_tuple;
          
          // Get dstID by converting instance_str to HierName and looking up pin
          let instance_hier_name = netlistdb::HierName::from_topdown_hier_iter(std::iter::once(instance_str));
          let arc_output_pin_compact = compact_str::CompactString::new_inline(arcOutputPin);
          let pin_key = (instance_hier_name, arc_output_pin_compact, None::<isize>); // No bus index for output pins
          
          if let Some(&eda_pin_id) = db.pinname2id.get(&pin_key) {
            let dstID = pinid2gatspiid[eda_pin_id];
            
            // Process arcInputPin to get ipinName
            let mut ipinName = arcInputPin.to_string();
            // Strip '(' and ')' characters
            ipinName = ipinName.replace("(", "").replace(")", "");
            
            // If ipinName contains spaces, take the part after the last space
            if ipinName.contains(' ') {
                if let Some(last_space_pos) = ipinName.rfind(' ') {
                    ipinName = ipinName[last_space_pos + 1..].to_string();
                }
            }
            
            // Determine baseSDFEntryOffset and baseSDFEntryStride based on edge type
            let (baseSDFEntryOffset, baseSDFEntryStride) = if arcInputPin.contains("posedge") {
                (0, 1 << numIPins)
            } else if arcInputPin.contains("negedge") {
                (1 << numIPins, 1 << numIPins)
            } else {
                (0, 1 << (numIPins + 1))
            };
            
            // Find srcID by following the path: instance â pin ID â net ID â driver pin â GATSPI ID
            let instance_hier_name = netlistdb::HierName::from_topdown_hier_iter(std::iter::once(instance_str));
            let ipin_name_compact = compact_str::CompactString::new_inline(&ipinName);
            let ipin_key = (instance_hier_name, ipin_name_compact.clone(), None::<isize>); // No bus index for input pins
            
            if let Some(&eda_pin_id) = db.pinname2id.get(&ipin_key) {
                // Find the connecting net's ID
                let net_id = db.pin2net[eda_pin_id];
                
                // Find the driver of the net (Direction::O pin)
                let driver_pin_id = db.net2pin.items[db.net2pin.start[net_id]..db.net2pin.start[net_id+1]]
                    .iter()
                    .find(|&&pin_id| db.pindirect[pin_id] == Direction::O)
                    .copied();
                
                if let Some(driver_pin) = driver_pin_id {
                    // Convert driver pin to GATSPI world
                    let srcID = pinid2gatspiid[driver_pin];
                    
                    // Find edge_id by reverse searching celltypeHash and looking up pintypeHash
                    let celltype_core_name = stdlib_info.core_name(&celltype_compact, &ipin_name_compact);
                    let pintype_key = format!("{}/{}", celltype_core_name, ipinName);
                    let edge_id_value = *stdlib_attributes::pintypeHash.get(pintype_key.as_str())
                        .expect(&format!("Could not find edge_id in pintypeHash for key='{}'", pintype_key));
                    
                    if let Some(conds) = CONDs {
                        let min_required = numIPins.saturating_sub(1) as usize;
                        if conds.len() < min_required {
                            panic!(
                                "cell: {} with SDF line: {} does not have enough conditions! exiting!...",
                                instance_str, sdf_line
                            );
                        }
                    }
                    
                    // Get SDFLUTBasePointer using findCSRIndices4Loads
                    let loads = vec![dstID];
                    let edge_ids = vec![edge_id_value];
                    let csr_indices = Self::findCSRIndices4Loads(
                        srcID,
                        &loads,
                        &edge_ids,
                        &these_start,
                        &items,
                        &values
                    );
                    
                    let (_, SDFPointerStartIndex) = csr_indices.first()
                        .expect(&format!("Could not find CSR index for srcID={}, dstID={}, edge_id={}", srcID, dstID, edge_id_value));
                    let SDFLUTBasePointer = SDFPointerStart[*SDFPointerStartIndex] + baseSDFEntryOffset;
                    
                    // Directly update SDFLUT using atomic pointer
                    unsafe {
                        let ptr = sdflut_ptr.load(Ordering::Relaxed);
                        
                                                 // If CONDs is None, assign arcDelay to SDFLUT within the index range
                         if CONDs.is_none() {
                             for i in SDFLUTBasePointer..SDFLUTBasePointer + baseSDFEntryStride {
                                 Self::assignArcDelayToSDFLUT_ptr(ptr, i, *arcDelay);
                             }
                         } else {
                             // Process CONDs to calculate specificArcOffset
                             let mut specificArcOffset = 0;
                             for cond_tuple in CONDs.as_ref().unwrap() {
                                 let (cond_pin, cond_value) = cond_tuple;
                                 if *cond_value == 1 {
                                    let cond_pintype_key = format!("{}/{}", celltype_core_name, cond_pin);
                                     let n = *stdlib_attributes::pintypeHash.get(cond_pintype_key.as_str())
                                         .expect(&format!("Could not find cond_pintype_key in pintypeHash: '{}'", cond_pintype_key));
                                     specificArcOffset += 1 << n;
                                 }
                             }
                             
                            // Assign arcDelay to specific indices in SDFLUT
                            let should_overwrite = stdlib_info.cond_sdf_overwrite();
                            let indices = [
                                SDFLUTBasePointer + specificArcOffset,
                                SDFLUTBasePointer + specificArcOffset + (1 << edge_id_value),
                            ];
                            for index in indices {
                                if should_overwrite {
                                    *ptr.add(index) = *arcDelay;
                                } else {
                                    Self::assignArcDelayToSDFLUT_ptr(ptr, index, *arcDelay);
                                }
                            }
                             
                             // If baseSDFEntryStride equals 1 << (numIPins + 1), assign to additional indices
                             if baseSDFEntryStride == 1 << (numIPins + 1) {
                                let extra_indices = [
                                    SDFLUTBasePointer + specificArcOffset + (1 << numIPins),
                                    SDFLUTBasePointer + specificArcOffset + (1 << edge_id_value) + (1 << numIPins),
                                ];
                                for index in extra_indices {
                                    if should_overwrite {
                                        *ptr.add(index) = *arcDelay;
                                    } else {
                                        Self::assignArcDelayToSDFLUT_ptr(ptr, index, *arcDelay);
                                    }
                                }
                             }
                         }
                    }
                }
            } else {
                println!("WARNING: Could not find pin in db.pinname2id for instance_str='{}', ipinName='{}'", 
                         instance_str, ipinName);
            }
          }
        }
      }
    });
  
  // Finish cell delays processing timer
  clilog::finish!(time_cell_delays);
  
  // Finish overall SDF processing timer
  clilog::finish!(time_sdf_processing);
  
  (Some(SDFLUT), Some(SDFPointerStart), Some(SDFPointerEnd), Some(interconnectDelays))
 } else {
     println!("No SDF file provided. SDF data structures will be None.");
     (None, None, None, None)
 };
 /*let missing_numbers: Vec<usize> = (0..these_celltypes.len()).filter(|&i| !items.contains(&i)).collect();
 println!("Missing numbers in items: {:?}", missing_numbers);
 let missing_celltypes: Vec<u16> = missing_numbers.iter().map(|&i| these_celltypes[i]).collect();
 println!("Celltypes of missing numbers: {:?}", missing_celltypes);*/
 

 let mut translation_dict: Vec<(String,String)> = (0..max_threads).into_par_iter().flat_map(|thread_id| {
  let start = legit_top_ports.len() + thread_id * stride;
  let end = if thread_id == max_threads - 1 { these_celltypes.len() } else { (legit_top_ports.len() + (thread_id + 1) * stride).min(these_celltypes.len()) };
  (start..end).map(|id| ( 
   if let None = db.pinnames[these_opins2[id]].2 { 
       format!("{:?}/{}", db.pinnames[these_opins2[id]].0, db.pinnames[these_opins2[id]].1).replace("HierName()/","").replace("HierName(","").replace(")/", "/") 
   } else { 
       format!("{:?}/{}[{}]", db.pinnames[these_opins2[id]].0, db.pinnames[these_opins2[id]].1, db.pinnames[these_opins2[id]].2.unwrap()).replace("HierName()/","").replace("HierName(","").replace(")/", "/") 
   }, 
   if let None = db.netnames[db.pin2net[these_opins2[id]]].2 { format!("{:?}/{}", db.netnames[db.pin2net[these_opins2[id]]].0, db.netnames[db.pin2net[these_opins2[id]]].1).replace("HierName()/","").replace("HierName(","").replace(")/", "/")  } 
   else { format!("{:?}/{}[{}]", db.netnames[db.pin2net[these_opins2[id]]].0, db.netnames[db.pin2net[these_opins2[id]]].1, db.netnames[db.pin2net[these_opins2[id]]].2.unwrap()).replace("HierName()/","").replace("HierName(","").replace(")/", "/") }
  ) ).collect::<Vec<_>>()
 }).collect();



   // Append 999 for VDD and GND nets if they exist
  if db.net_zero.is_some() {
     these_celltypes.push(999);
     translation_dict.push(("1'b0".to_string(), "1'b0".to_string()));
 }
 if db.net_one.is_some() {
     these_celltypes.push(999);
     translation_dict.push(("1'b1".to_string(), "1'b1".to_string()));
 }



 let mut port_dict = HashMap::new();
 for i in legit_top_ports.clone() {
  let hashkey = 
   if db.pindirect[i] == Direction::O { pinid2gatspiid[i] } else {
    db.net2pin.items[db.net2pin.start[db.pin2net[i]]..db.net2pin.start[db.pin2net[i]+1]].into_iter().filter(|&pin| db.pindirect[*pin] == Direction::O)
     .map(|&pin| pinid2gatspiid[pin]).collect::<Vec<_>>()[0]
   }; 
  let hashvalue = 
   if db.pinnames[i].2 == None { String::from(db.pinnames[i].1.clone()) } else {String::from(format!("{}[{}]", db.pinnames[i].1, db.pinnames[i].2.unwrap())) };
  port_dict.insert( hashkey , hashvalue );
 }
 
 // Check that the length of start is 1 more than the length of these_celltypes
 assert_eq!(these_start.len(), these_celltypes.len() + 1, 
     "Length of start ({}) should be 1 more than length of these_celltypes ({})", 
     these_start.len(), these_celltypes.len());
 
 //println!("Gatspi ID 618 name: {:?}", translation_dict.get(75 - legit_top_ports.len()));
  Self { 
   gatspi_cells: these_opins2.clone(), gatspi_celltypes: these_celltypes, start: these_start, items: Vec::from(items), values: Vec::from(values), pinid2gatspiid: Vec::from(pinid2gatspiid), 
   num_of_gatspi_cells: these_opins2.len(), num_of_top_ports: legit_top_ports.len(), gatspi_cellname_index: translation_dict, gatspi_port_index: port_dict, SDFLUT: SDFLUT,
   SDFPointerStart: SDFPointerStart, SDFPointerEnd: SDFPointerEnd, interconnectDelays: interconnectDelays,
  }
 }
}

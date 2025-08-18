use compact_str::CompactString;
use regex::Regex;
use lazy_static::lazy_static;
use std::collections::HashMap;
use netlistdb::*;
use sverilogparse::*;


//Will need to edit this file to suit your std cell lib definition
lazy_static! {
    pub static ref SEQ_REGEX: Regex = Regex::new(r".*(_DF|_LATCH_|_CLKGATE_|_RAMS_|_SYNC).*").unwrap();
}

/// standard cell celltype numerical attributes, and cell pin type numerical attributes
///
/// The GATSPI simulator uses this. Celltype number corresponds to 
/// row in logic truth table that houses truth tables for all cells
/// used in constructing NetlistDB.
///
/// pin type number tells simulator which pin connection the simulation graph
/// edge connection is representing.
///

pub struct GL0AMGenericVlibStdCellPinDefs();
impl LeafPinProvider for GL0AMGenericVlibStdCellPinDefs {
 fn direction_of(
  &self,
  macro_name: &CompactString,
  pin_name: &CompactString, pin_idx: Option<isize>
  ) -> Direction {
  if let true = SEQ_REGEX.is_match(macro_name.as_str()) {
   match pin_name.as_str() {
    "Q" | "QN" | "dout" | "DST_Q" | "SRC_D" | "q" => Direction::O,
    "D" | "SI" | "SE" | "CP" | "CDN" | "SDN" | "E" | "TE" | "clk" | "ra" | "re" | "wa" | "we" | "di" | "ore" | "d" | "clr_" | "set_" | 
    "SRC_D_NEXT" | "SRC_CLK" | "SRC_CLRN" | "DST_CLK" | "DST_CLRN" | "ATPG_CTL" | "TEST_MODE" | "byp_sel" | "dbyp" => Direction::Unknown,
    _ => { 
     use netlistdb::{GeneralPinName, HierName};
     panic!("Cannot recognize sequential pin type {}, please make sure the verilog netlist is synthesized from GENERIC tech lib.",
     (HierName::single(macro_name.clone()),
     pin_name, pin_idx).dbg_fmt_pin());
    }
   }
  } else {
   if *pin_name == "S" {
    let macro_str = macro_name.as_str();
    if macro_str.contains("MUX") { return Direction::I; } else if macro_str.contains("HA") || macro_str.contains("FA") { return Direction::O;
    } else {
     use netlistdb::{GeneralPinName, HierName};
     panic!("Cannot recognize pin S usage in module {}, please make sure the verilog netlist is synthesized from GENERIC tech lib.",
      (HierName::single(macro_name.clone()), pin_name, pin_idx).dbg_fmt_pin());
    }
   }
   match pin_name.as_str() {
    "Z" | "ZN" | "CO" => Direction::O,
    "A1" | "A2" | "A3" | "B1" | "B2" | "B3" | "C1" | "C2" | "C3" | "A" | "B" | "C" | "CI" | "A4" | "B4" | 
    "D1" | "D2" | "D3" | "D4" | 
    "I0" | "I1" | "I2" | "I3" | "S0" | "S1" | "I" => Direction::I,
    _ => { 
     use netlistdb::{GeneralPinName, HierName};
     panic!("Cannot recognize unknown pin type {}, please make sure the verilog netlist is synthesized from GENERIC tech lib.",
     (HierName::single(macro_name.clone()),
     pin_name, pin_idx).dbg_fmt_pin());
    }
   }
  }
 } 
    
 fn width_of(
  &self,
  macro_name: &CompactString,
  pin_name: &CompactString
 ) -> Option<SVerilogRange> {
  // Check if macro_name contains "_RAMS_" and pin_name is "dout"
  if macro_name.as_str().contains("_RAMS_") && pin_name.as_str() == "dout" {
    let macro_str = macro_name.as_str();
    
    // Find the last 'x' or 'X' in macro_name (case insensitive)
    let last_x_pos = macro_str.rfind('x').or(macro_str.rfind('X'));
    if let Some(last_x_pos) = last_x_pos {
      // Extract the substring after the last 'x' or 'X'
      let after_x = &macro_str[last_x_pos + 1..];
      
      // Try to parse the integer after the last 'x' or 'X'
      if let Ok(dout_width) = after_x.parse::<isize>() {
        return Some(SVerilogRange(dout_width - 1, 0));
      }
    }
  }
  
  None
 }
}

/* pub struct StdCellPinDefs();
impl LeafPinProvider for StdCellPinDefs {
 fn direction_of(
  &self,
  macro_name: &CompactString,
  pin_name: &CompactString, pin_idx: Option<isize>
  ) -> Direction {
  if let true = SEQ_REGEX.is_match(macro_name.as_str()) {
   match pin_name.as_str() {
    "GCLK" | "Q" | "QN" | "rd_out" => Direction::O,
    "D" | "SE" | "SI" | "CLK" | "RESETN" | "RESET" | "SETN" | "SET" | "ENA" | "addr_in" | "ce_in" | "clk" | "wd_in" | "we_in" => Direction::Unknown,
    _ => { 
     use netlistdb::{GeneralPinName, HierName};
     panic!("Cannot recognize sequential pin type {}, please make sure the verilog netlist is synthesized from Contest tech lib.",
     (HierName::single(macro_name.clone()),
     pin_name, pin_idx).dbg_fmt_pin());
    }
   }
  } else {
   match pin_name.as_str() {
    "Y" | "CON" | "SN"  => Direction::O,
    "A1" | "A2" | "A3" | "B1" | "B2" | "B3" | "C1" | "C2" | "C3" | "A" | "B" | "C" | "D" | "E" | "CI" => Direction::I,
    _ => { 
     use netlistdb::{GeneralPinName, HierName};
     panic!("Cannot recognize unknown pin type {}, please make sure the verilog netlist is synthesized from Contest tech lib.",
     (HierName::single(macro_name.clone()),
     pin_name, pin_idx).dbg_fmt_pin());
    }
   }
  }
 } 
    
 fn width_of(
  &self,
  _macro_name: &CompactString,
  _pin_name: &CompactString
 ) -> Option<SVerilogRange> {
  None
 }
} */


pub trait StandardCellTypeAttribute {
    /// This function is called from GATSPIGraph constructor to
    /// provide standard cell celltype attribute to the GATSPI
    /// simulation graph
    fn get_celltype(
        &self,
        macro_name: &CompactString, pin_name: &CompactString, top_name: &CompactString,
    ) -> u16;

    // pin type attribute
    fn get_pintype(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString
    ) -> u8;
}

lazy_static! {
    pub static ref celltypeHash: HashMap<&'static str, u16> =  HashMap::from([
("A2O1A1I",0),
("A2O1A1O1I",1),
("AO211",  2),
("AO21",   3),
("AO221",  4),
("AO222",  5),
("AO22",   6),
("AO31",   7),
("AO322",  8),
("AO32",   9),
("AO331",  10),
("AO332",  11),
("AO333",  12),
("AO33",   13),
("AOI211", 14),
("AOI21",  15),
("AOI221", 16),
("AOI222", 17),
("AOI22",  18),
("AOI311", 19),
("AOI31",  20),
("AOI321", 21),
("AOI322", 22),
("AOI32",  23),
("AOI331", 24),
("AOI332", 25),
("AOI333", 26),
("AOI33",  27),
("BUF",    28),
("CKINVDC",29),
("HB1",    30),
("HB2",    31),
("HB3",    32),
("HB4",    33),
("INV",    34),
("O2A1O1I",35),
("OA211",  36),
("OA21",   37),
("OA221",  38),
("OA222",  39),
("OA22",   40),
("OA31",   41),
("OA32",   42),
("OA331",  43),
("OA332",  44),
("OA333",  45),
("OA33",   46),
("OAI211", 47),
("OAI21",  48),
("OAI221", 49),
("OAI222", 50),
("OAI22",  51),
("OAI311", 52),
("OAI31",  53),
("OAI321", 54),
("OAI322", 55),
("OAI32",  56),
("OAI331", 57),
("OAI332", 58),
("OAI333", 59),
("OAI33",  60),
("AND2",   61),
("AND3",   62),
("AND4",   63),
("AND5",   64),
("FASN",     65),
("FACON",     66),
("HASN",     67),
("HACON",     68),
("MAJI",   69),
("MAJ",    70),
("NAND2",  71),
("NAND3",  72),
("NAND4",  73),
("NAND5",  74),
("NOR2",   75),
("NOR3",   76),
("NOR4",   77),
("NOR5",   78),
("OR2",    79),
("OR3",    80),
("OR4",    81),
("OR5",    82),
("XNOR2",  83),
("XOR2",   84),
("XNOR3",  85),
("XOR3",   86),
("AO2222", 87),
("AOAI211",88),
("OAOI211",89),
("IAND2",  90),
("IAOI21", 91),
("IAOI22", 92),
("IBAO21", 93),
("IBOA21", 94),
("INOR2",  95),
("INAND2", 96),
("INOR3",  97),
("INOR4",  98),
("IIOAI21",99),
("IIOAI22",100),
("IBOAI21",101),
("IOR2",   102),
("FAS",    103),
("FACO",   104),
("HAS",    105),
("HACO",   106),
("MAJORITYAOI222",107),
("MAJORITYAOI22",108),
("MAJORITYOAI22",109),
("MUX2",   110),
("MUX2N",  111),
("MUX3",   112),
("MUX3N",  113),
("MUX4",   114),
("MUX4N",  115),
]);
}

lazy_static! {
 pub static ref pintypeHash: HashMap<&'static str, u8> =  HashMap::from([
  ("A2O1A1I/A1", 3),
  ("A2O1A1I/A2", 2),
  ("A2O1A1I/B", 1),
  ("A2O1A1I/C", 0),
  ("A2O1A1O1I/A1", 4),
  ("A2O1A1O1I/A2", 3),
  ("A2O1A1O1I/B", 2),
  ("A2O1A1O1I/C", 1),
  ("A2O1A1O1I/D", 0),
  ("AO211/A1", 3),
  ("AO211/A2", 2),
  ("AO211/B", 1),
  ("AO211/C", 0),
  ("AO21/A1", 2),
  ("AO21/A2", 1),
  ("AO21/B", 0),
  ("AO221/A1", 4),
  ("AO221/A2", 3),
  ("AO221/B1", 2),
  ("AO221/B2", 1),
  ("AO221/C", 0),
  ("AO222/A1", 5),
  ("AO222/A2", 4),
  ("AO222/B1", 3),
  ("AO222/B2", 2),
  ("AO222/C1", 1),
  ("AO222/C2", 0),
  ("AO22/A1", 3),
  ("AO22/A2", 2),
  ("AO22/B1", 1),
  ("AO22/B2", 0),
  ("AO31/A1", 3),
  ("AO31/A2", 2),
  ("AO31/A3", 1),
  ("AO31/B", 0),
  ("AO322/A1", 6),
  ("AO322/A2", 5),
  ("AO322/A3", 4),
  ("AO322/B1", 3),
  ("AO322/B2", 2),
  ("AO322/C1", 1),
  ("AO322/C2", 0),
  ("AO32/A1", 4),
  ("AO32/A2", 3),
  ("AO32/A3", 2),
  ("AO32/B1", 1),
  ("AO32/B2", 0),
  ("AO331/A1", 6),
  ("AO331/A2", 5),
  ("AO331/A3", 4),
  ("AO331/B1", 3),
  ("AO331/B2", 2),
  ("AO331/B3", 1),
  ("AO331/C", 0),
  ("AO332/A1", 7),
  ("AO332/A2", 6),
  ("AO332/A3", 5),
  ("AO332/B1", 4),
  ("AO332/B2", 3),
  ("AO332/B3", 2),
  ("AO332/C1", 1),
  ("AO332/C2", 0),
  ("AO333/A1", 8),
  ("AO333/A2", 7),
  ("AO333/A3", 6),
  ("AO333/B1", 5),
  ("AO333/B2", 4),
  ("AO333/B3", 3),
  ("AO333/C1", 2),
  ("AO333/C2", 1),
  ("AO333/C3", 0),
  ("AO33/A1", 5),
  ("AO33/A2", 4),
  ("AO33/A3", 3),
  ("AO33/B1", 2),
  ("AO33/B2", 1),
  ("AO33/B3", 0),
  ("AOI211/A1", 3),
  ("AOI211/A2", 2),
  ("AOI211/B", 1),
  ("AOI211/C", 0),
  ("AOI21/A1", 2),
  ("AOI21/A2", 1),
  ("AOI21/B", 0),
  ("AOI221/A1", 4),
  ("AOI221/A2", 3),
  ("AOI221/B1", 2),
  ("AOI221/B2", 1),
  ("AOI221/C", 0),
  ("AOI222/A1", 5),
  ("AOI222/A2", 4),
  ("AOI222/B1", 3),
  ("AOI222/B2", 2),
  ("AOI222/C1", 1),
  ("AOI222/C2", 0),
  ("AOI22/A1", 3),
  ("AOI22/A2", 2),
  ("AOI22/B1", 1),
  ("AOI22/B2", 0),
  ("AOI311/A1", 4),
  ("AOI311/A2", 3),
  ("AOI311/A3", 2),
  ("AOI311/B", 1),
  ("AOI311/C", 0),
  ("AOI31/A1", 3),
  ("AOI31/A2", 2),
  ("AOI31/A3", 1),
  ("AOI31/B", 0), 
  ("AOI321/A1", 5),
  ("AOI321/A2", 4),
  ("AOI321/A3", 3),
  ("AOI321/B1", 2),
  ("AOI321/B2", 1),
  ("AOI321/C", 0),
  ("AOI322/A1", 6),
  ("AOI322/A2", 5),
  ("AOI322/A3", 4),
  ("AOI322/B1", 3),
  ("AOI322/B2", 2),
  ("AOI322/C1", 1),
  ("AOI322/C2", 0),
  ("AOI32/A1", 4),
  ("AOI32/A2", 3),
  ("AOI32/A3", 2),
  ("AOI32/B1", 1),
  ("AOI32/B2", 0),
  ("AOI331/A1", 6),
  ("AOI331/A2", 5),
  ("AOI331/A3", 4),
  ("AOI331/B1", 3),
  ("AOI331/B2", 2),
  ("AOI331/B3", 1),
  ("AOI331/C1", 0),
  ("AOI332/A1", 7),
  ("AOI332/A2", 6),
  ("AOI332/A3", 5),
  ("AOI332/B1", 4),
  ("AOI332/B2", 3),
  ("AOI332/B3", 2),
  ("AOI332/C1", 1),
  ("AOI332/C2", 0),
  ("AOI333/A1", 8),
  ("AOI333/A2", 7),
  ("AOI333/A3", 6),
  ("AOI333/B1", 5),
  ("AOI333/B2", 4),
  ("AOI333/B3", 3),
  ("AOI333/C1", 2),
  ("AOI333/C2", 1),
  ("AOI333/C3", 0),
  ("AOI33/A1", 5),
  ("AOI33/A2", 4),
  ("AOI33/A3", 3),
  ("AOI33/B1", 2),
  ("AOI33/B2", 1),
  ("AOI33/B3", 0),
  ("BUF/I", 0),
  ("CKINVDC/A", 0),
  ("HB1/A", 0),
  ("HB2/A", 0),
  ("HB3/A", 0),
  ("HB4/A", 0),
  ("INV/I", 0),
  ("O2A1O1I/A1", 3),
  ("O2A1O1I/A2", 2),
  ("O2A1O1I/B", 1),
  ("O2A1O1I/C", 0),
  ("OA211/A1", 3),
  ("OA211/A2", 2),
  ("OA211/B", 1),
  ("OA211/C", 0),
  ("OA21/A1", 2),
  ("OA21/A2", 1),
  ("OA21/B", 0),
  ("OA221/A1", 4),
  ("OA221/A2", 3),
  ("OA221/B1", 2),
  ("OA221/B2", 1),
  ("OA221/C", 0),
  ("OA222/A1", 5),
  ("OA222/A2", 4),
  ("OA222/B1", 3),
  ("OA222/B2", 2),
  ("OA222/C1", 1),
  ("OA222/C2", 0),
  ("OA22/A1", 3),
  ("OA22/A2", 2),
  ("OA22/B1", 1),
  ("OA22/B2", 0),
  ("OA31/A1", 3),
  ("OA31/A2", 2),
  ("OA31/A3", 1),
  ("OA31/B1", 0),
  ("OA31/B", 0),
  ("OA32/A1", 4),
  ("OA32/A2", 3),
  ("OA32/A3", 2),
  ("OA32/B1", 1),
  ("OA32/B2", 0),
  ("OA331/A1", 6),
  ("OA331/A2", 5),
  ("OA331/A3", 4),
  ("OA331/B1", 3),
  ("OA331/B2", 2),
  ("OA331/B3", 1),
  ("OA331/C1", 0),
  ("OA332/A1", 7),
  ("OA332/A2", 6),
  ("OA332/A3", 5),
  ("OA332/B1", 4),
  ("OA332/B2", 3),
  ("OA332/B3", 2),
  ("OA332/C1", 1),
  ("OA332/C2", 0),
  ("OA333/A1", 8),
  ("OA333/A2", 7),
  ("OA333/A3", 6),
  ("OA333/B1", 5),
  ("OA333/B2", 4),
  ("OA333/B3", 3),
  ("OA333/C1", 2),
  ("OA333/C2", 1),
  ("OA333/C3", 0),
  ("OA33/A1", 5),
  ("OA33/A2", 4),
  ("OA33/A3", 3),
  ("OA33/B1", 2),
  ("OA33/B2", 1),
  ("OA33/B3", 0),
  ("OAI211/A1", 3),
  ("OAI211/A2", 2),
  ("OAI211/B", 1),
  ("OAI211/C", 0),
  ("OAI21/A1", 2),
  ("OAI21/A2", 1),
  ("OAI21/B", 0),
  ("OAI221/A1", 4),
  ("OAI221/A2", 3),
  ("OAI221/B1", 2),
  ("OAI221/B2", 1),
  ("OAI221/C", 0),
  ("OAI222/A1", 5),
  ("OAI222/A2", 4),
  ("OAI222/B1", 3),
  ("OAI222/B2", 2),
  ("OAI222/C1", 1),
  ("OAI222/C2", 0),
  ("OAI22/A1", 3),
  ("OAI22/A2", 2),
  ("OAI22/B1", 1),
  ("OAI22/B2", 0),
  ("OAI311/A1", 4),
  ("OAI311/A2", 3),
  ("OAI311/A3", 2),
  ("OAI311/B1", 1),
  ("OAI311/C1", 0),
  ("OAI31/A1", 3),
  ("OAI31/A2", 2),
  ("OAI31/A3", 1),
  ("OAI31/B", 0),
  ("OAI321/A1", 5),
  ("OAI321/A2", 4),
  ("OAI321/A3", 3),
  ("OAI321/B1", 2),
  ("OAI321/B2", 1),
  ("OAI321/C", 0),
  ("OAI322/A1", 6),
  ("OAI322/A2", 5),
  ("OAI322/A3", 4),
  ("OAI322/B1", 3),
  ("OAI322/B2", 2),
  ("OAI322/C1", 1),
  ("OAI322/C2", 0),
  ("OAI32/A1", 4),
  ("OAI32/A2", 3),
  ("OAI32/A3", 2),
  ("OAI32/B1", 1),
  ("OAI32/B2", 0),
  ("OAI331/A1", 6),
  ("OAI331/A2", 5),
  ("OAI331/A3", 4),
  ("OAI331/B1", 3),
  ("OAI331/B2", 2),
  ("OAI331/B3", 1),
  ("OAI331/C1", 0),
  ("OAI332/A1", 7),
  ("OAI332/A2", 6),
  ("OAI332/A3", 5),
  ("OAI332/B1", 4),
  ("OAI332/B2", 3),
  ("OAI332/B3", 2),
  ("OAI332/C1", 1),
  ("OAI332/C2", 0),
  ("OAI333/A1", 8),
  ("OAI333/A2", 7),
  ("OAI333/A3", 6),
  ("OAI333/B1", 5),
  ("OAI333/B2", 4),
  ("OAI333/B3", 3),
  ("OAI333/C1", 2),
  ("OAI333/C2", 1),
  ("OAI333/C3", 0),
  ("OAI33/A1", 5),
  ("OAI33/A2", 4),
  ("OAI33/A3", 3),
  ("OAI33/B1", 2),
  ("OAI33/B2", 1),
  ("OAI33/B3", 0),
  ("AND2/A", 1),
  ("AND2/B", 0),
  ("AND2/A1", 1),
  ("AND2/A2", 0),
  ("AND3/A", 2),
  ("AND3/B", 1),
  ("AND3/C", 0),
  ("AND3/A1", 2),
  ("AND3/A2", 1),
  ("AND3/A3", 0),
  ("AND4/A", 3),
  ("AND4/B", 2),
  ("AND4/C", 1),
  ("AND4/D", 0),
  ("AND4/A1", 3),
  ("AND4/A2", 2),
  ("AND4/A3", 1),
  ("AND4/A4", 0),
  ("AND5/A", 4),
  ("AND5/B", 3),
  ("AND5/C", 2),
  ("AND5/D", 1),
  ("AND5/E", 0),
  ("AND5/A1", 4),
  ("AND5/A2", 3),
  ("AND5/A3", 2),
  ("AND5/A4", 1),
  ("AND5/A5", 0),
  ("FA/A", 2),
  ("FA/B", 1),
  ("FA/CI", 0),
  ("HA/A", 1),
  ("HA/B", 0),
  ("MAJI/A", 2),
  ("MAJI/B", 1),
  ("MAJI/C", 0),
  ("MAJ/A", 2),
  ("MAJ/B", 1),
  ("MAJ/C", 0),
  ("NAND2/A", 1),
  ("NAND2/B", 0),
  ("NAND2/A1", 1),
  ("NAND2/A2", 0),
  ("NAND3/A", 2),
  ("NAND3/B", 1),
  ("NAND3/C", 0),
  ("NAND3/A1", 2),
  ("NAND3/A2", 1),
  ("NAND3/A3", 0),
  ("NAND4/A", 3),
  ("NAND4/B", 2),
  ("NAND4/C", 1),
  ("NAND4/D", 0),
  ("NAND4/A1", 3),
  ("NAND4/A2", 2),
  ("NAND4/A3", 1),
  ("NAND4/A4", 0),
  ("NAND5/A", 4),
  ("NAND5/B", 3),
  ("NAND5/C", 2),
  ("NAND5/D", 1),
  ("NAND5/E", 0),
  ("NAND5/A1", 4),
  ("NAND5/A2", 3),
  ("NAND5/A3", 2),
  ("NAND5/A4", 1),
  ("NAND5/A5", 0),
  ("NOR2/A", 1),
  ("NOR2/B", 0),
  ("NOR2/A1", 1),
  ("NOR2/A2", 0),
  ("NOR3/A", 2),
  ("NOR3/B", 1),
  ("NOR3/C", 0),
  ("NOR3/A1", 2),
  ("NOR3/A2", 1),
  ("NOR3/A3", 0),
  ("NOR4/A", 3),
  ("NOR4/B", 2),
  ("NOR4/C", 1),
  ("NOR4/D", 0),
  ("NOR4/A1", 3),
  ("NOR4/A2", 2),
  ("NOR4/A3", 1),
  ("NOR4/A4", 0),
  ("NOR5/A", 4),
  ("NOR5/B", 3),
  ("NOR5/C", 2),
  ("NOR5/D", 1),
  ("NOR5/E", 0),
  ("NOR5/A1", 4),
  ("NOR5/A2", 3),
  ("NOR5/A3", 2),
  ("NOR5/A4", 1),
  ("NOR5/A5", 0),
  ("OR2/A", 1),
  ("OR2/B", 0),
  ("OR2/A1", 1),
  ("OR2/A2", 0),
  ("OR3/A", 2),
  ("OR3/B", 1),
  ("OR3/C", 0),
  ("OR3/A1", 2),
  ("OR3/A2", 1),
  ("OR3/A3", 0),
  ("OR4/A", 3),
  ("OR4/B", 2),
  ("OR4/C", 1),
  ("OR4/D", 0),
  ("OR4/A1", 3),
  ("OR4/A2", 2),
  ("OR4/A3", 1),
  ("OR4/A4", 0),
  ("OR5/A", 4),
  ("OR5/B", 3),
  ("OR5/C", 2),
  ("OR5/D", 1),
  ("OR5/E", 0),
  ("OR5/A1", 4),
  ("OR5/A2", 3),
  ("OR5/A3", 2),
  ("OR5/A4", 1),
  ("OR5/A5", 0),
  ("XNOR2/A1", 1),
  ("XNOR2/A2", 0),
  ("XNOR3/A1", 2),
  ("XNOR3/A2", 1),
  ("XNOR3/A3", 0),
  ("XOR2/A1", 1),
  ("XOR2/A2", 0),
  ("XOR3/A1", 2),
  ("XOR3/A2", 1),
  ("XOR3/A3", 0),
  ("AO2222/A1", 7),
  ("AO2222/A2", 6),
  ("AO2222/B1", 5),
  ("AO2222/B2", 4),
  ("AO2222/C1", 3),
  ("AO2222/C2", 2),
  ("AO2222/D1", 1),
  ("AO2222/D2", 0),
  ("AOAI211/A1", 3),
  ("AOAI211/A2", 2),
  ("AOAI211/B", 1),
  ("AOAI211/C", 0),
  ("IAOI21/A1", 2),
  ("IAOI21/A2", 1),
  ("IAOI21/B", 0),
  ("IBAO21/A1", 2),
  ("IBAO21/A2", 1),
  ("IBAO21/B", 0),
  ("IBOA21/A1", 2),
  ("IBOA21/A2", 1),
  ("IBOA21/B", 0),
  ("INAND2/A1", 1),
  ("INAND2/B1", 0),
  ("INOR2/A1", 1),
  ("INOR2/B1", 0),
  ("INOR3/A1", 2),
  ("INOR3/B1", 1),
  ("INOR3/B2", 0),
  ("INOR4/A1", 3),
  ("INOR4/B1", 2),
  ("INOR4/B2", 1),
  ("INOR4/B3", 0),
  ("IIOAI21/A1", 2),
  ("IIOAI21/A2", 1),
  ("IIOAI21/B", 0),
  ("IIOAI22/A1", 3),
  ("IIOAI22/A2", 2),
  ("IIOAI22/B1", 1),
  ("IIOAI22/B2", 0),
  ("IBOAI21/A1", 2),
  ("IBOAI21/A2", 1),
  ("IBOAI21/B", 0),
  ("IOR2/A1", 1),
  ("IOR2/B1", 0),
  ("IAND2/A1", 1),
  ("IAND2/B1", 0),
  ("IAOI22/A1", 3),
  ("IAOI22/A2", 2),
  ("IAOI22/B1", 1),
  ("IAOI22/B2", 0),
  ("MAJORITYAOI222/A", 2),
  ("MAJORITYAOI222/B", 1),
  ("MAJORITYAOI222/C", 0),
  ("MAJORITYAOI22/A1", 3),
  ("MAJORITYAOI22/A2", 2),
  ("MAJORITYAOI22/B1", 1),
  ("MAJORITYAOI22/B2", 0),
  ("MAJORITYOAI22/A1", 3),
  ("MAJORITYOAI22/A2", 2),
  ("MAJORITYOAI22/B1", 1),
  ("MAJORITYOAI22/B2", 0),
  ("MUX2/I0", 2),
  ("MUX2/I1", 1),
  ("MUX2/S", 0),
  ("MUX2N/I0", 2),
  ("MUX2N/I1", 1),
  ("MUX2N/S", 0),
  ("MUX3/I0", 4),
  ("MUX3/I1", 3),
  ("MUX3/I2", 2),
  ("MUX3/S0", 1),
  ("MUX3/S1", 0),
  ("MUX3N/I0", 4),
  ("MUX3N/I1", 3),
  ("MUX3N/I2", 2),
  ("MUX3N/S0", 1),
  ("MUX3N/S1", 0),
  ("MUX4/I0", 5),
  ("MUX4/I1", 4),
  ("MUX4/I2", 3),
  ("MUX4/I3", 2),
  ("MUX4/S0", 1),
  ("MUX4/S1", 0),
  ("MUX4N/I0", 5),
  ("MUX4N/I1", 4),
  ("MUX4N/I2", 3),
  ("MUX4N/I3", 2),
  ("MUX4N/S0", 1),
  ("MUX4N/S1", 0),
  ("OAOI211/A1", 3),
  ("OAOI211/A2", 2),
  ("OAOI211/B", 1),
  ("OAOI211/C", 0),
 ]);
}

lazy_static! {
 pub static ref ipinsInCelltype: Vec<u8> = vec![
  4,  // A2O1A1I: A1, A2, B, C
  5,  // A2O1A1O1I: A1, A2, B, C, D
  4,  // AO211: A1, A2, B, C
  3,  // AO21: A1, A2, B
  5,  // AO221: A1, A2, B1, B2, C
  6,  // AO222: A1, A2, B1, B2, C1, C2
  4,  // AO22: A1, A2, B1, B2
  4,  // AO31: A1, A2, A3, B
  7,  // AO322: A1, A2, A3, B1, B2, C1, C2
  5,  // AO32: A1, A2, A3, B1, B2
  7,  // AO331: A1, A2, A3, B1, B2, B3, C
  8,  // AO332: A1, A2, A3, B1, B2, B3, C1, C2
  9,  // AO333: A1, A2, A3, B1, B2, B3, C1, C2, C3
  6,  // AO33: A1, A2, A3, B1, B2, B3
  4,  // AOI211: A1, A2, B, C
  3,  // AOI21: A1, A2, B
  5,  // AOI221: A1, A2, B1, B2, C
  6,  // AOI222: A1, A2, B1, B2, C1, C2
  4,  // AOI22: A1, A2, B1, B2
  5,  // AOI311: A1, A2, A3, B, C
  4,  // AOI31: A1, A2, A3, B
  6,  // AOI321: A1, A2, A3, B1, B2, C
  7,  // AOI322: A1, A2, A3, B1, B2, C1, C2
  5,  // AOI32: A1, A2, A3, B1, B2
  7,  // AOI331: A1, A2, A3, B1, B2, B3, C1
  8,  // AOI332: A1, A2, A3, B1, B2, B3, C1, C2
  9,  // AOI333: A1, A2, A3, B1, B2, B3, C1, C2, C3
  6,  // AOI33: A1, A2, A3, B1, B2, B3
  1,  // BUF: I
  1,  // CKINVDC: A
  1,  // HB1: A
  1,  // HB2: A
  1,  // HB3: A
  1,  // HB4: A
  1,  // INV: I
  4,  // O2A1O1I: A1, A2, B, C
  4,  // OA211: A1, A2, B, C
  3,  // OA21: A1, A2, B
  5,  // OA221: A1, A2, B1, B2, C
  6,  // OA222: A1, A2, B1, B2, C1, C2
  4,  // OA22: A1, A2, B1, B2
  4,  // OA31: A1, A2, A3, B1
  5,  // OA32: A1, A2, A3, B1, B2
  7,  // OA331: A1, A2, A3, B1, B2, B3, C1
  8,  // OA332: A1, A2, A3, B1, B2, B3, C1, C2
  9,  // OA333: A1, A2, A3, B1, B2, B3, C1, C2, C3
  6,  // OA33: A1, A2, A3, B1, B2, B3
  4,  // OAI211: A1, A2, B, C
  3,  // OAI21: A1, A2, B
  5,  // OAI221: A1, A2, B1, B2, C
  6,  // OAI222: A1, A2, B1, B2, C1, C2
  4,  // OAI22: A1, A2, B1, B2
  5,  // OAI311: A1, A2, A3, B1, C1
  4,  // OAI31: A1, A2, A3, B
  6,  // OAI321: A1, A2, A3, B1, B2, C
  7,  // OAI322: A1, A2, A3, B1, B2, C1, C2
  5,  // OAI32: A1, A2, A3, B1, B2
  7,  // OAI331: A1, A2, A3, B1, B2, B3, C1
  8,  // OAI332: A1, A2, A3, B1, B2, B3, C1, C2
  9,  // OAI333: A1, A2, A3, B1, B2, B3, C1, C2, C3
  6,  // OAI33: A1, A2, A3, B1, B2, B3
  2,  // AND2: A, B
  3,  // AND3: A, B, C
  4,  // AND4: A, B, C, D
  5,  // AND5: A, B, C, D, E
  3,  // FASN: A, B, CI
  3,  // FACON: A, B, CI
  2,  // HASN: A, B
  2,  // HACON: A, B
  3,  // MAJI: A, B, C
  3,  // MAJ: A, B, C
  2,  // NAND2: A, B
  3,  // NAND3: A, B, C
  4,  // NAND4: A, B, C, D
  5,  // NAND5: A, B, C, D, E
  2,  // NOR2: A, B
  3,  // NOR3: A, B, C
  4,  // NOR4: A, B, C, D
  5,  // NOR5: A, B, C, D, E
  2,  // OR2: A, B
  3,  // OR3: A, B, C
  4,  // OR4: A, B, C, D
  5,  // OR5: A, B, C, D, E
  2,  // XNOR2: A1, A2
  2,  // XOR2: A1, A2
  3,  // XNOR3: A1, A2, A3
  3,  // XOR3: A1, A2, A3
  8,  // AO2222: A1, A2, B1, B2, C1, C2, D1, D2
  4,  // AOAI211: A1, A2, B, C
  4,  // OAOI211: A1, A2, B, C
  2,  // IAND2: A1, B1
  3,  // IAOI21: A1, A2, B
  4,  // IAOI22: A1, A2, B1, B2
  3,  // IBAO21: A1, A2, B
  3,  // IBOA21: A1, A2, B
  2,  // INOR2: A1, B1
  2,  // INAND2: A1, B1
  3,  // INOR3: A1, B1, B2
  4,  // INOR4: A1, B1, B2, B3
  3,  // IIOAI21: A1, A2, B
  4,  // IIOAI22: A1, A2, B1, B2
  3,  // IBOAI21: A1, A2, B
  2,  // IOR2: A1, B1
  3,  // FAS: A, B, CI
  3,  // FACO: A, B, CI
  2,  // HAS: A, B
  2,  // HACO: A, B
  3,  // MAJORITYAOI222: A, B, C
  4,  // MAJORITYAOI22: A1, A2, B1, B2
  4,  // MAJORITYOAI22: A1, A2, B1, B2
  3,  // MUX2: I0, I1, S
  3,  // MUX2N: I0, I1, S
  5,  // MUX3: I0, I1, I2, S0, S1
  5,  // MUX3N: I0, I1, I2, S0, S1
  6,  // MUX4: I0, I1, I2, I3, S0, S1
  6,  // MUX4N: I0, I1, I2, I3, S0, S1
 ];
}

/*
pub struct MLCADDesignContest2025StdLib();

impl StandardCellTypeAttribute for MLCADDesignContest2025StdLib {
 fn get_celltype(
  &self,
  macro_name: &CompactString, pin_name: &CompactString, top_name: &CompactString,
  ) -> u16 {
   let celltype_name = 
    if (macro_name.split("x").collect::<Vec<_>>()[0] != CompactString::from("HA")) &&
    (macro_name.split("x").collect::<Vec<_>>()[0] != CompactString::from("FA")) { macro_name.split("x").collect::<Vec<_>>()[0] } 
    else { &(format!("{}{}", macro_name.split("x").collect::<Vec<_>>()[0], pin_name)) };
   match celltypeHash.get(&celltype_name) {
    Some(number) => *number,
    _ => match SEQ_REGEX.is_match(macro_name.as_str()) || macro_name == top_name {
     true => 999,
     _ => {use netlistdb::{HierName};
           panic!("Cannot recognize unknown celltype type {}, please make sure the verilog netlist is synthesized from Contest tech lib.",
           HierName::single(macro_name.clone()))
          }
    }
   }
  }

       // pin type attribute
 fn get_pintype(
  &self,
  macro_name: &CompactString,
  pin_name: &CompactString
  ) -> u8 {
   let celltype_name = macro_name.split("x").collect::<Vec<_>>()[0];
   let pintype_hashkey : &str = &(format!("{}/{}", celltype_name, *pin_name));
   match pintypeHash.get(pintype_hashkey) {
    Some(number) => *number,
    _ => {
           panic!("Cannot recognize unknown pin type {}, please make sure the verilog netlist is synthesized from Contest tech lib.",
           pintype_hashkey)
          }
   }
  }


}
*/

pub struct GL0AMStdLib();
impl StandardCellTypeAttribute for GL0AMStdLib {
 fn get_celltype(
  &self,
  macro_name: &CompactString, pin_name: &CompactString, top_name: &CompactString,
  ) -> u16 {
   let celltype_name = {
    let parts: Vec<&str> = macro_name.split('_').collect();
    if parts.len() >= 2 {
     parts[1]
    } else {
     macro_name.as_str()
    }
   };
   
   let celltype_name = 
    if celltype_name != "HA" && celltype_name != "FA" { 
     celltype_name 
    } else { 
     &(format!("{}{}", celltype_name, pin_name)) 
    };
   
   match celltypeHash.get(celltype_name) {
    Some(number) => *number,
    _ => match SEQ_REGEX.is_match(macro_name.as_str()) || macro_name == top_name {
     true => 999,
     _ => {use netlistdb::{HierName};
           panic!("Cannot recognize unknown celltype type {}, please make sure the verilog netlist is synthesized from Contest tech lib.",
           HierName::single(macro_name.clone()))
          }
    }
   }
  }

       // pin type attribute
 fn get_pintype(
  &self,
  macro_name: &CompactString,
  pin_name: &CompactString
  ) -> u8 {
   let celltype_name = {
    let parts: Vec<&str> = macro_name.split('_').collect();
    if parts.len() >= 2 {
     parts[1]
    } else {
     macro_name.as_str()
    }
   };
   let pintype_hashkey : &str = &(format!("{}/{}", celltype_name, *pin_name));
   match pintypeHash.get(pintype_hashkey) {
    Some(number) => *number,
    _ => {
           panic!("Cannot recognize unknown pin type {}, please make sure the verilog netlist is synthesized from Contest tech lib.",
           pintype_hashkey)
          }
   }
  }


}

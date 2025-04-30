use sverilogparse::*;
use compact_str::CompactString;
//std::iter::zip;
use std::fmt;
use itertools::izip;
use lazy_static::lazy_static;
use regex::Regex;


pub enum StdCellTypeDef {
 Seq,
 Combo,
 Other,
}


lazy_static! {
    static ref FLOP_REGEX: Regex = Regex::new(r".*DF.*").unwrap();
}


impl fmt::Debug for StdCellTypeDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use StdCellTypeDef::*;
        write!(f, "{}", match self {
            Seq => "seq",
            Combo => "combo",
            Other => "other",
        })
    }
}

pub struct StdCellLibModules {
 pub lib_cells: Vec<(CompactString, StdCellTypeDef, usize)>,
}

impl StdCellLibModules {
 pub fn parse_module_names(svmodules: &SVerilog) -> Self {
  let x = svmodules.modules.clone().into_iter().map(|m| m.0);
  let y = x.clone().map(|m| match FLOP_REGEX.is_match(&m) {
                                            true => StdCellTypeDef::Seq,
                                            _ => StdCellTypeDef::Combo,
                                           });
  let id = 0..x.len();
  Self { 
   lib_cells: izip!(x,y,id).collect::<Vec<_>>(), 
  }
 }
}
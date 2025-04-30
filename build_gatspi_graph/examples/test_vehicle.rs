use netlistdb::NetlistDB;
use std::env;
use build_gatspi_graph::*;

use std::fs::File;
use std::io::Write;

fn print_type<T>(_: &T) { 
    println!("{:?}", std::any::type_name::<T>());
}


fn main() {
    clilog::init_stderr_color_debug();
    clilog::enable_timer("netlistdb");
    clilog::enable_timer("");
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 2 || args.len() == 3,
            "Usage: {} <verilog_path> [<top_module>]", args[0]);

    let db = NetlistDB::from_sverilog_file(
        &args[1],
        args.get(2).map(|x| x.as_ref()),
        &build_gatspi_graph::StdCellPinDefs()
    ).expect("Error parsing the verilog into netlist");

    println!("Benchmark statistics for {}", args[1]);
    println!("top module: {}", db.name);
    println!("num cells:  {}", db.num_cells);
    println!("num nets:   {}", db.num_nets);
    println!("num pins:   {}", db.num_pins);
    
    let mut w = File::create("foo.txt").unwrap();
    writeln!(&mut w, "{:?}", db).unwrap();


    let time_build_gatspi = clilog::stimer!("build_gatspi");
    let x = GATSPIGraph::build_graph(&db, &build_gatspi_graph::stdlib_attributes::MLCADDesignContest2025StdLib() );
    print_type(&x);
    clilog::finish!(time_build_gatspi);
    writeln!(&mut w, "GATSPI: {:?}", x).unwrap();

    let mut ww = File::create("gatspi.pkl").unwrap();
    serde_pickle::to_writer(&mut ww, &x, Default::default()).unwrap();
    
    let mut www = File::create("netlistdb.debug.pkl").unwrap();
    serde_pickle::to_writer(&mut www, &(Vec::from(db.net2pin.start.clone()), Vec::from(db.net2pin.items.clone()), Vec::from(db.pin2net.clone()), format!("{:?}", &db.pinnames) ) , Default::default()).unwrap();
    
}

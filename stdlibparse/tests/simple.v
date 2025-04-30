module simple (
inp1,
inp2,
ispd_clk,
out
);

// Start PIs
   (* src = "test attribute" *)
input inp1;
input inp2;
input ispd_clk;

// Start POs
output out;

// Start wires
wire n1;
wire n2;
wire inp1;
wire inp2;
wire ispd_clk;
wire out;

// Start cells
na02s01 u1 ( .a(inp1), .b(inp2), .o(n1) );
ms00f80 f1 ( .d(n1), .ck(ispd_clk), .o(n2) );
in01s01 u2 ( .a(n2), .o(out) );

endmodule

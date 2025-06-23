#!/usr/bin/env python3
"""
Test Script for Synthesis Feedback Loop
=======================================

This script demonstrates the fixed synthesis feedback mechanism.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.eda_langchain_tools import yosys_synthesize

def test_synthesis_feedback():
    """Test the synthesis feedback loop"""
    
    print("🧪 Testing Synthesis Feedback Loop")
    print("=" * 50)
    
    # Test with broken RTL (the original generated code)
    broken_rtl = """
// Generated RTL for dot_product
module dot_product #(
    parameter int N = 8,
    parameter int WIDTH = 8
) (
    input  logic clk,
    input  logic rst,
    input  logic signed [N-1:0][WIDTH-1:0] A,
    input  logic signed [N-1:0][WIDTH-1:0] B,
    output logic signed [2*WIDTH+3:0] dot_out,
    output logic valid
);

    // Basic logic implementation
    always_comb begin
        // Add combinational logic here
    end
    
    // Register implementation if needed
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset logic
        end else begin
            // Sequential logic
        end
    end
    
endmodule
"""

    # Write broken RTL to file
    with open("/tmp/broken_rtl.v", "w") as f:
        f.write(broken_rtl)
    
    print("🔥 Testing broken RTL synthesis...")
    broken_result = yosys_synthesize("/tmp/broken_rtl.v", "dot_product", "generic")
    
    print(f"❌ Broken RTL synthesis success: {broken_result['success']}")
    if not broken_result['success']:
        print(f"💥 Synthesis errors: {broken_result['errors'][:200]}...")
        print(f"📋 This would trigger the feedback loop!")
    
    # Test with fixed RTL
    fixed_rtl = """
// Fixed RTL for dot_product  
module dot_product #(
    parameter N = 8,
    parameter WIDTH = 8
) (
    input  wire clk,
    input  wire rst,
    input  wire signed [N*WIDTH-1:0] A,  
    input  wire signed [N*WIDTH-1:0] B,  
    output reg signed [2*WIDTH+3:0] dot_out,
    output reg valid
);

    reg signed [2*WIDTH+3:0] sum;
    integer i;

    always @(posedge clk) begin
        if (rst) begin
            dot_out <= 0;
            valid <= 0;
            sum <= 0;
        end else begin
            sum = 0;
            for (i = 0; i < N; i = i + 1) begin
                sum = sum + A[i*WIDTH +: WIDTH] * B[i*WIDTH +: WIDTH];
            end
            dot_out <= sum;
            valid <= 1'b1;
        end
    end
    
endmodule
"""

    # Write fixed RTL to file
    with open("/tmp/fixed_rtl.v", "w") as f:
        f.write(fixed_rtl)
    
    print("\n✅ Testing fixed RTL synthesis...")
    fixed_result = yosys_synthesize("/tmp/fixed_rtl.v", "dot_product", "generic")
    
    print(f"✅ Fixed RTL synthesis success: {fixed_result['success']}")
    if fixed_result['success']:
        print(f"🎉 Fixed RTL synthesizes successfully!")
        print(f"📊 Stats: {fixed_result.get('stats', {})}")
    
    print("\n" + "=" * 50)
    print("🧠 **KEY INSIGHT**: The new system will:")
    print("1. ❌ Detect synthesis failures")  
    print("2. 🔧 Extract specific error messages")
    print("3. ➡️  Route back to RTL generator with feedback")
    print("4. 🔄 Regenerate corrected RTL")
    print("5. ✅ Achieve synthesis success")
    print("\nThis creates a **self-correcting agent system**! 🤖")

if __name__ == "__main__":
    test_synthesis_feedback() 
# SDC constraints for seq_detector_0011
# Generated automatically

# Clock definition
create_clock -name clk -period 1.1 [get_ports clk]

# Input/Output delays (adjust as needed)
set_input_delay -clock clk 0.1 [all_inputs]
set_output_delay -clock clk 0.1 [all_outputs]

# Drive strengths and loads
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 [all_inputs]
set_load 0.1 [all_outputs]

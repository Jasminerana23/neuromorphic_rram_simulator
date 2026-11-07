"""
CROSSBAR ARRAY
Neuromorphic RRAM Crossbar Simulator
128×128 array of RRAM devices
"""

import numpy as np
from rram_device import RRAMDevice

class CrossbarArray:
    """
    2D array of RRAM devices arranged as crossbar
    
    Structure:
    - Rows: Input signals (neurons)
    - Columns: Output signals (neurons)
    - Intersections: RRAM devices (synapses)
    
    Physics: Matrix multiplication using Ohm's law
    I = V / R (for each RRAM)
    """
    
    def __init__(self, rows=128, cols=128):
        """
        Initialize crossbar with RRAM devices
        
        Args:
            rows (int): Number of rows (input neurons)
            cols (int): Number of columns (output neurons)
        """
        self.rows = rows
        self.cols = cols
        
        # Create 2D array of RRAM devices
        # devices[i][j] = RRAM at intersection (i, j)
        self.devices = []
        for i in range(rows):
            row = []
            for j in range(cols):
                # Each intersection gets one RRAM device
                rram = RRAMDevice(initial_resistance=10000)
                row.append(rram)
            self.devices.append(row)
        
        print(f"✅ Crossbar created: {rows}×{cols} = {rows*cols} RRAMs")
    
    # ================================================
    # FORWARD PASS: Read Operation (Inference)
    # ================================================
    
    def forward(self, input_voltages):
        """
        Perform forward pass: Convert input to output
        
        Physics: Ohm's Law
        I = V / R
        V_out = I × R_load (virtual load resistance)
        
        Args:
            input_voltages (list): Voltage on each row
                                 Length should be = self.rows
        
        Returns:
            list: Output current/voltage from each column
        """
        
        # Check input size
        if len(input_voltages) != self.rows:
            raise ValueError(f"Expected {self.rows} inputs, got {len(input_voltages)}")
        
        # Calculate output for each column
        output_currents = []
        
        for col in range(self.cols):
            # Sum currents from all rows in this column
            total_current = 0
            
            for row in range(self.rows):
                # Get RRAM at this intersection
                rram = self.devices[row][col]
                
                # Get current using Ohm's law: I = V / R
                input_voltage = input_voltages[row]
                current = rram.read(voltage=input_voltage)
                
                # Add to total
                total_current += current
            
            # Convert current to voltage (for next layer)
            # Assuming load resistance = 1000 Ω
            output_voltage = total_current * 1000
            output_currents.append(output_voltage)
        
        return output_currents
    
    # ================================================
    # BACKWARD PASS: Learning (Hebbian Rule)
    # ================================================
    
    def hebbian_learn(self, input_voltages, output_voltages, 
                     learning_rate=0.01):
        """
        Update RRAM resistances using Hebbian learning rule
        
        Rule: "Neurons that fire together, wire together"
        If input AND output both high → Strengthen (decrease R)
        If input high AND output low → Weaken (increase R)
        
        Args:
            input_voltages (list): Input signals
            output_voltages (list): Output signals
            learning_rate (float): How fast to learn (0.0 to 1.0)
        """
        
        for row in range(self.rows):
            for col in range(self.cols):
                input_v = input_voltages[row]
                output_v = output_voltages[col]
                
                # Hebbian rule: ΔW = η × V_in × V_out
                # Where: η = learning rate
                
                # Normalize voltages to 0-1 range (assuming 0-1V range)
                input_norm = min(input_v / 1.0, 1.0)
                output_norm = min(output_v / 1.0, 1.0)
                
                # Calculate voltage to apply to RRAM
                # Positive: If both high → Strengthen (SET)
                # Negative: If mismatch → Weaken (RESET)
                
                if input_norm > 0.5 and output_norm > 0.5:
                    # BOTH HIGH: Strengthen connection
                    # Apply positive voltage
                    program_voltage = +1.8
                elif input_norm > 0.5 and output_norm <= 0.5:
                    # INPUT HIGH, OUTPUT LOW: Weaken
                    # Apply negative voltage
                    program_voltage = -1.8
                else:
                    # Otherwise: No change
                    program_voltage = 0
                
                # Apply voltage to RRAM
                if program_voltage != 0:
                    rram = self.devices[row][col]
                    rram.write(voltage=program_voltage, 
                              duration=learning_rate*1e-9)
    
    # ================================================
    # UTILITY FUNCTIONS
    # ================================================
    
    def get_weights(self):
        """
        Get current resistance values as 2D array
        Higher R = Weaker connection
        Lower R = Stronger connection
        
        Returns:
            numpy array: 2D array of resistances
        """
        weights = np.zeros((self.rows, self.cols))
        
        for i in range(self.rows):
            for j in range(self.cols):
                weights[i][j] = self.devices[i][j].resistance
        
        return weights
    
    def set_weights(self, weights):
        """
        Directly set RRAM resistances from array
        Useful for loading pre-trained weights
        
        Args:
            weights (numpy array or list): 2D array of resistances
        """
        weights = np.array(weights)
        
        if weights.shape != (self.rows, self.cols):
            raise ValueError(f"Weight shape {weights.shape} doesn't match "
                           f"crossbar shape ({self.rows}, {self.cols})")
        
        for i in range(self.rows):
            for j in range(self.cols):
                self.devices[i][j].resistance = weights[i][j]
    
    def get_conductance(self):
        """
        Get conductance (1/R) as 2D array
        Higher conductance = Stronger connection
        
        Returns:
            numpy array: 2D array of conductances
        """
        weights = self.get_weights()
        conductance = 1.0 / weights
        return conductance
    
    def get_multilevel_weights(self):
        """
        Convert all resistances to 8-bit levels (0-255)
        
        Returns:
            numpy array: 2D array of levels
        """
        weights = np.zeros((self.rows, self.cols), dtype=int)
        
        for i in range(self.rows):
            for j in range(self.cols):
                level = self.devices[i][j].get_multilevel()
                weights[i][j] = level
        
        return weights
    
    def reset_all(self):
        """Reset all RRAMs to initial state"""
        for i in range(self.rows):
            for j in range(self.cols):
                self.devices[i][j].reset()
    
    def get_statistics(self):
        """
        Get statistics about crossbar state
        
        Returns:
            dict: Statistics (mean R, std R, min R, max R)
        """
        weights = self.get_weights()
        
        return {
            'mean_resistance': np.mean(weights),
            'std_resistance': np.std(weights),
            'min_resistance': np.min(weights),
            'max_resistance': np.max(weights),
            'mean_conductance': np.mean(1.0/weights),
        }


# ================================================
# TEST CODE
# ================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CROSSBAR ARRAY SIMULATOR")
    print("=" * 60)
    
    # Create small crossbar for testing
    crossbar = CrossbarArray(rows=3, cols=3)
    
    # Create some input signals
    inputs = [0.5, 0.3, 0.8]  # 3 inputs for 3 rows
    
    print(f"\n📥 Input voltages: {inputs}")
    
    # Forward pass (inference)
    outputs = crossbar.forward(inputs)
    print(f"📤 Output voltages: {[f'{x:.4f}' for x in outputs]}")
    
    # Get weights before learning
    print("\n📊 WEIGHTS BEFORE LEARNING:")
    weights_before = crossbar.get_weights()
    print(weights_before)
    
    # Simulate learning
    print("\n🧠 LEARNING (Hebbian rule)...")
    for epoch in range(5):
        outputs = crossbar.forward(inputs)
        crossbar.hebbian_learn(inputs, outputs, learning_rate=0.5)
        print(f"  Epoch {epoch+1} complete")
    
    # Get weights after learning
    print("\n📊 WEIGHTS AFTER LEARNING:")
    weights_after = crossbar.get_weights()
    print(weights_after)
    
    # Check differences
    print("\n🔄 WEIGHT CHANGES:")
    print(weights_after - weights_before)
    
    # Get statistics
    print("\n📈 STATISTICS:")
    stats = crossbar.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n✅ Crossbar test complete!")
    
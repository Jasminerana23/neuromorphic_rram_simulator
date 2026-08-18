"""
RRAM Device Model
Simulates physical RRAM device with resistance switching
"""

import numpy as np

class RRAMDevice:
    """Virtual RRAM Device"""
    
    def __init__(self, initial_resistance=10000, 
                 max_resistance=100000, 
                 min_resistance=1000):
        self.resistance = initial_resistance
        self.max_resistance = max_resistance
        self.min_resistance = min_resistance
        self.history = []
        
    def read(self, voltage=0.1):
        """Read current using Ohm's Law: I = V / R"""
        current = voltage / self.resistance
        return current
    
    def write(self, voltage, duration=1e-9):
        """Change resistance based on voltage"""
        alpha = 0.001
        
        if voltage > 1.5:
            dR = -alpha * voltage * duration * self.resistance
        elif voltage < -1.5:
            dR = alpha * abs(voltage) * duration * self.resistance
        else:
            dR = 0
        
        self.resistance += dR
        self.resistance = np.clip(self.resistance, 
                                  self.min_resistance, 
                                  self.max_resistance)
        self.history.append(self.resistance)
    
    def get_multilevel(self):
        """Convert resistance to 8-bit level (0-255)"""
        normalized = (self.resistance - self.min_resistance) / \
                     (self.max_resistance - self.min_resistance)
        return int(np.clip(normalized * 255, 0, 255))
    
    def reset(self):
        """Reset device"""
        self.resistance = 10000
        self.history = []

if __name__ == "__main__":
    rram = RRAMDevice()
    print("Initial resistance:", rram.resistance)
    
    rram.write(voltage=+1.8, duration=50e-9)
    print("After SET:", rram.resistance)
    
    current = rram.read(voltage=0.1)
    print("Current:", current)
    
    print("✅ RRAM Device working!")
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

dirt_level = ctrl.Antecedent(np.arange(0, 101, 1), 'dirt_level')
load_size = ctrl.Antecedent(np.arange(0, 11, 1), 'load_size')


wash_time = ctrl.Consequent(np.arange(0, 81, 1), 'wash_time')

# Membership functions for dirt_level (Low, Medium, High)
dirt_level['Low'] = fuzz.trimf(dirt_level.universe, [0, 0, 50])
dirt_level['Medium'] = fuzz.trimf(dirt_level.universe, [40, 60, 80])
dirt_level['High'] = fuzz.trimf(dirt_level.universe, [75, 100, 100])

# Membership functions for load_size (Small, Medium, Large)
load_size['Small'] = fuzz.trimf(load_size.universe, [0, 2, 4])
load_size['Medium'] = fuzz.trimf(load_size.universe, [3, 5, 7])
load_size['Large'] = fuzz.trimf(load_size.universe, [6, 8, 10])

# Membership functions for wash_time (Short, Standard, Long, Extended)
wash_time['Short'] = fuzz.trimf(wash_time.universe, [0, 0, 30])
wash_time['Standard'] = fuzz.trimf(wash_time.universe, [15, 35, 55])
wash_time['Long'] = fuzz.trimf(wash_time.universe, [40, 60, 80])
wash_time['Extended'] = fuzz.trimf(wash_time.universe, [60, 80, 80])



rule1 = ctrl.Rule(dirt_level['Low'] & load_size['Small'], wash_time['Short'])
rule2 = ctrl.Rule(dirt_level['Medium'] & load_size['Medium'], wash_time['Standard'])
rule3 = ctrl.Rule(dirt_level['High'] & load_size['Small'], wash_time['Long'])
rule4 = ctrl.Rule(dirt_level['High'] & load_size['Large'], wash_time['Extended'])
rule5 = ctrl.Rule(dirt_level['Low'] & load_size['Large'], wash_time['Standard'])

washing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])

washing_machine = ctrl.ControlSystemSimulation(washing_ctrl)


washing_machine.input['dirt_level'] = 20
washing_machine.input['load_size'] = 2.5

# Compute the result
washing_machine.compute()

print("--- Example 1: Low Dirt, Small Load ---")
print(f"Dirt Level: 20, Load Size: 2.5 kg")
print(f"Recommended Wash Time: {washing_machine.output['wash_time']:.2f} minutes")
wash_time.view(sim=washing_machine)


washing_machine.input['dirt_level'] = 90
washing_machine.input['load_size'] = 8

washing_machine.compute()

print("\n--- Example 2: High Dirt, Large Load ---")
print(f"Dirt Level: 90, Load Size: 8 kg")
print(f"Recommended Wash Time: {washing_machine.output['wash_time']:.2f} minutes")
wash_time.view(sim=washing_machine)


washing_machine.input['dirt_level'] = 30
washing_machine.input['load_size'] = 9

washing_machine.compute()

print("\n--- Example 3: Low Dirt, Large Load ---")
print(f"Dirt Level: 30, Load Size: 9 kg")
print(f"Recommended Wash Time: {washing_machine.output['wash_time']:.2f} minutes")
wash_time.view(sim=washing_machine)
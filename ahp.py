'''This Python script implements the Analytic Hierarchy Process (AHP) to prioritize biodiversity and ecosystem
function criteria for ecological analysis. It calculates priority weights for high-level criteria (biotic
composition and ecosystem function) and their sub-criteria (e.g., species richness, habitat quality) using
expert pairwise comparison inputs. The script computes consistency metrics, performs sensitivity analysis
by perturbing inputs (±10%), and generates visualizations (bar plots) for global weights and sensitivity
results.'''


#import libraries
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns


# Define criteria as constants
HIGH_LEVEL_CRITERIA = ["biotic composition (bc)", "ecosystem function (ef)"]
BC_SUB_CRITERIA = ["species richness (bc_1)", "species diversity (bc_2)", "red list species (bc_3)"]
EF_SUB_CRITERIA = ["habitat quality (ef_1)", "habitat connectivity (ef_2)", "pollinator abundance (ef_3)", 
                   "soil quality (ef_4)", "human recreation potential (ef_5)"]

# Expert inputs for pairwise comparisons (values reflect relative importance, e.g., 1=equal, 9=much more important)
EXPERT_INPUTS_HIGH = [1/3, 1, 1/7, 1, 1, 3, 1/7, 1/2, 7]  # BC/EF comparisons
EXPERT_INPUTS_BC = {
    ("species richness (bc_1)", "species diversity (bc_2)"): [1/7, 1, 1/7, 1/9, 1, 1/8, 1/7, 1, 1/8],
    ("species richness (bc_1)", "red list species (bc_3)"): [1/4, 1/7, 8, 1/9, 6, 1/7, 6, 1/2, 7],
    ("species diversity (bc_2)", "red list species (bc_3)"): [6, 1/9, 8, 9, 6, 1, 8, 1/2, 8]
}
EXPERT_INPUTS_EF = {
    ("habitat quality (ef_1)", "habitat connectivity (ef_2)"): [3, 1, 6, 1, 6, 1, 1, 2, 7],
    ("habitat quality (ef_1)", "pollinator abundance (ef_3)"): [7, 9, 8, 9, 6, 5, 6, 2, 6],
    ("habitat quality (ef_1)", "soil quality (ef_4)"): [1, 9, 9, 1, 6, 1, 7, 2, 6],
    ("habitat quality (ef_1)", "human recreation potential (ef_5)"): [9, 9, 9, 9, 8, 9, 8, 5, 7],
    ("habitat connectivity (ef_2)", "pollinator abundance (ef_3)"): [7, 9, 8, 8, 6, 1, 1, 2, 1/6],
    ("habitat connectivity (ef_2)", "soil quality (ef_4)"): [1, 9, 9, 1, 6, 1, 6, 2, 1/7],
    ("habitat connectivity (ef_2)", "human recreation potential (ef_5)"): [9, 9, 9, 9, 6, 9, 7, 3, 1/6],
    ("pollinator abundance (ef_3)", "soil quality (ef_4)"): [1/4, 9, 9, 1/8, 1/6, 1, 1, 1, 1/6],
    ("pollinator abundance (ef_3)", "human recreation potential (ef_5)"): [9, 9, 7, 8, 7, 7, 7, 3, 6],
    ("soil quality (ef_4)", "human recreation potential (ef_5)"): [9, 1, 7, 9, 7, 5, 7, 3, 6]
}

def calculate_priority_weights(ahp_matrix: np.ndarray) -> np.ndarray:
    """Calculate priority weights from an AHP matrix using the eigenvector method."""
    '''square matrix of pairwise comparisons'''
    '''return: normalized priority weights'''
    eigenvalues, eigenvectors = np.linalg.eig(ahp_matrix)
    max_eigenvalue_index = np.argmax(np.abs(eigenvalues))
    principal_eigenvector = np.abs(eigenvectors[:, max_eigenvalue_index])
    return principal_eigenvector / np.sum(principal_eigenvector)

def calculate_consistency(ahp_matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """Calculate consistency metrics for an AHP matrix"""
    '''return: tuple of lambda_max, CI, RI, CR'''
    n = ahp_matrix.shape[0]
    eigenvalues, _ = np.linalg.eig(ahp_matrix)
    lambda_max = np.real(np.max(eigenvalues))
    
    if n == 2:
        return lambda_max, 0, 0, 0
    
    CI = (lambda_max - n) / (n - 1)
    RI_values = {2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    RI = RI_values.get(n, 1.41)
    CR = CI / RI if RI != 0 else 0
    return lambda_max, CI, RI, CR

def build_ahp_matrix(criteria: List[str], expert_inputs: Dict[Tuple[str, str], List[float]]) -> np.ndarray:
    """Build an AHP matrix from expert inputs using geometric means."""
    '''criteria: list of criteria names'''
    '''expert_inputs: dictionary of pairwise comparison with expert input values'''
    '''return: AHP matrix'''
    n = len(criteria)
    ahp_matrix = np.ones((n, n))
    criteria_indices = {c: i for i, c in enumerate(criteria)}
    
    for (c1, c2), values in expert_inputs.items():
        gm = np.prod(values) ** (1/len(values))
        i, j = criteria_indices[c1], criteria_indices[c2]
        ahp_matrix[i, j] = gm
        ahp_matrix[j, i] = 1/gm
        #print(f"Pair ({c1}, {c2}): Values {values}, Geometric Mean = {gm}")
    
    return ahp_matrix

def process_ahp_level(criteria: List[str], expert_inputs: Union[List[float], Dict[Tuple[str, str], List[float]]], 
                     level_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Process an AHP level (high-level or sub-criteria) ato compute weights and consistency"""
    '''criteria: list of criteria names'''
    '''expert_input: list of highlevel or dict of subcriteria of expert inputs'''
    '''level_name: name of the level'''
    '''return: tuple of priority weights/ ahp matrix'''
    if isinstance(expert_inputs, list):  # High-level case
        geometric_mean = np.prod(expert_inputs) ** (1/len(expert_inputs))
        ahp_matrix = np.array([[1, geometric_mean], [1/geometric_mean, 1]])
    else:  # Sub-criteria case
        ahp_matrix = build_ahp_matrix(criteria, expert_inputs)
    
    weights = calculate_priority_weights(ahp_matrix)
    lambda_max, CI, RI, CR = calculate_consistency(ahp_matrix)
    
    #print results for transparency
    print(f"\n{level_name} AHP Matrix:")
    print(ahp_matrix)
    print(f"\n{level_name} Priority Weights:")
    print(pd.DataFrame({"Criteria": criteria, "Weight": weights}))
    print(f"\n{level_name} Consistency: lambda_max={lambda_max:.4f}, CI={CI:.4f}, RI={RI}, CR={CR:.4f}")
    print("Consistency acceptable" if CR <= 0.1 else "Consistency not acceptable")
    
    return weights, ahp_matrix

def perturb_expert_inputs(expert_inputs: Union[List[float], Dict[Tuple[str, str], List[float]]], 
                         perturbation: float) -> Union[List[float], Dict[Tuple[str, str], List[float]]]:
    """Perturb expert inputs by a given percentage (0.1 for ±10%)."""
    if isinstance(expert_inputs, list):
        return [x * (1 + perturbation) for x in expert_inputs]
    else:
        perturbed_inputs = {}
        for key, values in expert_inputs.items():
            perturbed_inputs[key] = [x * (1 + perturbation) for x in values]
        return perturbed_inputs

def sensitivity_analysis(criteria: List[str], expert_inputs: Union[List[float], Dict[Tuple[str, str], List[float]]], 
                        level_name: str, perturbation: float = 0.1):
    """Perform sensitivity analysis by perturbing inputs and comparing weights."""
    '''perturbation: level for perturbation'''
    # Original weights
    original_weights, _ = process_ahp_level(criteria, expert_inputs, f"{level_name} (Original)")
    
    # Perturbed weights (increase)
    perturbed_up_inputs = perturb_expert_inputs(expert_inputs, perturbation)
    perturbed_up_weights, _ = process_ahp_level(criteria, perturbed_up_inputs, f"{level_name} (+{perturbation*100}%)")
    
    # Perturbed weights (decrease)
    perturbed_down_inputs = perturb_expert_inputs(expert_inputs, -perturbation)
    perturbed_down_weights, _ = process_ahp_level(criteria, perturbed_down_inputs, f"{level_name} (-{perturbation*100}%)")

    # Compile results
    sensitivity_df = pd.DataFrame({
        "Criteria": criteria,
        "Original Weight": original_weights,
        f"Weight (+{perturbation*100}%)": perturbed_up_weights,
        f"Weight (-{perturbation*100}%)": perturbed_down_weights
    })
    print(f"\nSensitivity Analysis for {level_name}:")
    print(sensitivity_df)

def main():
    '''main fucntion to run AHP analysis and sensitivity analysis for all levels'''
    # High-Level Sensitivity Analysis
    sensitivity_analysis(HIGH_LEVEL_CRITERIA, EXPERT_INPUTS_HIGH, "High-Level", perturbation=0.1)

    # BC Sub-Criteria (original weights only for reference)
    sensitivity_analysis(BC_SUB_CRITERIA, EXPERT_INPUTS_BC, "BC Sub-Criteria", perturbation = 0.1)

    # EF Sub-Criteria Sensitivity Analysis
    sensitivity_analysis(EF_SUB_CRITERIA, EXPERT_INPUTS_EF, "EF Sub-Criteria", perturbation=0.1)

    # High-Level Original Weights for Global Calculation
    high_level_weights, _ = process_ahp_level(HIGH_LEVEL_CRITERIA, EXPERT_INPUTS_HIGH, "High-Level")

    # Global Weights (using original weights)
    bc_weights, _ = process_ahp_level(BC_SUB_CRITERIA, EXPERT_INPUTS_BC, "BC Sub-Criteria")
    bc_global_weights = bc_weights * high_level_weights[0]
    ef_weights, _ = process_ahp_level(EF_SUB_CRITERIA, EXPERT_INPUTS_EF, "EF Sub-Criteria")
    ef_global_weights = ef_weights * high_level_weights[1]
    all_criteria = BC_SUB_CRITERIA + EF_SUB_CRITERIA
    all_global_weights = np.concatenate([bc_global_weights, ef_global_weights])
    
    print("\nFinal Global Weights (Original):")
    print(pd.DataFrame({"Criteria": all_criteria, "Global Weight": all_global_weights}))
    print(f"\nTotal Global Weight Sum: {sum(all_global_weights):.6f}")

if __name__ == "__main__":
    main()


'''--- Visualization of Global Weights ---'''

# Create DataFrame for global weights plot

data = {
    "Criteria": ["Species Richness", "Species Diversity", "Red List Species", 
                 "Habitat Quality", "Habitat Connectivity", "Pollinator Abundance", 
                 "Soil Quality", "Human Recreation Potential"],
    "Weight": [0.074945, 0.264761, 0.087508, 0.267983, 0.134672, 0.068740, 0.080788, 0.020604],
    "Dimension": ["BC", "BC", "BC", "EF", "EF", "EF", "EF", "EF"]
}
df = pd.DataFrame(data)

#create bar plot of global weights
plt.figure(figsize=(10, 6))
sns.barplot(x="Criteria", y="Weight", hue="Dimension", palette={"BC": "blue", "EF": "green"}, data=df)
plt.title("Global Weights of Biodiversity Sub-Criteria")
plt.ylabel("Global Weight")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("global_weights_barplot.png")
plt.show()


'''Sensitivity Analysis Plot'''

# Define criteria and dimensions
criteria = [
    "Species Richness (BC1)", "Species Diversity (BC2)", "Red List Species (BC3)",
    "Habitat Quality (EF1)", "Habitat Connectivity (EF2)", "Pollinator Abundance (EF3)", 
    "Soil Quality (EF4)", "Human Recreation Potential (EF5)"
]
dimensions = ["BC", "BC", "BC", "EF", "EF", "EF", "EF", "EF"]

# High-level weights
hl_original = [0.427214, 0.572786]
hl_plus_10 = [0.450681, 0.549319]
hl_minus_10 = [0.401651, 0.598349]
hl_plus_20 = [0.472302, 0.527698]
hl_minus_20 = [0.373701, 0.626299]

# BC sub-criteria weights
bc_original = [0.175427, 0.619739, 0.204834]
bc_plus_10 = [0.187142, 0.620422, 0.192436]
bc_minus_10 = [0.163038, 0.617881, 0.219081]
bc_plus_20 = [0.198253, 0.620217, 0.181531]
bc_minus_20 = [0.149889, 0.614451, 0.235661]

# EF sub-criteria weights
ef_original = [0.467858, 0.235117, 0.120009, 0.141044, 0.035971]
ef_plus_10 = [0.486465, 0.235282, 0.115676, 0.130285, 0.032292]
ef_minus_10 = [0.446879, 0.234374, 0.124647, 0.153666, 0.040435]
ef_plus_20 = [0.503103, 0.235033, 0.111637, 0.121014, 0.029213]
ef_minus_20 = [0.423005, 0.232809, 0.129580, 0.168662, 0.045944]

# Compute global weights
bc_global_original = np.array(bc_original) * hl_original[0]
ef_global_original = np.array(ef_original) * hl_original[1]
global_original = np.concatenate([bc_global_original, ef_global_original])

bc_global_plus_10 = np.array(bc_plus_10) * hl_plus_10[0]
ef_global_plus_10 = np.array(ef_plus_10) * hl_plus_10[1]
global_plus_10 = np.concatenate([bc_global_plus_10, ef_global_plus_10])

bc_global_minus_10 = np.array(bc_minus_10) * hl_minus_10[0]
ef_global_minus_10 = np.array(ef_minus_10) * hl_minus_10[1]
global_minus_10 = np.concatenate([bc_global_minus_10, ef_global_minus_10])

bc_global_plus_20 = np.array(bc_plus_20) * hl_plus_20[0]
ef_global_plus_20 = np.array(ef_plus_20) * hl_plus_20[1]
global_plus_20 = np.concatenate([bc_global_plus_20, ef_global_plus_20])

bc_global_minus_20 = np.array(bc_minus_20) * hl_minus_20[0]
ef_global_minus_20 = np.array(ef_minus_20) * hl_minus_20[1]
global_minus_20 = np.concatenate([bc_global_minus_20, ef_global_minus_20])

# Create DataFrame
data = {
    "Criteria": criteria * 5,
    "Dimension": dimensions * 5,
    "Weight": (
        global_original.tolist() +
        global_plus_10.tolist() +
        global_minus_10.tolist() +
        global_plus_20.tolist() +
        global_minus_20.tolist()
    ),
    "Scenario": (
        ["Original"] * len(criteria) +
        ["+10%"] * len(criteria) +
        ["-10%"] * len(criteria) +
        ["+20%"] * len(criteria) +
        ["-20%"] * len(criteria)
    )
}
df = pd.DataFrame(data)

# Clean criteria names for display
df["Criteria"] = df["Criteria"].str.replace(r"\s*\([^)]+\)", "", regex=True)

# Set plot style
sns.set_style("whitegrid")

# Define custom color palette
custom_palette = {
    "Original": "#808080",  # Grey
    "+10%": "#87CEEB",     # Light blue
    "-10%": "#4682B4",     # Dark blue
    "+20%": "#FFDEAD",     # Light orange
    "-20%": "#FF8C00"      # Dark orange
}

# Create grouped bar plot
plt.figure(figsize=(14, 8))
bar_plot = sns.barplot(
    x="Criteria",
    y="Weight",
    hue="Scenario",
    palette= custom_palette,
    data=df
)

# Customize plot
plt.title("Sensitivity Analysis of Global Weights Across Scenarios", fontsize=14, pad=15)
plt.ylabel("Global Weight", fontsize=12)
plt.xlabel("Criteria", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.legend(title="Scenario", fontsize=10, title_fontsize=12)



# Save plot
plt.savefig("sensitivity_global_weights_barplot.png", dpi=300, bbox_inches="tight")
plt.show()

''' not relevant for further analysis'''

'''Global weights without habitat connectivity

EF_SUB_CRITERIA_nhb= ["habitat quality (ef_1)", "pollinator abundance (ef_3)", 
                   "soil quality (ef_4)", "human recreation potential (ef_5)"]

EXPERT_INPUTS_EF_nhb = {
    ("habitat quality (ef_1)", "pollinator abundance (ef_3)"): [7, 9, 8, 9, 6, 5, 6, 2, 6],
    ("habitat quality (ef_1)", "soil quality (ef_4)"): [1, 9, 9, 1, 6, 1, 7, 2, 6],
    ("habitat quality (ef_1)", "human recreation potential (ef_5)"): [9, 9, 9, 9, 8, 9, 8, 5, 7],
    ("pollinator abundance (ef_3)", "soil quality (ef_4)"): [1/4, 9, 9, 1/8, 1/6, 1, 1, 1, 1/6],
    ("pollinator abundance (ef_3)", "human recreation potential (ef_5)"): [9, 9, 7, 8, 7, 7, 7, 3, 6],
    ("soil quality (ef_4)", "human recreation potential (ef_5)"): [9, 1, 7, 9, 7, 5, 7, 3, 6]
}

def main():
    # High-Level Criteria
    high_level_weights, _ = process_ahp_level(HIGH_LEVEL_CRITERIA, EXPERT_INPUTS_HIGH, "High-Level")

    # BC Sub-Criteria
    bc_weights, _ = process_ahp_level(BC_SUB_CRITERIA, EXPERT_INPUTS_BC, "BC Sub-Criteria")

    # EF Sub-Criteria (without habitat connectivity)
    ef_weights,_ =  process_ahp_level(EF_SUB_CRITERIA, EXPERT_INPUTS_EF,"EF Sub-Criteria ")

    # Global Weights
    bc_global_weights = bc_weights * high_level_weights[0]
    ef_global_weights = ef_weights * high_level_weights [1]
    all_criteria = BC_SUB_CRITERIA + EF_SUB_CRITERIA
    all_global_weights = np.concatenate([bc_global_weights, ef_global_weights])
    
    print("\nFinal Global Weights:")
    print(pd.DataFrame({"Criteria": all_criteria, "Global Weight": all_global_weights}))
    print(f"\nTotal Global Weight Sum: {sum(all_global_weights):.6f}")

if __name__ == "__main__":
    main()

'''
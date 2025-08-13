import pandas as pd
import numpy as np


# Load the CSV file
df = pd.read_csv('/home/ubuntu/master/data/ahp_test XXXXX.csv')

# Extract column names related to pairwise comparisons
pairwise_questions = [col for col in df.columns if "When assessing biodiversity" in col]  #pairwise questions
importance_values = [df.columns[df.columns.get_loc(col) + 1] for col in pairwise_questions] #importance ratings


# Extract unique criteria from the questions
criteria_set = set()
for question in pairwise_questions:
    try:
        # Normalize criteria names
        criteria_pair = [c.strip().replace("?", "").replace("\n", "").lower() for c in question.split(":")[1].split(" or ")]
        criteria_set.update(criteria_pair)
        

    except IndexError:
        print(f"Skipping malformed question: {question}")

criteria_list = sorted(criteria_set)  # Keep criteria ordered
criteria_index = {c: i for i, c in enumerate(criteria_list)}

# Initialize the AHP matrix
n = len(criteria_list)
ahp_matrix = np.ones((n, n))  # Identity matrix since self-comparison = 1

# Fill the matrix with pairwise comparisons
for _, row in df.iterrows():
    for question_col, importance_col in zip(pairwise_questions, importance_values):
        try:
            criteria_pair = [c.strip().replace("?", "").replace("\n", "").lower() for c in question_col.split(":")[1].split(" or ")]
            if len(criteria_pair) == 2:
                c1, c2 = criteria_pair
                
                # Get importance rating
                importance = row[importance_col]
                if pd.notna(importance):  # Ignore empty responses
                    importance = float(importance)
                    
                    if c1 in criteria_index and c2 in criteria_index:
                        i, j = criteria_index[c1], criteria_index[c2]
                        ahp_matrix[i, j] = importance
                        ahp_matrix[j, i] = 1 / importance if importance != 0 else 1

                    else:
                        print(f"Skipping unknown criteria: {c1}, {c2}")
        except Exception as e:
            print(f"Error processing row: {e}")

# Print the final matrix
print("AHP Matrix:")
print(ahp_matrix)


'''
def print_nicer_matrix(matrix):
    np.set_printoptions(suppress=True, precision=4)
    print("AHP Matrix:\n", np.array(matrix))

print_nicer_matrix(ahp_matrix)
'''

## Calculate priority weights
# Sum each column
column_sums = np.sum(ahp_matrix, axis=0)

# Normalize by dividing each element by the column sum
normalized_matrix = ahp_matrix / column_sums

# Compute the average of each row
priority_weights = np.mean(normalized_matrix, axis=1)

# Print results
print("Normalized AHP Matrix:")
print(normalized_matrix)

print("\nPriority Weights:")
print(priority_weights)


'''
TEST Normalized AHP Matrix:
[[0.18590926 0.18255578 0.225      0.40909091 0.05       0.41860465
  0.0989011  0.15503876]
 [0.18590926 0.18255578 0.225      0.29220779 0.05       0.31395349
  0.0989011  0.15503876]
 [0.02065658 0.02028398 0.025      0.00649351 0.05       0.00581395
  0.0989011  0.15503876]
 [0.02655847 0.03651116 0.225      0.05844156 0.05       0.05232558
  0.0989011  0.15503876]
 [0.18590926 0.18255578 0.025      0.05844156 0.05       0.05232558
  0.01098901 0.03100775]
 [0.02323866 0.03042596 0.225      0.05844156 0.05       0.05232558
  0.0989011  0.15503876]
 [0.18590926 0.18255578 0.025      0.05844156 0.45       0.05232558
  0.0989011  0.03875969]
 [0.18590926 0.18255578 0.025      0.05844156 0.25       0.05232558
  0.3956044  0.15503876]]

Priority Weights:
[0.21563756 0.18794577 0.04777348 0.08784708 0.07452862 0.08667145
 0.13648662 0.16310942]

'''


# Create a DataFrame to display the results more clearly
criteria_df = pd.DataFrame({
    'Criteria': criteria_list,
    'Priority Weight': priority_weights
})

print("\nCriteria with Priority Weights:")
print(criteria_df)

'''
TEST Criteria with Priority Weights:
                            Criteria  Priority Weight
0        habitat connectivity (ef_2)         0.215638
1             habitat quality (ef_1)         0.187946
2  human recreation potential (ef_5)         0.047773
3        pollinator abundance (ef_3)         0.087847
4            red list species (bc_3)         0.074529
5                soil quality (ef_4)         0.086671
6           species diversity (bc_2)         0.136487
7            species richness (bc_1)         0.163109
'''

## Consistency Check
# Mu & Pereyra-Rojas (2017) 

# n is the number of criteria (8)
n = len(ahp_matrix)

# Calculate the eigenvalues of the matrix
eigvals, _ = np.linalg.eig(ahp_matrix)

#Find the dominant eigenvalue (λ_max)
lambda_max = np.real(np.max(eigvals))

#Calculate the Consistency Index (CI)
CI = (lambda_max - n) / (n - 1)

# Calculate the Random Consistency Index (RI) for n=8
RI = 1.41  # This is the standard value for n=8 (Saaty 1980)

# Calculate the Consistency Ratio (CR)
CR = CI / RI

# Output the results
print(f"Maximum Eigenvalue (lambda_max): {lambda_max}")
print(f"Consistency Index (CI): {CI}")
print(f"Random Consistency Index (RI) for n=8: {RI}")
print(f"Consistency Ratio (CR): {CR}")

# Step 7: Check if the consistency ratio is acceptable
if CR <= 0.1:
    print("The consistency ratio is acceptable (CR <= 0.1).")
else:
    print("The consistency ratio is not acceptable (CR > 0.1).")


'''
TEST 
Maximum Eigenvalue (lambda_max): 10.946837377075195
Consistency Index (CI): 0.42097676815359925
Random Consistency Index (RI) for n=8: 1.41
Consistency Ratio (CR): 0.2985650837968789
The consistency ratio is not acceptable (CR > 0.1).

--> The consistency of the pairwise comparisons is not acceptable. 
This indicates that the judgments made during the pairwise comparison might be inconsistent, 
which could affect the reliability of the results.
'''




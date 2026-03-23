#!/bin/bash
# Fix CEC2013 submodule bug: get_info() method missing 'self.' prefix
# This script patches the upstream bug while keeping the submodule intact

set -e

CEC2013_FILE="benchmarks/CEC2013/python3/cec2013/cec2013.py"

if [ ! -f "$CEC2013_FILE" ]; then
    echo "ERROR: CEC2013 submodule not found at $CEC2013_FILE"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# Check if already patched (look for 'self.get_fitness_goptima' in get_info)
if grep -q "self.get_fitness_goptima" "$CEC2013_FILE"; then
    echo "CEC2013 already patched."
    exit 0
fi

echo "Patching CEC2013 get_info() method..."

python3 << 'PATCH_SCRIPT'
filepath = "benchmarks/CEC2013/python3/cec2013/cec2013.py"

with open(filepath, 'r') as f:
    lines = f.readlines()

# Function names dictionary
FUNC_NAMES = {
    1: "Five-Uneven-Peak", 2: "Equal Maxima", 3: "Uneven Decreasing Max.",
    4: "Himmelblau", 5: "Six-Hump Camel Back", 6: "Shubert", 7: "Vincent",
    8: "Shubert", 9: "Vincent", 10: "Modified Rastrigin",
    11: "Composition Function 1", 12: "Composition Function 2",
    13: "Composition Function 3", 14: "Composition Function 3",
    15: "Composition Function 4", 16: "Composition Function 3",
    17: "Composition Function 4", 18: "Composition Function 3",
    19: "Composition Function 4", 20: "Composition Function 4"
}

new_lines = []
i = 0
patched = False
func_names_added = False

while i < len(lines):
    line = lines[i]

    # Add function names dictionary after __dimensions_ line
    if '__dimensions_ = ' in line and not func_names_added:
        new_lines.append(line)
        new_lines.append('\n')
        new_lines.append('    # Function names for get_info() - added by fix-cec2013.sh\n')
        new_lines.append('    __func_names_ = {\n')
        for k, v in FUNC_NAMES.items():
            new_lines.append(f'        {k}: "{v}",\n')
        new_lines.append('    }\n')
        func_names_added = True
        i += 1
        continue

    # Replace get_info method
    if 'def get_info(self):' in line:
        # Write the new get_info method
        new_lines.append('    def get_info(self):\n')
        new_lines.append('        return {\n')
        new_lines.append('            "name": self.__func_names_.get(self._CEC2013__nfunc_, f"Function {self._CEC2013__nfunc_}"),\n')
        new_lines.append('            "fbest": self.get_fitness_goptima(),\n')
        new_lines.append('            "dimension": self.get_dimension(),\n')
        new_lines.append('            "nogoptima": self.get_no_goptima(),\n')
        new_lines.append('            "maxfes": self.get_maxfes(),\n')
        new_lines.append('            "rho": self.get_rho(),\n')
        new_lines.append('        }\n')

        # Skip the old get_info method (find the closing brace)
        i += 1
        brace_count = 0
        while i < len(lines):
            if '{' in lines[i]:
                brace_count += lines[i].count('{')
            if '}' in lines[i]:
                brace_count -= lines[i].count('}')
                if brace_count <= 0:
                    i += 1
                    break
            i += 1
        patched = True
        continue

    new_lines.append(line)
    i += 1

if not patched:
    print("ERROR: Could not find get_info() method to patch")
    exit(1)

with open(filepath, 'w') as f:
    f.writelines(new_lines)

print("CEC2013 patched successfully!")
PATCH_SCRIPT

# Verify the patch worked
if grep -q "self.get_fitness_goptima" "$CEC2013_FILE"; then
    echo "Patch verified: get_info() now uses self. prefix"
else
    echo "ERROR: Patch verification failed"
    exit 1
fi

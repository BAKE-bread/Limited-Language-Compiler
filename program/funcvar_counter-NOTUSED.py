import sys
import re

def extract_var_counts(source_code):
    """
    Parse the source code according to the EBNF grammar and count local variable declarations
    and function parameters in each function and the main block.
    
    Returns a dictionary with function names as keys and variable counts as values.
    """
    # Remove single-line comments
    source_code = re.sub(r'//.*', '', source_code, flags=re.MULTILINE)
    
    # Extract program structure
    program_match = re.match(r'\s*program\s+(\w+)\s*{(.*)}', source_code, re.DOTALL)
    if not program_match:
        return {"error": "Invalid program structure"}
    
    program_body = program_match.group(2)
    
    # Dictionary to store var counts for each function and main
    var_counts = {}
    
    # Extract all function definitions
    func_pattern = re.compile(r'func\s+(\w+)\s*\(([^)]*)\)\s*{(.*?)\s*return\s+[^;]+;?\s*}', re.DOTALL)
    for func_match in func_pattern.finditer(program_body):
        func_name = func_match.group(1)
        func_params = [p.strip() for p in func_match.group(2).split(',') if p.strip()]
        func_body = func_match.group(3)
        
        # Count parameters as variables
        param_count = len(func_params)
        
        # Count 'let' variable declarations in function body
        let_vars = re.findall(r'let\s+(\w+)', func_body)
        
        # Total count is parameters + let declarations
        total_count = param_count + len(let_vars)
        
        var_counts[func_name] = total_count
    
    # Extract main block
    main_match = re.search(r'main\s*{(.*?)}', program_body, re.DOTALL)
    if main_match:
        main_body = main_match.group(1)
        let_vars_main = re.findall(r'let\s+(\w+)', main_body)
        var_counts["main"] = len(let_vars_main)
    
    return var_counts

def main():
    # Check if a filename was provided
    if len(sys.argv) != 2:
        print("Usage: python var_counter.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        # Read the source code file
        with open(filename, "r") as f:
            source_code = f.read()
        
        # Extract the variable counts
        var_counts = extract_var_counts(source_code)
        
        # Prepare output text
        output_text = ""
        for func_name, count in sorted(var_counts.items()):
            line = f"{func_name}:{count}"
            output_text += line + "\n"
            print(line)
        
        # Write to output file
        output_filename = f"var_counts.txt"
        with open(output_filename, "w") as f:
            f.write(output_text)
            
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
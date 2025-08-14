import re

# --- EBNF Based Lexer ---
# The Lexer (or scanner/tokenizer) is responsible for converting a stream of characters (the source code)
# into a stream of tokens. Each token represents a meaningful unit in the language, like a keyword,
# identifier, number, or operator.
class Lexer:
    def __init__(self, text):
        self.text = text  # The source code string.
        self.pos = 0  # The current position in the text.
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None
        self.tokens = self._tokenize()  # The list of all tokens generated from the text.
        self.token_idx = 0  # The index of the current token to be served to the parser.

    # Advances the position pointer and updates the current character.
    def _advance(self):
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    # Skips over whitespace and single-line comments ('//').
    def _skip_whitespace_and_comments(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self._advance()
            elif self.current_char == '/' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == '/':
                while self.current_char is not None and self.current_char != '\n':
                    self._advance()
            else:
                break

    # The core method that reads characters and forms the next token.
    def _get_next_token_str(self):
        self._skip_whitespace_and_comments()
        if self.current_char is None:
            return ('EOF', None)  # End of File token.

        # Identifiers and Keywords (e.g., my_var, program, if)
        if self.current_char.isalpha() or self.current_char == '_':
            ident = ""
            while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
                ident += self.current_char
                self._advance()
            keywords = {"program", "func", "main", "return", "let", "if", "else", "while", "input", "output", "struct", "int"}
            if ident in keywords:
                return (ident.upper(), ident) # Return as a keyword token (e.g., 'IF')
            return ('IDENT', ident) # Return as a generic identifier token

        # Numbers (e.g., 123, 0)
        if self.current_char.isdigit():
            num_str = ""
            while self.current_char is not None and self.current_char.isdigit():
                num_str += self.current_char
                self._advance()
            return ('NUMBER', int(num_str))

        # Two-character operators (e.g., ==, !=, <=, >=)
        op_map_two_char = {'==': 'EQ', '!=': 'NEQ', '<=': 'LTE', '>=': 'GTE'}
        if self.pos + 1 < len(self.text):
            two_char_op = self.text[self.pos:self.pos+2]
            if two_char_op in op_map_two_char:
                self._advance()
                self._advance()
                return (op_map_two_char[two_char_op], two_char_op)

        # One-character operators and delimiters (e.g., +, -, *, /, (, ), {, }, ;)
        op_map_one_char = {
            '+': 'PLUS', '-': 'MINUS', '*': 'MUL', '/': 'DIV',
            '(': 'LPAREN', ')': 'RPAREN', '{': 'LBRACE', '}': 'RBRACE',
            '[': 'LBRACKET', ']': 'RBRACKET',
            ';': 'SEMI', ',': 'COMMA', '=': 'ASSIGN',
            '<': 'LT', '>': 'GT', ':': 'COLON', '.': 'DOT_OP',
        }
        if self.current_char in op_map_one_char:
            char = self.current_char
            token_type = op_map_one_char[char]
            self._advance()
            return (token_type, char)

        raise Exception(f"Lexer error: Unknown character '{self.current_char}' at pos {self.pos}")

    # Performs the initial pass to convert the entire text into a list of tokens.
    def _tokenize(self):
        tokens_list = []
        while True:
            token = self._get_next_token_str()
            tokens_list.append(token)
            if token[0] == 'EOF':
                break
        # Reset position to allow the text to be re-read if necessary.
        self.pos = 0
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None
        return tokens_list

    # Gets the next token from the pre-tokenized list. This is the main interface for the parser.
    def get_next_token(self):
        if self.token_idx < len(self.tokens):
            token = self.tokens[self.token_idx]
            self.token_idx += 1
            return token
        return ('EOF', None)

    # Looks ahead in the token stream without consuming the token. Useful for predictive parsing.
    def peek_token(self, offset=0):
        peek_idx = self.token_idx + offset
        if peek_idx < len(self.tokens):
            return self.tokens[peek_idx]
        return ('EOF', None)

# --- Symbol Table ---
# Represents a single symbol (variable, procedure, struct definition) in the program.
class Symbol:
    def __init__(self, name, kind, block_level, ar_level, addr, scope_name,
                 type_info=None, is_param=False):
        self.name = name  # Identifier name
        self.kind = kind  # 'var', 'proc', 'struct_def'
        self.block_level = block_level  # Lexical block level within a scope
        self.ar_level = ar_level  # Activation Record level (nesting depth of functions)
        self.addr = addr  # Address/offset in the activation record or a label for procs
        self.scope_name = scope_name  # The name of the scope it belongs to (e.g., "main", "my_func")
        self.is_param = is_param # Flag to identify if a variable is a function parameter

        # Type-related information
        self.type = type_info.get('type', 'UNDEFINED') # e.g., 'int', 'struct', 'array', 'proc'
        self.type_name = type_info.get('type_name', 'UNDEFINED') # e.g., 'int', 'MyStruct'
        self.base_type_name = type_info.get('base_type_name', None) # For arrays, the type of elements
        self.array_size = type_info.get('array_size', None) # For arrays, the number of elements
        self.struct_def_name = type_info.get('struct_def_name', None) # For struct vars, the name of the struct type
        self.struct_fields = type_info.get('struct_fields', None) # For struct defs, a dict of field info
        self.struct_field_order = type_info.get('struct_field_order', []) # For struct defs, ordered list of field names
        self.total_size = type_info.get('total_size', 1) # Total memory words the symbol occupies

        # Procedure-specific information
        self.proc_param_names = []  # Ordered list of parameter names
        self.proc_param_syms = {} # Map of parameter names to their Symbol objects
        self.proc_param_syms_ordered = [] # Ordered list of parameter Symbol objects
        self.proc_frame_size = 0  # The total size of the procedure's activation record
        self.proc_return_slot_addr = None # Address for storing the return value
        self.proc_total_scalar_args = 0 # Total number of primitive values passed as arguments


# Manages all symbols, their scopes, and type definitions.
class SymbolTable:
    def __init__(self):
        # The main table storing symbols. Key is a tuple of (name, scope_name, block_level).
        self.table = {}
        # Tracks the next available memory offset for variables in each scope.
        self.scope_addr_counters = {}
        # A dedicated dictionary to quickly look up struct type definitions.
        self.struct_definitions = {}

    # Creates a unique key for a symbol based on its name, scope, and block level.
    def _make_key(self, name, scope_name, block_level):
        return (name, scope_name, block_level)

    # Adds a parsed struct definition to the symbol table.
    def add_struct_def(self, struct_def_symbol):
        if struct_def_symbol.name in self.struct_definitions:
            print(f"Warning: Struct '{struct_def_symbol.name}' redefined.")
        self.struct_definitions[struct_def_symbol.name] = struct_def_symbol

    # Retrieves a struct definition by its name.
    def lookup_struct_def(self, struct_name):
        return self.struct_definitions.get(struct_name)

    # Recursively calculates the size (in memory words) of a given type.
    def get_type_size(self, type_info):
        type_cat = type_info.get('type')
        if type_cat == "int":
            return 1
        elif type_cat == "struct":
            struct_name_to_lookup = type_info.get('type_name') or type_info.get('struct_def_name')
            struct_def = self.lookup_struct_def(struct_name_to_lookup)
            if not struct_def:
                raise Exception(f"Unknown struct type '{struct_name_to_lookup}' for size calculation. Ensure it's defined before use.")
            return struct_def.total_size
        elif type_cat == "array":
            array_sz = type_info.get('array_size')
            base_el_type_name = type_info.get('base_type_name')
            is_struct_element = self.lookup_struct_def(base_el_type_name) is not None
            
            # Recursively find the size of a single element in the array.
            base_el_type_info = {
                'type': "struct" if is_struct_element else "int", 
                'type_name': base_el_type_name,
                'struct_def_name': base_el_type_name if is_struct_element else None
            }
            element_size = self.get_type_size(base_el_type_info)
            if array_sz is None or element_size is None:
                 raise Exception(f"Cannot calculate array size: array_sz={array_sz}, element_size={element_size} for type {type_info}")
            return array_sz * element_size
        return 1 # Default size

    # Adds a new symbol to the table.
    def add_symbol(self, symbol):
        key = self._make_key(symbol.name, symbol.scope_name, symbol.block_level)
        if key in self.table:
            print(f"Warning: Symbol {key} redefined.")

        # If the symbol is a variable, calculate its total size in memory.
        if symbol.kind == 'var' or (symbol.kind == 'var' and symbol.is_param):
             symbol.total_size = self.get_type_size({
                'type': symbol.type,
                'type_name': symbol.type_name,
                'base_type_name': symbol.base_type_name,
                'array_size': symbol.array_size,
                'struct_def_name': symbol.struct_def_name
            })
        elif symbol.kind == 'struct_def':
            pass # Size is calculated during its parsing.

        self.table[key] = symbol

    # Searches for a symbol by name, following the language's scoping rules.
    def lookup_symbol(self, name, current_scope_name, current_block_level, current_ar_level):
        # 1. Search from the current block level up to the top level of the current scope (function).
        for blk_lvl in range(current_block_level, -1, -1):
            key_current_scope = self._make_key(name, current_scope_name, blk_lvl)
            if key_current_scope in self.table:
                return self.table[key_current_scope]

        # 2. If in a nested function, check for the symbol in the 'main' scope.
        if current_scope_name != "main" and current_scope_name != "program":
            key_main_var_l1 = self._make_key(name, "main", 1)
            if key_main_var_l1 in self.table and self.table[key_main_var_l1].kind == 'var':
                 return self.table[key_main_var_l1]
            key_main_var_l0 = self._make_key(name, "main", 0) 
            if key_main_var_l0 in self.table and self.table[key_main_var_l0].kind == 'var':
                 return self.table[key_main_var_l0]

        # 3. Check for a global procedure definition.
        key_proc = self._make_key(name, "program", 0)
        if key_proc in self.table and self.table[key_proc].kind == 'proc':
            return self.table[key_proc]
        
        # 4. Check for a global struct type definition.
        key_struct_type_as_symbol = self._make_key(name, "program", 0) 
        if key_struct_type_as_symbol in self.table and self.table[key_struct_type_as_symbol].kind == 'struct_def':
            return self.table[key_struct_type_as_symbol]

        return None # Symbol not found.

    # Allocates the next available memory offset for a new variable in a given scope.
    def get_next_addr_offset(self, scope_name, ar_level, size_of_symbol=1):
        scope_key = (scope_name, ar_level)
        if scope_key not in self.scope_addr_counters:
            # The first 4 slots of an activation record are for SL, DL, RA, NARGS.
            self.scope_addr_counters[scope_key] = 4 if ar_level > 0 else 3
        addr = self.scope_addr_counters[scope_key]
        self.scope_addr_counters[scope_key] += size_of_symbol
        return addr

    # Gets the total size of the activation record for a given scope.
    def get_max_offset_used(self, scope_name, ar_level): 
        scope_key = (scope_name, ar_level)
        if scope_key in self.scope_addr_counters:
            return self.scope_addr_counters[scope_key]
        return 4 if ar_level > 0 else 3 # Return the base size if no variables were declared.

    # A utility function to print the contents of the symbol table for debugging.
    def display(self):
        print("\nSymtable:")
        print(f"{'Name':<18} {'Kind':<12} {'Type':<15} {'Scope':<15} {'Blk':<5} {'AR':<5} {'Addr':<20} {'Size':<5} {'Details'}")
        print("-" * 120)
        # Sort symbols for a clean, organized display.
        sorted_keys = sorted(self.table.keys(), key=lambda k: (
            self.table[k].ar_level, 
            self.table[k].scope_name, 
            self.table[k].block_level,
            self.table[k].addr if isinstance(self.table[k].addr, int) else float('-inf') 
        ))

        for key in sorted_keys:
            sym = self.table[key]
            addr_str = str(sym.addr) if isinstance(sym.addr, int) else sym.addr
            type_display = sym.type_name
            if sym.type == "array":
                type_display = f"{sym.base_type_name}[{sym.array_size or ''}]"
            elif sym.type == "struct" and sym.kind == "var": 
                 type_display = f"struct {sym.type_name}"
            elif sym.type == "struct_def": 
                 type_display = f"struct_def"

            details = ""
            if sym.is_param: details += "param; "
            if sym.kind == 'struct_def': 
                details += f"Fields: {len(sym.struct_fields or [])}, DefSize: {sym.total_size}"
            if sym.kind == 'proc':
                details += f"Params: {len(sym.proc_param_names)}, Scalars: {sym.proc_total_scalar_args}, FrameSize: {sym.proc_frame_size}"
            if sym.name.startswith("ret_") and sym.block_level == 0 and sym.kind == 'var': details += "ret_slot; "
            
            size_display = str(sym.total_size) if sym.kind == 'var' or sym.kind == 'struct_def' else ""
            print(f"{sym.name:<18} {sym.kind:<12} {type_display:<15} {sym.scope_name:<15} {sym.block_level:<5} {sym.ar_level:<5} {addr_str:<20} {size_display:<5} {details}")

        print("\nStruct Definitions (from dedicated dict):")
        for name, struct_sym_def in self.struct_definitions.items():
            print(f"  Struct Type: {name} (TotalSize: {struct_sym_def.total_size})")
            if struct_sym_def.struct_fields:
                for fname_ordered in struct_sym_def.struct_field_order:
                    finfo = struct_sym_def.struct_fields[fname_ordered]
                    f_type_display = finfo['type_name']
                    if finfo['type'] == "array":
                        f_type_display = f"{finfo['base_type_name']}[{finfo['array_size'] or ''}]"
                    elif finfo['type'] == "struct": 
                         f_type_display = f"struct {finfo['type_name']}"
                    print(f"    .{fname_ordered}: {f_type_display} (Offset_in_struct: {finfo['offset']}, FieldSize: {finfo['size']})")
        print("-" * 120)

# --- VM Code Generator (and Parser) ---
# This class implements a recursive descent parser that also generates P-code (VM instructions) on the fly.
class ParserAndGenerator:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()
        self.vm_code = [] # The list of generated VM instructions.
        self.symtable = SymbolTable()
        self.pc_counter = 0 # Program Counter for the next instruction to be emitted.
        # State variables to track the current parsing context.
        self.current_scope_name = "program"
        self.current_ar_level = 0
        self.current_block_level = 0
        # For backpatching: storing the index of 'int' instructions to later fill in the frame size.
        self.func_int_patch_indices = {}
        # For backpatching: mapping label names to their final PC addresses.
        self.label_to_pc = {}

    # Emits a single VM instruction and adds it to the code list.
    def _emit(self, op, l=None, a=None, comment="", nargs=None):
        instr_pc = self.pc_counter
        instruction = {'op': op, 'l': l, 'a': a, 'pc': instr_pc, 'comment': comment}
        if op == 'cal' and nargs is not None:
            instruction['nargs'] = nargs # Special field for 'cal' to know how many args to pop.
        self.vm_code.append(instruction)
        self.pc_counter += 1
        return instr_pc

    # Backpatching function: Fills in the target address of a jump instruction ('jpc', 'jmp').
    def _patch_jump_address(self, instr_index, target_pc_or_label):
        self.vm_code[instr_index]['a'] = target_pc_or_label

    # Backpatching function: Fills in the frame size of an 'int' instruction.
    def _patch_int_size(self, instr_index, frame_size):
        self.vm_code[instr_index]['a'] = frame_size

    # Reports a parsing error with context.
    def _error(self, message):
        # Display tokens around the point of error for better debugging.
        start_idx = max(0, self.lexer.token_idx - 6)
        end_idx = min(len(self.lexer.tokens), self.lexer.token_idx + 5)

        near_tokens_display = []
        for i in range(start_idx, end_idx):
            tok = self.lexer.tokens[i]
            val_str = str(tok[1]) if tok[1] is not None else ""
            is_current = "(<--ERROR HERE)" if i == self.lexer.token_idx -1 else ""
            near_tokens_display.append(f"{tok[0]}({val_str}){is_current}")

        near_tokens_str = ' '.join(near_tokens_display)
        current_tok_val_str = str(self.current_token[1]) if self.current_token[1] is not None else ""

        raise Exception(f"Parser error: {message}. At token {self.current_token[0]} ('{current_tok_val_str}'). Nearby tokens: ...{near_tokens_str}...")

    # Consumes the current token if it matches the expected type, then advances to the next token.
    def _eat(self, token_type, token_value=None):
        if self.current_token[0] == token_type and \
           (token_value is None or self.current_token[1] == token_value):
            eaten_token = self.current_token
            self.current_token = self.lexer.get_next_token()
            return eaten_token
        else:
            expected = f"'{token_type}'" + (f" with value '{token_value}'" if token_value else "")
            self._error(f"Expected {expected} but got {self.current_token[0]} ('{self.current_token[1]}')")

    # Generates a unique label name for jump targets.
    def _get_label(self, prefix="L"):
        return f"_{prefix}{self.pc_counter}_{len(self.label_to_pc)}"

    # Marks the current PC as the location for a given label.
    def _mark_label(self, label_name):
        if label_name in self.label_to_pc:
            self._error(f"Internal error: Label '{label_name}' redefined.")
        self.label_to_pc[label_name] = self.pc_counter

    # The entry point for the parser. It parses the entire program.
    def parse(self):
        self._eat('PROGRAM'); self._eat('IDENT'); self._eat('LBRACE')
        # Jump over function definitions to the start of the main block.
        main_jmp_instr_idx = self._emit('jmp', 0, "_main_entry_label", "Initial jump to main")

        # Parse any struct definitions at the top level.
        self._parse_struct_def_list()

        # Parse all function definitions.
        self.current_scope_name = "program"; self.current_ar_level = 0; self.current_block_level = 0
        while self.current_token[0] == 'FUNC':
            self._parse_func_def()

        # Mark the entry point for the main execution block.
        self._mark_label("_main_entry_label")
        self._eat('MAIN')

        # Set up the scope for the 'main' function.
        prev_scope_parser, prev_ar_parser, prev_block_parser = self.current_scope_name, self.current_ar_level, self.current_block_level
        self.current_scope_name = "main"; self.current_ar_level = 0; self.current_block_level = 0

        # Emit placeholder for main's frame allocation.
        main_int_idx = self._emit('int', 0, 0, "Allocate frame for main")
        self.func_int_patch_indices[("main", 0)] = main_int_idx

        # Parse the statements inside main.
        self._eat('LBRACE'); self.current_block_level += 1
        self._parse_stmt_list()
        self.current_block_level -= 1; self._eat('RBRACE')

        # Backpatch main's frame size now that we know how many variables it has.
        main_frame_size = self.symtable.get_max_offset_used("main", 0)
        self._patch_int_size(main_int_idx, main_frame_size)
        # Emit the halt instruction.
        self._emit('opr', 0, 0, "Return from main / Halt program")

        self.current_scope_name, self.current_ar_level, self.current_block_level = prev_scope_parser, prev_ar_parser, prev_block_parser
        self._eat('RBRACE'); self._eat('EOF')

        # Final pass: replace all label names in jump instructions with their actual PC addresses.
        for instr in self.vm_code:
            if isinstance(instr['a'], str) and instr['a'].startswith("_"):
                label = instr['a']
                if label not in self.label_to_pc:
                     self._error(f"Internal Error: Undefined label '{label}' encountered during final patch pass.")
                instr['a'] = self.label_to_pc[label]
        return self.vm_code, self.symtable

    # Parses a list of struct definitions.
    def _parse_struct_def_list(self):
        while self.current_token[0] == 'STRUCT':
            self._parse_struct_def()

    # Parses a base type specifier, e.g., 'int' or 'struct MyStruct'.
    def _parse_base_type_specifier(self):
        type_info = {'type': None, 'type_name': None, 'struct_def_name': None, 'total_size': None}
        if self.current_token[0] == 'INT':
            type_info['type'] = 'int'
            type_info['type_name'] = self._eat('INT')[1]
            type_info['total_size'] = 1
        elif self.current_token[0] == 'STRUCT':
            self._eat('STRUCT')
            struct_name = self._eat('IDENT')[1]
            struct_def_sym = self.symtable.lookup_struct_def(struct_name)
            if not struct_def_sym: self._error(f"Struct type '{struct_name}' not defined or not found.")
            type_info['type'] = 'struct'
            type_info['type_name'] = struct_name
            type_info['struct_def_name'] = struct_name
            type_info['total_size'] = struct_def_sym.total_size
        else:
            self._error(f"Expected base type specifier (int, struct IDENT) but got {self.current_token[0]}")
        return type_info

    # Parses a single struct definition block.
    def _parse_struct_def(self):
        self._eat('STRUCT')
        struct_name = self._eat('IDENT')[1]
        if self.symtable.lookup_struct_def(struct_name):
            self._error(f"Struct type '{struct_name}' already defined.")

        # Create a symbol for the struct definition itself.
        type_info_for_def_itself = {'type': "struct_def", 'type_name': struct_name}
        struct_sym_entry = Symbol(struct_name, 'struct_def', 0, 0, f"label_def_{struct_name}", "program", type_info=type_info_for_def_itself)

        fields_dict = {}
        field_order_list = []
        current_field_offset_within_struct = 0

        self._eat('LBRACE')
        # Iterate through all field declarations inside the struct.
        while self.current_token[0] != 'RBRACE':
            field_base_type_info = self._parse_base_type_specifier()
            field_name = self._eat('IDENT')[1]

            current_field_type_info = field_base_type_info.copy()
            current_field_type_info['array_size'] = None
            current_field_type_info['base_type_name'] = None

            # Handle array fields, e.g., 'int scores[10];'
            if self.current_token[0] == 'LBRACKET':
                current_field_type_info['base_type_name'] = field_base_type_info['type_name']
                current_field_type_info['type'] = 'array'

                self._eat('LBRACKET')
                if self.current_token[0] != 'NUMBER': self._error("Expected array size (number) for field.")
                size_val = self._eat('NUMBER')[1]
                if size_val <= 0: self._error("Array size for field must be positive.")
                current_field_type_info['array_size'] = size_val
                self._eat('RBRACKET')
                current_field_type_info['type_name'] = f"array_of_{current_field_type_info['base_type_name']}"

            self._eat('SEMI')

            if field_name in fields_dict:
                self._error(f"Duplicate field name '{field_name}' in struct '{struct_name}'")

            # Calculate the size of this field and its offset from the start of the struct.
            this_field_total_size = self.symtable.get_type_size(current_field_type_info)

            fields_dict[field_name] = {
                'offset': current_field_offset_within_struct,
                'type': current_field_type_info['type'],
                'type_name': current_field_type_info['type_name'],
                'base_type_name': current_field_type_info.get('base_type_name'),
                'array_size': current_field_type_info.get('array_size'),
                'size': this_field_total_size,
                'struct_def_name': current_field_type_info.get('struct_def_name')
            }
            field_order_list.append(field_name)
            current_field_offset_within_struct += this_field_total_size
        self._eat('RBRACE')
        self._eat('SEMI')

        # Store the collected field information and total size in the struct's symbol.
        struct_sym_entry.struct_fields = fields_dict
        struct_sym_entry.struct_field_order = field_order_list
        struct_sym_entry.total_size = current_field_offset_within_struct

        # Add the completed struct definition to the symbol table.
        self.symtable.add_struct_def(struct_sym_entry)
        self.symtable.add_symbol(struct_sym_entry)

    # Parses a function definition.
    def _parse_func_def(self):
        self._eat('FUNC'); func_name = self._eat('IDENT')[1]
        func_label = f"_{func_name}_entry"; self._mark_label(func_label)

        # Add the function itself as a 'proc' symbol.
        proc_type_info = {'type': 'proc', 'type_name': 'proc'}
        proc_sym = Symbol(func_name, 'proc', 0, 0, func_label, "program", type_info=proc_type_info)
        self.symtable.add_symbol(proc_sym)

        # Enter a new scope for the function.
        prev_scope_parser, prev_ar_parser, prev_block_parser = self.current_scope_name, self.current_ar_level, self.current_block_level
        self.current_scope_name = func_name
        self.current_ar_level = prev_ar_parser + 1
        self.current_block_level = 0

        # Emit a placeholder for the function's frame allocation.
        func_int_idx = self._emit('int', 0, 0, f"Allocate frame for {func_name}")
        self.func_int_patch_indices[(func_name, self.current_ar_level)] = func_int_idx

        # Parse the parameter list.
        param_syms_ordered = []
        self._eat('LPAREN')
        if self.current_token[0] != 'RPAREN':
            while True:
                param_name = self._eat('IDENT')[1]
                param_base_type_info = None

                # Parameters have explicit types (e.g., 'p: struct Point').
                if self.current_token[0] == 'COLON':
                    self._eat('COLON')
                    param_base_type_info = self._parse_base_type_specifier()
                    if self.current_token[0] == 'LBRACKET':
                        self._error("Array parameters by value are not directly supported as distinct types here.")
                else: # Default type is 'int'.
                    param_base_type_info = {'type': 'int', 'type_name': 'int', 'total_size': 1, 'struct_def_name': None}

                # Add the parameter to the symbol table as a variable within the function's scope.
                param_total_size = param_base_type_info['total_size']
                param_addr_in_ar = self.symtable.get_next_addr_offset(func_name, self.current_ar_level, param_total_size)

                param_sym_entry = Symbol(param_name, 'var', 0, self.current_ar_level,
                                   param_addr_in_ar, func_name, type_info=param_base_type_info, is_param=True)
                self.symtable.add_symbol(param_sym_entry)
                proc_sym.proc_param_names.append(param_name)
                proc_sym.proc_param_syms[param_name] = param_sym_entry
                param_syms_ordered.append(param_sym_entry)

                if self.current_token[0] == 'COMMA': self._eat('COMMA')
                else: break
        self._eat('RPAREN')
        proc_sym.proc_param_syms_ordered = param_syms_ordered

        # Calculate the total number of scalar values the parameters represent.
        total_scalars_for_proc = sum(p.total_size for p in param_syms_ordered)
        proc_sym.proc_total_scalar_args = total_scalars_for_proc

        # Allocate a slot in the activation record for the function's return value.
        ret_slot_type_info = {'type':'int', 'type_name':'int', 'total_size':1}
        ret_slot_name = f"ret_{func_name}"
        ret_slot_addr = self.symtable.get_next_addr_offset(func_name, self.current_ar_level, 1)
        self.symtable.add_symbol(Symbol(ret_slot_name, 'var', 0, self.current_ar_level, ret_slot_addr, func_name, type_info=ret_slot_type_info))
        proc_sym.proc_return_slot_addr = ret_slot_addr

        # **IMPORTANT**: Generate code to copy arguments passed by the caller into the callee's local parameter variables.
        # Arguments are on the stack below the new BP. This code loads them and stores them into their designated parameter addresses.
        if proc_sym.proc_total_scalar_args > 0:
            current_arg_read_offset_from_bp = -proc_sym.proc_total_scalar_args
            for param_s_def in proc_sym.proc_param_syms_ordered:
                if param_s_def.type == 'struct': # For struct parameters, copy each field individually.
                    struct_def_of_param = self.symtable.lookup_struct_def(param_s_def.type_name)
                    if not struct_def_of_param: self._error(f"Internal: Struct def '{param_s_def.type_name}' for param copy not found.")

                    for field_name_ordered in struct_def_of_param.struct_field_order:
                        field_s_info = struct_def_of_param.struct_fields[field_name_ordered]

                        # If a field is an array, copy each element of the array.
                        if field_s_info['type'] == 'array':
                            is_array_of_structs = self.symtable.lookup_struct_def(field_s_info['base_type_name']) is not None

                            if is_array_of_structs: # Handle arrays of structs.
                                element_struct_def = self.symtable.lookup_struct_def(field_s_info['base_type_name'])
                                if not element_struct_def: self._error(f"Internal: Expected struct def for array element {field_s_info['base_type_name']}")
                                size_of_one_struct_element = element_struct_def.total_size

                                for i in range(field_s_info['array_size']):
                                    for sub_field_name in element_struct_def.struct_field_order:
                                        sub_field_s_info = element_struct_def.struct_fields[sub_field_name]
                                        self._emit('lod', 0, current_arg_read_offset_from_bp, f"Load arg {param_s_def.name}.{field_name_ordered}[{i}].{sub_field_name}")
                                        target_addr = param_s_def.addr + field_s_info['offset'] + (i * size_of_one_struct_element) + sub_field_s_info['offset']
                                        self._emit('sto', 0, target_addr, f"Store to param {param_s_def.name}.{field_name_ordered}[{i}].{sub_field_name}")
                                        current_arg_read_offset_from_bp += 1
                            else: # Handle arrays of simple types (e.g. int).
                                size_of_one_primitive_element = 1
                                for i in range(field_s_info['array_size']):
                                    self._emit('lod', 0, current_arg_read_offset_from_bp, f"Load arg {param_s_def.name}.{field_name_ordered}[{i}]")
                                    target_addr = param_s_def.addr + field_s_info['offset'] + (i * size_of_one_primitive_element)
                                    self._emit('sto', 0, target_addr, f"Store to param {param_s_def.name}.{field_name_ordered}[{i}]")
                                    current_arg_read_offset_from_bp += 1
                        else: # Handle scalar fields of the struct parameter.
                            self._emit('lod', 0, current_arg_read_offset_from_bp, f"Load arg {param_s_def.name}.{field_name_ordered}")
                            target_addr = param_s_def.addr + field_s_info['offset']
                            self._emit('sto', 0, target_addr, f"Store to param {param_s_def.name}.{field_name_ordered}")
                            current_arg_read_offset_from_bp += 1
                else: # Handle simple scalar parameters (e.g. int).
                    self._emit('lod', 0, current_arg_read_offset_from_bp, f"Load arg {param_s_def.name}")
                    self._emit('sto', 0, param_s_def.addr, f"Store to param {param_s_def.name}")
                    current_arg_read_offset_from_bp += 1

        # Parse the function's body (list of statements).
        self._eat('LBRACE'); self.current_block_level += 1
        self._parse_stmt_list()
        # Parse the mandatory return statement.
        self._eat('RETURN'); self._parse_expr(); self._eat('SEMI')

        # Store the result of the return expression into the dedicated return slot.
        self._emit('sto', 0, proc_sym.proc_return_slot_addr, f"Store ret val to {func_name}.{ret_slot_name}")
        # Load the return value back to the top of the stack, as expected by the 'opr 0,0' convention.
        self._emit('lod', 0, proc_sym.proc_return_slot_addr, f"Load ret val from slot to TOS for OPR 0,0")

        self.current_block_level -= 1; self._eat('RBRACE')
        # Emit the return instruction.
        self._emit('opr', 0, 0, f"Return from {func_name}")

        # Backpatch the function's frame size.
        func_frame_size = self.symtable.get_max_offset_used(func_name, self.current_ar_level)
        self._patch_int_size(func_int_idx, func_frame_size)
        proc_sym.proc_frame_size = func_frame_size

        # Restore the parser's state to the previous scope.
        self.current_scope_name, self.current_ar_level, self.current_block_level = prev_scope_parser, prev_ar_parser, prev_block_parser

    # Parses a list of statements within a block.
    def _parse_stmt_list(self):
        valid_starts = ['LET', 'IDENT', 'IF', 'WHILE', 'INPUT', 'OUTPUT']
        while self.current_token[0] in valid_starts :
            self._parse_stmt(); self._eat('SEMI')
            # Safety break to prevent infinite loops during development.
            if self.lexer.token_idx > len(self.lexer.tokens) * 5 and len(self.lexer.tokens) > 100:
                self._error("Parser seems to be in a loop in stmt_list. Aborting.")

    # Parses a single statement by dispatching to the appropriate parsing function.
    def _parse_stmt(self):
        token_type = self.current_token[0]
        if token_type == 'LET': self._parse_declare_stmt()
        elif token_type == 'IDENT':
            self._parse_assignment_or_call_stmt()
        elif token_type == 'IF': self._parse_if_stmt()
        elif token_type == 'WHILE': self._parse_while_stmt()
        elif token_type == 'INPUT': self._parse_input_stmt()
        elif token_type == 'OUTPUT': self._parse_output_stmt()
        else: self._error(f"Unexpected token at start of statement: {self.current_token}")

    # Parses a variable declaration statement ('let ...').
    def _parse_declare_stmt(self):
        self._eat('LET')
        var_name = self._eat('IDENT')[1]

        var_type_info = None
        is_initialized = False
        has_explicit_type = False

        # Check for explicit type declaration (e.g., ': int', ': struct Point').
        if self.current_token[0] == 'COLON':
            self._eat('COLON')
            var_base_type_info = self._parse_base_type_specifier()
            var_type_info = var_base_type_info.copy()
            var_type_info['array_size'] = None
            var_type_info['base_type_name'] = None
            has_explicit_type = True

            # Handle array declarations (e.g., 'let a: int[5]').
            if self.current_token[0] == 'LBRACKET':
                var_type_info['base_type_name'] = var_base_type_info['type_name']
                var_type_info['type'] = 'array'
                self._eat('LBRACKET')
                if self.current_token[0] != 'NUMBER': self._error("Expected array size (number) for variable.")
                size = self._eat('NUMBER')[1]
                if size <= 0: self._error("Array size for variable must be positive.")
                var_type_info['array_size'] = size
                self._eat('RBRACKET')
                var_type_info['type_name'] = f"array_of_{var_type_info['base_type_name']}"

            if self.current_token[0] == 'ASSIGN':
                is_initialized = True
        
        # Handle declaration with implicit type and initialization (e.g., 'let x = 5;'). Type is inferred as 'int'.
        elif self.current_token[0] == 'ASSIGN':
            var_type_info = {'type': 'int', 'type_name': 'int', 'array_size': None, 'base_type_name': None, 'struct_def_name': None, 'total_size': 1}
            is_initialized = True
        # Handle declaration with implicit type and no initialization (e.g., 'let x;'). Type is 'int'.
        elif self.current_token[0] == 'SEMI':
            var_type_info = {'type': 'int', 'type_name': 'int', 'array_size': None, 'base_type_name': None, 'struct_def_name': None, 'total_size': 1}
            is_initialized = False
        else:
            self._error(f"Expected ':' for type declaration, '=' for initialization, or ';' for default type declaration after 'let {var_name}'. Got {self.current_token[0]} ({self.current_token[1]})")

        # Add the new variable to the symbol table.
        var_total_size = self.symtable.get_type_size(var_type_info)
        var_addr = self.symtable.get_next_addr_offset(self.current_scope_name, self.current_ar_level, var_total_size)
        var_sym = Symbol(var_name, 'var', self.current_block_level, self.current_ar_level,
                         var_addr, self.current_scope_name, type_info=var_type_info)
        if var_sym.total_size is None and var_total_size is not None: var_sym.total_size = var_total_size
        self.symtable.add_symbol(var_sym)

        # If initialized, parse the expression and emit code to store its value.
        if is_initialized:
            if var_sym.type == 'array' or var_sym.type == 'struct':
                 # This language disallows direct initialization of aggregates.
                 if has_explicit_type:
                     self._error(f"Direct initialization of explicitly typed arrays/structs ('{var_name}' of type {var_sym.type_name}) with '=' is not supported. Assign elements/fields individually.")
            self._eat('ASSIGN')
            self._parse_expr() # Leaves the value on top of the stack.
            level_diff = self.current_ar_level - var_sym.ar_level
            self._emit('sto', level_diff, var_sym.addr, f"Init {var_sym.scope_name}.{var_name}")

    # Parses a statement that begins with an identifier, which could be an assignment or a procedure call.
    def _parse_assignment_or_call_stmt(self):
        ident_token_obj = self.current_token
        ident_name = ident_token_obj[1]

        # Look up the identifier to see what it is.
        base_sym = self.symtable.lookup_symbol(ident_name, self.current_scope_name, self.current_block_level, self.current_ar_level)
        if not base_sym:
            self._error(f"Undeclared identifier '{ident_name}'")

        self._eat('IDENT')

        # If it's a procedure and is followed by '(', it's a procedure call statement.
        if base_sym.kind == 'proc' and self.current_token[0] == 'LPAREN':
            self._parse_func_call_expr(base_sym)
            # A function call used as a statement has its return value unused. Pop it.
            self._emit('opr', 0, 16, "Pop func call result if unused")
            return

        # Otherwise, it must be an L-value (the left-hand side of an assignment).
        current_lhs_type_info = {
            'type': base_sym.type,
            'type_name': base_sym.type_name,
            'base_type_name': base_sym.base_type_name,
            'array_size': base_sym.array_size,
            'struct_def_name': base_sym.struct_def_name,
        }
        num_selectors_parsed = 0

        # For complex L-values (e.g., a.b[i].c), first push the base address of the variable onto the stack.
        if self.current_token[0] == 'DOT_OP' or self.current_token[0] == 'LBRACKET':
            self._emit('lit', 0, base_sym.addr, f"LIT base_var_offset_of_{base_sym.name}")

        # Parse a chain of field access ('.') and array indexing ('[]').
        while self.current_token[0] == 'DOT_OP' or self.current_token[0] == 'LBRACKET':
            num_selectors_parsed += 1
            if self.current_token[0] == 'DOT_OP': # Field access
                if current_lhs_type_info['type'] != 'struct':
                    self._error(f"Cannot apply '.field' access to non-struct type '{current_lhs_type_info.get('type_name','<unknown>')}'.")

                self._eat('DOT_OP')
                field_name = self._eat('IDENT')[1]

                struct_def = self.symtable.lookup_struct_def(current_lhs_type_info.get('struct_def_name'))
                if not struct_def or field_name not in struct_def.struct_fields:
                    self._error(f"Field '{field_name}' not found in struct '{current_lhs_type_info.get('struct_def_name','<unknown>')}'.")

                field_info_dict = struct_def.struct_fields[field_name]

                # Add the field's offset to the base address already on the stack.
                self._emit('lit', 0, field_info_dict['offset'], f"LIT offset_of_field_{field_name}")
                self._emit('opr', 0, 2, "ADD: current_base_offset + field_offset")

                # Update the type of the expression to the type of the field.
                current_lhs_type_info = {
                    'type': field_info_dict['type'], 'type_name': field_info_dict['type_name'],
                    'base_type_name': field_info_dict.get('base_type_name'), 'array_size': field_info_dict.get('array_size'),
                    'struct_def_name': field_info_dict.get('struct_def_name')
                }

            elif self.current_token[0] == 'LBRACKET': # Array indexing
                if current_lhs_type_info['type'] != 'array':
                    self._error(f"Cannot apply array index '[]' to non-array type '{current_lhs_type_info.get('type_name','<unknown>')}'.")

                self._eat('LBRACKET')
                self._parse_expr() # Parse the index expression, leaves result on stack.
                self._eat('RBRACKET')
                
                # Emit bounds checking instruction.
                array_size = current_lhs_type_info.get('array_size')
                if array_size is not None:
                    self._emit('chk', 0, array_size, f"Check bounds for array assignment (size={array_size})")

                # Calculate the offset: index * element_size.
                element_base_type_name = current_lhs_type_info.get('base_type_name')
                element_is_struct = self.symtable.lookup_struct_def(element_base_type_name) is not None
                temp_element_type_info_for_size = {
                    'type': 'struct' if element_is_struct else 'int',
                    'type_name': element_base_type_name,
                    'struct_def_name': element_base_type_name if element_is_struct else None
                }
                element_size = self.symtable.get_type_size(temp_element_type_info_for_size)
                if element_size != 1:
                    self._emit('lit', 0, element_size, f"LIT element_size_of_{element_base_type_name}")
                    self._emit('opr', 0, 4, "MUL: idx * element_size")
                
                # Add the scaled index offset to the address on the stack.
                self._emit('opr', 0, 2, "ADD: current_base_offset + scaled_idx_offset")

                # Update the type of the expression to the type of the array element.
                current_lhs_type_info = {
                    'type': 'struct' if element_is_struct else 'int',
                    'type_name': element_base_type_name,
                    'struct_def_name': element_base_type_name if element_is_struct else None,
                    'array_size': None, 'base_type_name': None
                }
            else: break

        # After parsing the L-value, expect an assignment operator '='.
        if self.current_token[0] == 'ASSIGN':
            self._eat('ASSIGN')
            self._parse_expr() # Parse the R-value, leaves result on stack.

            level_diff = self.current_ar_level - base_sym.ar_level

            if num_selectors_parsed == 0: # Simple assignment (e.g., x = 5)
                if base_sym.type == 'array' or base_sym.type == 'struct':
                    self._error(f"Direct assignment to entire array/struct '{base_sym.name}' is not allowed. Assign to elements/fields.")
                if base_sym.kind == 'proc':
                     self._error(f"Cannot assign to a procedure '{base_sym.name}'.")
                # Use 'sto' for direct storage.
                self._emit('sto', level_diff, base_sym.addr, f"Assign to {base_sym.name} (simple)")
            else: # Complex assignment (e.g., a.b[i] = 5)
                if current_lhs_type_info.get('type') == 'array' or current_lhs_type_info.get('type') == 'struct':
                    self._error(f"Cannot assign directly to an entire array or struct L-value ('{current_lhs_type_info.get('type_name','<complex>')}') after complex access. Must be scalar for STOS.")
                # Use 'stos' (store indirect), which takes the target address from the stack.
                self._emit('stos', level_diff, 0, f"Assign to {ident_name}...(final type: {current_lhs_type_info.get('type_name')})")

        elif self.current_token[0] == 'LPAREN' and num_selectors_parsed == 0 :
             self._error(f"Identifier '{ident_name}' is a variable, not a procedure. Cannot call.")
        else:
            self._error(f"Expected '=' for assignment after L-value '{ident_name}{''.join(['...'] if num_selectors_parsed > 0 else '')}' or '(' for proc call. Got {self.current_token[0]} ('{self.current_token[1]}').")

    # Parses an if-else statement.
    def _parse_if_stmt(self):
        self._eat('IF'); self._eat('LPAREN'); self._parse_bool_expr(); self._eat('RPAREN')
        # Emit a conditional jump. The target address is unknown for now.
        jpc_to_else_or_endif_idx = self._emit('jpc', 0, "_placeholder_if_jpc", "If JPC")
        
        # Parse the 'then' block.
        self.current_block_level += 1
        self._eat('LBRACE'); self._parse_stmt_list(); self._eat('RBRACE')
        self.current_block_level -= 1

        endif_label = self._get_label("endif")

        # Check for an 'else' part.
        if self.current_token[0] == 'ELSE':
            # If there's an 'else', the 'then' block needs to jump over it.
            jmp_from_then_to_endif_idx = self._emit('jmp', 0, endif_label, "If JMP from then to endif")
            
            # The conditional jump from the 'if' now targets the 'else' block.
            else_label = self._get_label("else_part"); self._mark_label(else_label)
            self._patch_jump_address(jpc_to_else_or_endif_idx, else_label)

            # Parse the 'else' block.
            self._eat('ELSE')
            self.current_block_level += 1
            self._eat('LBRACE'); self._parse_stmt_list(); self._eat('RBRACE')
            self.current_block_level -= 1
        else:
            # If no 'else', the conditional jump targets the end of the if statement.
            self._patch_jump_address(jpc_to_else_or_endif_idx, endif_label)

        # Mark the label for the end of the entire if-else structure.
        self._mark_label(endif_label)

    # Parses a while loop.
    def _parse_while_stmt(self):
        self._eat('WHILE')
        loop_cond_label = self._get_label("while_cond")
        loop_end_label = self._get_label("while_end")

        # Mark the beginning of the loop for the condition check.
        self._mark_label(loop_cond_label)
        self._eat('LPAREN'); self._parse_bool_expr(); self._eat('RPAREN')
        # If condition is false, jump to the end of the loop.
        jpc_to_end_idx = self._emit('jpc', 0, loop_end_label, "While JPC to end")

        # Parse the loop body.
        self.current_block_level += 1
        self._eat('LBRACE'); self._parse_stmt_list(); self._eat('RBRACE')
        self.current_block_level -= 1

        # After the body, jump back to the condition check.
        self._emit('jmp', 0, loop_cond_label, "While JMP to cond")
        # Mark the end of the loop.
        self._mark_label(loop_end_label)

    # Parses an input statement.
    def _parse_input_stmt(self):
        self._eat('INPUT'); self._eat('LPAREN')
        var_name = self._eat('IDENT')[1]
        var_sym = self.symtable.lookup_symbol(var_name, self.current_scope_name, self.current_block_level, self.current_ar_level)
        if not var_sym or var_sym.kind != 'var' or var_sym.type != 'int': self._error(f"Can only input into simple int var: '{var_name}'")
        level_diff = self.current_ar_level - var_sym.ar_level
        self._emit('inp', level_diff, var_sym.addr, f"Input to {var_sym.scope_name}.{var_sym.name}")
        # Allow multiple inputs, e.g., input(a, b, c)
        while self.current_token[0] == 'COMMA':
            self._eat('COMMA'); var_name = self._eat('IDENT')[1]
            var_sym = self.symtable.lookup_symbol(var_name, self.current_scope_name, self.current_block_level, self.current_ar_level)
            if not var_sym or var_sym.kind != 'var' or var_sym.type != 'int': self._error(f"Can only input into simple int var: '{var_name}'")
            level_diff = self.current_ar_level - var_sym.ar_level
            self._emit('inp', level_diff, var_sym.addr, f"Input to {var_sym.scope_name}.{var_sym.name}")
        self._eat('RPAREN')

    # Parses an output statement.
    def _parse_output_stmt(self):
        self._eat('OUTPUT'); self._eat('LPAREN')
        self._parse_expr()
        self._emit('opr', 0, 14, "Output value"); self._emit('opr', 0, 15, "Post-output space/newline")
        # Allow multiple outputs, e.g., output(a, b, c)
        while self.current_token[0] == 'COMMA':
            self._eat('COMMA'); self._parse_expr()
            self._emit('opr', 0, 14, "Output value"); self._emit('opr', 0, 15, "Post-output space/newline")
        self._eat('RPAREN')

    # Parses a boolean expression (e.g., x > 5).
    def _parse_bool_expr(self):
        self._parse_expr() # Parse left-hand side.
        op_token_type = self.current_token[0]
        op_token_val = self.current_token[1]
        if op_token_type not in ['EQ', 'NEQ', 'LT', 'LTE', 'GT', 'GTE']: self._error(f"Expected boolean operator, got {op_token_type}")
        self._eat(op_token_type)
        self._parse_expr() # Parse right-hand side.
        # Map the token to the corresponding OPR A-value.
        op_map = {'EQ': 8, 'NEQ': 9, 'LT': 10, 'GTE': 11, 'GT': 12, 'LTE': 13}
        self._emit('opr', 0, op_map[op_token_type], f"Bool op: {op_token_val}")

    # Parses an expression with addition/subtraction.
    def _parse_expr(self):
        is_neg = False
        if self.current_token[0] == 'MINUS': self._eat('MINUS'); self._emit('lit', 0, 0); is_neg = True # Handle unary minus by doing 0 - term.
        elif self.current_token[0] == 'PLUS': self._eat('PLUS')
        self._parse_term()
        if is_neg: self._emit('opr', 0, 3, "Unary minus")
        while self.current_token[0] in ['PLUS', 'MINUS']:
            op = self.current_token[0]
            self._eat(op)
            self._parse_term()
            self._emit('opr', 0, 2 if op == 'PLUS' else 3, "Add/Sub")

    # Parses a term with multiplication/division.
    def _parse_term(self):
        self._parse_factor()
        while self.current_token[0] in ['MUL', 'DIV']:
            op = self.current_token[0]
            self._eat(op)
            self._parse_factor()
            self._emit('opr', 0, 4 if op == 'MUL' else 5, "Mul/Div")

    # Parses a factor, the base unit of an expression.
    def _parse_factor(self):
        if self.current_token[0] == 'NUMBER':
            self._emit('lit', 0, self._eat('NUMBER')[1])
        elif self.current_token[0] == 'LPAREN':
            self._eat('LPAREN'); self._parse_expr(); self._eat('RPAREN')
        elif self.current_token[0] == 'IDENT':
            ident_name = self.current_token[1]
            base_sym = self.symtable.lookup_symbol(ident_name, self.current_scope_name, self.current_block_level, self.current_ar_level)
            if not base_sym:
                 self._error(f"Undeclared identifier '{ident_name}' in factor.")

            original_base_sym_for_L_or_val = base_sym # Keep original for level diff calculation
            self._eat('IDENT')

            # If it's a procedure call, parse it as such.
            if base_sym.kind == 'proc' and self.current_token[0] == 'LPAREN':
                self._parse_func_call_expr(base_sym)
                return

            # This is an R-value access (getting the value of a variable).
            current_type_sym_info = {
                'type': base_sym.type, 'type_name': base_sym.type_name,
                'base_type_name': base_sym.base_type_name, 'array_size': base_sym.array_size,
                'struct_def_name': base_sym.struct_def_name,
            }
            num_selectors_parsed = 0

            # If it's a complex access (e.g., s.f, a[i]), first push the base address offset.
            if self.current_token[0] == 'DOT_OP' or self.current_token[0] == 'LBRACKET':
                self._emit('lit', 0, base_sym.addr, f"LIT base_var_offset_of_{base_sym.name} (for R-val)")

            # Parse selectors just like in an L-value, calculating the final address on the stack.
            while self.current_token[0] == 'DOT_OP' or self.current_token[0] == 'LBRACKET':
                num_selectors_parsed += 1
                if self.current_token[0] == 'DOT_OP':
                    # ... (logic is identical to _parse_assignment_or_call_stmt) ...
                    if current_type_sym_info['type'] != 'struct': self._error(f"Cannot apply '.field' to non-struct type '{current_type_sym_info.get('type_name','<unknown>')}'.")
                    self._eat('DOT_OP'); field_name = self._eat('IDENT')[1]
                    struct_def = self.symtable.lookup_struct_def(current_type_sym_info.get('struct_def_name'))
                    if not struct_def or field_name not in struct_def.struct_fields: self._error(f"Field '{field_name}' not found in struct '{current_type_sym_info.get('struct_def_name','<unknown>')}'.")
                    field_info_dict = struct_def.struct_fields[field_name]
                    self._emit('lit', 0, field_info_dict['offset'], f"LIT offset_of_field_{field_name}")
                    self._emit('opr', 0, 2, "ADD: current_base_offset + field_offset")
                    current_type_sym_info = {
                        'type': field_info_dict['type'], 'type_name': field_info_dict['type_name'],
                        'base_type_name': field_info_dict.get('base_type_name'), 'array_size': field_info_dict.get('array_size'),
                        'struct_def_name': field_info_dict.get('struct_def_name')
                    }
                elif self.current_token[0] == 'LBRACKET':
                    # ... (logic is identical to _parse_assignment_or_call_stmt) ...
                    if current_type_sym_info['type'] != 'array': self._error(f"Cannot apply '[]' to non-array type '{current_type_sym_info.get('type_name','<unknown>')}'.")
                    self._eat('LBRACKET'); self._parse_expr(); self._eat('RBRACKET') # Index expr on TOS
                    
                    array_size = current_type_sym_info.get('array_size')
                    if array_size is not None:
                        self._emit('chk', 0, array_size, f"Check bounds for array of size {array_size}")
                    
                    element_base_type_name = current_type_sym_info.get('base_type_name')
                    element_is_struct = self.symtable.lookup_struct_def(element_base_type_name) is not None
                    temp_element_type_info_for_size = {
                        'type': 'struct' if element_is_struct else 'int',
                        'type_name':element_base_type_name,
                        'struct_def_name': element_base_type_name if element_is_struct else None
                    }
                    element_size = self.symtable.get_type_size(temp_element_type_info_for_size)
                    if element_size != 1:
                        self._emit('lit', 0, element_size, f"LIT element_size_of_{element_base_type_name}")
                        self._emit('opr', 0, 4, "MUL: idx * element_size")
                    self._emit('opr', 0, 2, "ADD: current_base_offset + scaled_idx_offset")
                    current_type_sym_info = {
                        'type': 'struct' if element_is_struct else 'int',
                        'type_name': element_base_type_name,
                        'struct_def_name': element_base_type_name if element_is_struct else None,
                        'array_size': None, 'base_type_name': None
                    }
                else: break

            if num_selectors_parsed > 0: # Complex R-value (e.g. s.f, a[i])
                if current_type_sym_info.get('type') == 'array' or current_type_sym_info.get('type') == 'struct':
                     self._error(f"Cannot use entire array/struct value from '{ident_name}...' directly as factor.")
                # Use 'lods' (load indirect) to fetch the value from the address that was just computed on the stack.
                level_diff_for_lods = self.current_ar_level - original_base_sym_for_L_or_val.ar_level
                self._emit('lods', level_diff_for_lods, 0, f"Load value from {ident_name}... (complex R-value)")
            else: # Simple variable R-value (e.g. x)
                if original_base_sym_for_L_or_val.type == 'struct' or original_base_sym_for_L_or_val.type == 'array':
                     self._error(f"Cannot use entire struct/array '{original_base_sym_for_L_or_val.name}' directly as a value in an expression.")
                if original_base_sym_for_L_or_val.kind == 'proc':
                     self._error(f"Cannot use procedure '{original_base_sym_for_L_or_val.name}' as a variable value.")

                # Use 'lod' for a direct load from a variable's address.
                level_diff_lod = self.current_ar_level - original_base_sym_for_L_or_val.ar_level
                self._emit('lod', level_diff_lod, original_base_sym_for_L_or_val.addr, f"Load {original_base_sym_for_L_or_val.name}")
        else:
            self._error(f"Invalid factor start: {self.current_token}")

    # Parses a function call expression, which involves handling arguments.
    def _parse_func_call_expr(self, proc_sym):
        self._eat('LPAREN')

        expected_params_ordered = proc_sym.proc_param_syms_ordered
        total_scalar_values_pushed_for_all_args = 0
        arg_idx = 0

        if self.current_token[0] != 'RPAREN':
            while True:
                if arg_idx >= len(expected_params_ordered):
                    self._error(f"Too many arguments for function '{proc_sym.name}'")
                current_expected_param_sym = expected_params_ordered[arg_idx]

                # **IMPORTANT**: Handling struct arguments. Since the VM can only pass scalar values,
                # a struct argument must be "unrolled" by pushing each of its primitive fields onto the stack individually.
                if current_expected_param_sym.type == 'struct':
                    arg_base_token = self.current_token
                    if arg_base_token[0] != 'IDENT': self._error(f"Expected identifier for struct argument '{current_expected_param_sym.name}', got {arg_base_token[0]}")
                    arg_var_name = self._eat('IDENT')[1]
                    arg_var_sym = self.symtable.lookup_symbol(arg_var_name, self.current_scope_name, self.current_block_level, self.current_ar_level)
                    if not arg_var_sym: self._error(f"Undeclared identifier '{arg_var_name}' for struct argument.")

                    arg_level_diff = self.current_ar_level - arg_var_sym.ar_level
                    struct_def_for_param_type = self.symtable.lookup_struct_def(current_expected_param_sym.type_name)
                    if not struct_def_for_param_type: self._error(f"Internal: Param '{current_expected_param_sym.name}' its struct def '{current_expected_param_sym.type_name}' not found.")

                    # Case 1: The argument is a struct variable (e.g., my_func(my_struct_var))
                    if arg_var_sym.type == 'struct' and arg_var_sym.type_name == current_expected_param_sym.type_name:
                        base_addr_of_arg_struct_in_caller = arg_var_sym.addr
                        comment_arg_prefix = arg_var_name

                        # Iterate through every field of the struct and push its value onto the stack.
                        for field_name in struct_def_for_param_type.struct_field_order:
                            field_info = struct_def_for_param_type.struct_fields[field_name]

                            # If a field is an array, push every element.
                            if field_info['type'] == 'array':
                                is_array_of_structs = self.symtable.lookup_struct_def(field_info['base_type_name']) is not None
                                if is_array_of_structs: # If it's an array of structs, unroll those too.
                                    element_struct_def = self.symtable.lookup_struct_def(field_info['base_type_name'])
                                    if not element_struct_def: self._error(f"Internal: Expected struct def for array element {field_info['base_type_name']}")
                                    size_of_one_struct_element = element_struct_def.total_size

                                    for i in range(field_info['array_size']):
                                        base_addr_of_element = base_addr_of_arg_struct_in_caller + field_info['offset'] + (i * size_of_one_struct_element)
                                        for sub_field_name in element_struct_def.struct_field_order:
                                            sub_field_s_info = element_struct_def.struct_fields[sub_field_name]
                                            self._emit('lod', arg_level_diff,
                                                       base_addr_of_element + sub_field_s_info['offset'],
                                                       f"Load arg {comment_arg_prefix}.{field_name}[{i}].{sub_field_name}")
                                            total_scalar_values_pushed_for_all_args += 1
                                else: # If it's an array of primitives.
                                    size_of_one_primitive_element = 1
                                    for i in range(field_info['array_size']):
                                        addr_of_primitive_element = base_addr_of_arg_struct_in_caller + \
                                                                    field_info['offset'] + \
                                                                    (i * size_of_one_primitive_element)
                                        self._emit('lod', arg_level_diff,
                                                   addr_of_primitive_element,
                                                   f"Load arg {comment_arg_prefix}.{field_name}[{i}]")
                                        total_scalar_values_pushed_for_all_args += 1
                            elif field_info['type'] == 'struct': # If a field is another struct, unroll it recursively.
                                inner_struct_def = self.symtable.lookup_struct_def(field_info['struct_def_name'])
                                if not inner_struct_def: self._error(f"Internal: Struct def for direct field '{field_info['struct_def_name']}' not found.")
                                base_addr_of_inner_struct_field = base_addr_of_arg_struct_in_caller + field_info['offset']
                                for sub_field_name in inner_struct_def.struct_field_order:
                                    sub_field_s_info = inner_struct_def.struct_fields[sub_field_name]
                                    self._emit('lod', arg_level_diff,
                                               base_addr_of_inner_struct_field + sub_field_s_info['offset'],
                                               f"Load arg {comment_arg_prefix}.{field_name}.{sub_field_name}")
                                    total_scalar_values_pushed_for_all_args +=1
                            else: # If a field is a simple scalar.
                                self._emit('lod', arg_level_diff,
                                           base_addr_of_arg_struct_in_caller + field_info['offset'],
                                           f"Load arg {comment_arg_prefix}.{field_name}")
                                total_scalar_values_pushed_for_all_args += 1

                    # Case 2: The argument is an element of an array of structs (e.g., my_func(arr_of_structs[3]))
                    elif arg_var_sym.type == 'array' and \
                         arg_var_sym.base_type_name == current_expected_param_sym.type_name and \
                         self.current_token[0] == 'LBRACKET':
                        
                        self._eat('LBRACKET')
                        if self.current_token[0] == 'NUMBER': # This implementation only supports constant indices for this case.
                            const_idx = self._eat('NUMBER')[1]; self._eat('RBRACKET')
                            if not (0 <= const_idx < arg_var_sym.array_size):
                                 self._error(f"Array index {const_idx} out of bounds for {arg_var_sym.name}[{arg_var_sym.array_size}] as argument.")

                            size_of_one_struct_in_array = struct_def_for_param_type.total_size
                            base_addr_of_struct_element_in_caller = arg_var_sym.addr + const_idx * size_of_one_struct_in_array
                            comment_arg_prefix = f"{arg_var_name}[{const_idx}]"

                            # The logic to unroll the struct is the same as above, just with a different base address.
                            for field_name in struct_def_for_param_type.struct_field_order:
                                field_info = struct_def_for_param_type.struct_fields[field_name]
                                if field_info['type'] == 'array':
                                    # ... (code identical to the unrolling logic in Case 1) ...
                                    is_array_of_structs = self.symtable.lookup_struct_def(field_info['base_type_name']) is not None
                                    if is_array_of_structs:
                                        element_struct_def = self.symtable.lookup_struct_def(field_info['base_type_name'])
                                        if not element_struct_def: self._error(f"Internal: Expected struct def for array elem {field_info['base_type_name']}")
                                        size_of_one_struct_element = element_struct_def.total_size
                                        for i in range(field_info['array_size']):
                                            base_addr_of_element = base_addr_of_struct_element_in_caller + field_info['offset'] + (i * size_of_one_struct_element)
                                            for sub_field_name in element_struct_def.struct_field_order:
                                                sub_field_s_info = element_struct_def.struct_fields[sub_field_name]
                                                self._emit('lod', arg_level_diff,
                                                           base_addr_of_element + sub_field_s_info['offset'],
                                                           f"Load arg {comment_arg_prefix}.{field_name}[{i}].{sub_field_name}")
                                                total_scalar_values_pushed_for_all_args += 1
                                    else:
                                        size_of_one_primitive_element = 1
                                        for i in range(field_info['array_size']):
                                            addr_of_primitive_element = base_addr_of_struct_element_in_caller + \
                                                                        field_info['offset'] + \
                                                                        (i * size_of_one_primitive_element)
                                            self._emit('lod', arg_level_diff, addr_of_primitive_element, f"Load arg {comment_arg_prefix}.{field_name}[{i}]")
                                            total_scalar_values_pushed_for_all_args += 1
                                elif field_info['type'] == 'struct':
                                    # ... (code identical to the unrolling logic in Case 1) ...
                                    inner_struct_def = self.symtable.lookup_struct_def(field_info['struct_def_name'])
                                    if not inner_struct_def: self._error(f"Internal: Struct def for direct field '{field_info['struct_def_name']}' not found.")
                                    base_addr_of_inner_struct_field = base_addr_of_struct_element_in_caller + field_info['offset']
                                    for sub_field_name in inner_struct_def.struct_field_order:
                                        sub_field_s_info = inner_struct_def.struct_fields[sub_field_name]
                                        self._emit('lod', arg_level_diff,
                                                   base_addr_of_inner_struct_field + sub_field_s_info['offset'],
                                                   f"Load arg {comment_arg_prefix}.{field_name}.{sub_field_name}")
                                        total_scalar_values_pushed_for_all_args +=1
                                else: # Simple scalar field
                                    self._emit('lod', arg_level_diff,
                                               base_addr_of_struct_element_in_caller + field_info['offset'],
                                               f"Load arg {comment_arg_prefix}.{field_name}")
                                    total_scalar_values_pushed_for_all_args += 1
                        else:
                            self._error(f"Passing a struct element from array '{arg_var_sym.name}' using a variable index as a function argument is disallowed. Use a constant index or assign to a temporary local struct variable first.")
                    else:
                        self._error(f"Argument type mismatch for param '{current_expected_param_sym.name}'. Expected struct '{current_expected_param_sym.type_name}', got incompatible type from '{arg_var_name}' (type {arg_var_sym.type}, name {arg_var_sym.type_name}).")

                else: # For scalar parameters (int), just parse the expression.
                    self._parse_expr()
                    if current_expected_param_sym.type != 'int': self._error(f"Type mismatch for scalar param '{current_expected_param_sym.name}'. Expected 'int'.")
                    total_scalar_values_pushed_for_all_args += 1

                arg_idx += 1
                if self.current_token[0] == 'COMMA': self._eat('COMMA')
                else: break
        self._eat('RPAREN')

        # Verify that the correct number of arguments were provided.
        if arg_idx != len(expected_params_ordered): self._error(f"Argument count mismatch for '{proc_sym.name}'. Expected {len(expected_params_ordered)}, got {arg_idx}.")
        if total_scalar_values_pushed_for_all_args != proc_sym.proc_total_scalar_args: self._error(f"Internal consistency error: Scalar arguments pushed ({total_scalar_values_pushed_for_all_args}) for '{proc_sym.name}' does not match expected count ({proc_sym.proc_total_scalar_args}).")

        # Emit the 'cal' instruction to perform the function call.
        level_diff_cal = self.current_ar_level - proc_sym.ar_level
        cal_target_pc = proc_sym.addr
        self._emit('cal', level_diff_cal, cal_target_pc, f"Call {proc_sym.name}", nargs=proc_sym.proc_total_scalar_args)


# --- VM Interpreter ---
# Executes the P-code generated by the parser.
class VM:
    def __init__(self, code, debug_vm=False, debug_file_path=None):
        self.code = code  # The list of VM instructions.
        self.stack = [0] * 8192 # The runtime stack.
        # VM registers
        self.pc = 0 # Program Counter: points to the next instruction to execute.
        self.bp = 0 # Base Pointer: points to the base of the current activation record.
        self.sp = 0 # Stack Pointer: points to the top of the stack.
        self.running = True # Flag to control the execution loop.
        self.output_buffer = [] # Stores the output of the program.
        self.current_runtime_ar_level = 0 # Tracks the current function nesting level.
        self.level_history = [] # Stores (bp, ar_level) pairs for returning from functions.
        # Debugging utilities
        self.debug_vm = debug_vm
        self.debug_file_path = debug_file_path
        self.debug_file_handle = None
        self.input_callback = None # For integrating with a GUI's input dialog.

    # Prints the current state of the VM for debugging purposes.
    def _print_debug_state(self, event_tag, executed_instr_obj):
        if not self.debug_vm:
            return

        output_lines = []
        output_lines.append(f"\n--- VM DEBUG: {event_tag} ---")
        executed_pc = executed_instr_obj['pc']
        op = executed_instr_obj['op']
        l_val = executed_instr_obj['l']
        a_val = executed_instr_obj['a']
        comment = executed_instr_obj['comment']
        nargs_str = f" nargs:{executed_instr_obj['nargs']}" if 'nargs' in executed_instr_obj else ""

        output_lines.append(f"  Executed Instr (PC: {executed_pc:03d}): {op:<5} {str(l_val):<5} {str(a_val):<12}{nargs_str:<10}; {comment}")
        output_lines.append(f"  VM State After Exec: Next_PC={self.pc:03d}, BP={self.bp:03d}, SP={self.sp:03d}, AR_Level={self.current_runtime_ar_level}")
        output_lines.append(f"  Stack Detail:")

        # Logic to intelligently display relevant parts of the stack.
        indices_to_display = set()
        if self.bp >= 0 and self.bp < len(self.stack):
            for i in range(max(0, self.bp - 2), min(len(self.stack), self.bp + 4 + 5)):
                indices_to_display.add(i)
        if self.sp >= 0 and self.sp < len(self.stack):
            for i in range(max(0, self.sp - 7), min(len(self.stack), self.sp + 3)):
                indices_to_display.add(i)
        if self.bp == 0 and self.sp > 10:
            for i in range(0, min(len(self.stack), 5)):
                indices_to_display.add(i)
        if not indices_to_display and (self.sp > 0 or self.bp > 0):
             for i in range(0, min(len(self.stack), max(5, self.sp + 1))):
                 indices_to_display.add(i)
        elif not indices_to_display and self.sp == 0 and self.bp == 0:
             if len(self.stack) > 0: indices_to_display.add(0)

        if not indices_to_display:
            output_lines.append("    Stack pointers potentially out of typical range or stack empty for focused view.")
        else:
            sorted_indices = sorted(list(indices_to_display))
            if sorted_indices:
                last_printed_idx = -2
                for i in sorted_indices:
                    if i > last_printed_idx + 1 and last_printed_idx != -2:
                         output_lines.append(f"      ...") # Indicate a gap
                    
                    val = self.stack[i]
                    is_bp_marker = " <-- BP" if i == self.bp else ""
                    is_sp_marker = " <-- SP" if i == self.sp else ""
                    cl_marker = ""
                    # Mark the control links in the activation record.
                    if self.bp >= 0 and self.bp + 3 < len(self.stack) and self.bp <= self.sp :
                        if i == self.bp: cl_marker = " (SL)" # Static Link
                        elif i == self.bp + 1: cl_marker = " (DL)" # Dynamic Link
                        elif i == self.bp + 2: cl_marker = " (RA)" # Return Address
                        elif i == self.bp + 3: cl_marker = " (NARGS)" # Number of arguments
                    output_lines.append(f"    stack[{i:03d}]: {str(val):<10}{is_bp_marker}{is_sp_marker}{cl_marker}")
                    last_printed_idx = i
            else:
                 output_lines.append("    No specific stack indices to display.")


        output_lines.append(f"--- End VM DEBUG ({event_tag}) ---")
        full_output_string = "\n".join(output_lines) + "\n"

        if self.debug_file_handle:
            try:
                self.debug_file_handle.write(full_output_string)
            except IOError as e:
                print(f"ERROR: Failed to write to debug file '{self.debug_file_path}'. Outputting to console instead. Error: {e}")
                print(full_output_string)
                try: self.debug_file_handle.close()
                except: pass
                self.debug_file_handle = None
        else:
            print(full_output_string)

    # Calculates the base address of an activation record 'l' levels down the static link chain.
    def _get_frame_base(self, l_from_instruction):
        base = self.bp
        l = l_from_instruction
        while l > 0:
            if not (0 <= base < len(self.stack)):
                self._runtime_error(f"Invalid BP ({base}) during static link traversal (L={l_from_instruction}). PC={self.pc-1 if self.pc > 0 else 0}")
            base = self.stack[base] # Follow the static link.
            l -= 1
        if not (0 <= base < len(self.stack)):
             self._runtime_error(f"Resolved base ({base}) out of bounds. PC={self.pc-1 if self.pc > 0 else 0}")
        return base

    # Handles runtime errors and halts the VM.
    def _runtime_error(self, message):
        self.running = False
        instr_pc = self.pc -1 if self.pc > 0 else 0
        instr_detail = "N/A"
        current_instr_obj = None
        if 0 <= instr_pc < len(self.code):
            current_instr_obj = self.code[instr_pc]
            nargs_str = f" nargs:{current_instr_obj.get('nargs')}" if 'nargs' in current_instr_obj else ""
            instr_detail = f"{current_instr_obj.get('pc',-1):02d}: {current_instr_obj['op']} {current_instr_obj.get('l','')} {current_instr_obj.get('a','')}{nargs_str}"

        error_console_msg = [
            f"\n!!! VM RUNTIME ERROR !!!",
            f"  Message: {message}",
            f"  PC of Error: {instr_pc}, Instruction: {instr_detail} ; {self.code[instr_pc].get('comment','') if 0 <= instr_pc < len(self.code) else ''}",
            f"  BP: {self.bp}, SP: {self.sp}, AR_Level: {self.current_runtime_ar_level}",
            f"  Level History (caller_bp, caller_ar_level): {self.level_history}"
        ]
        print("\n".join(error_console_msg))

        if current_instr_obj:
            original_debug_vm_status = self.debug_vm
            self.debug_vm = True
            self._print_debug_state(f"RUNTIME ERROR ({message})", current_instr_obj)
            self.debug_vm = original_debug_vm_status
        else:
            print("  (Minimal stack trace due to missing instruction object at error point)")
            sp_start = max(0, self.sp - 8)
            sp_end = min(len(self.stack), self.sp + 3)
            for i in range(sp_start, sp_end):
                print(f"    stack[{i:03d}]: {self.stack[i]}")

        raise Exception(f"Runtime Error: {message}")

    # The main execution loop of the virtual machine.
    def run(self, initial_debug_vm_state=None, initial_debug_file_path=None):
        if initial_debug_vm_state is not None:
            self.debug_vm = initial_debug_vm_state
        if initial_debug_file_path is not None:
            self.debug_file_path = initial_debug_file_path

        # Reset VM state for a fresh run.
        self.pc = 0; self.bp = 0; self.sp = 0; self.current_runtime_ar_level = 0
        self.level_history = []; self.output_buffer = []
        self.running = True; max_instr_cycles = 50000; instr_count = 0

        # Open debug log file if requested.
        if self.debug_vm and self.debug_file_path:
            try:
                self.debug_file_handle = open(self.debug_file_path, 'w')
                if self.debug_file_handle:
                     self.debug_file_handle.write("--- VM DEBUG LOG START ---\n")
            except IOError as e:
                print(f"Warning: Could not open debug file '{self.debug_file_path}': {e}. Debugging to console instead.")
                self.debug_file_handle = None

        try:
            if self.debug_vm and not self.debug_file_handle :
                print("--- VM RUN START (Debugging to console at key scope changes) ---")
            elif self.debug_vm and self.debug_file_handle:
                print(f"--- VM RUN START (Debugging to file: '{self.debug_file_path}' at key scope changes) ---")


            while self.running and 0 <= self.pc < len(self.code):
                if instr_count > max_instr_cycles: self._runtime_error("Max instruction cycles reached."); break
                instr_count += 1

                executed_instr = self.code[self.pc]
                op, l_val, a_val = executed_instr['op'], executed_instr['l'], executed_instr['a']
                self.pc += 1

                # Instruction implementations
                if op == 'lit': # Push a literal value onto the stack.
                    if self.sp >= len(self.stack): self._runtime_error(f"LIT stack overflow. SP={self.sp}")
                    self.stack[self.sp] = a_val; self.sp += 1
                elif op == 'lod': # Load a value from memory onto the stack.
                    base = self._get_frame_base(l_val)
                    addr_to_load = base + a_val
                    if not (0 <= addr_to_load < len(self.stack)): self._runtime_error(f"LOD Address out of bounds: {addr_to_load}")
                    if self.sp >= len(self.stack): self._runtime_error(f"LOD stack overflow before push. SP={self.sp}")
                    self.stack[self.sp] = self.stack[addr_to_load]; self.sp += 1
                elif op == 'sto': # Store a value from the stack into memory.
                    self.sp -= 1
                    if self.sp < 0 : self._runtime_error(f"STO stack underflow. SP became {self.sp}")
                    val_to_store = self.stack[self.sp]
                    base = self._get_frame_base(l_val)
                    addr_to_store = base + a_val
                    if not (0 <= addr_to_store < len(self.stack)): self._runtime_error(f"STO Address out of bounds: {addr_to_store}")
                    self.stack[addr_to_store] = val_to_store
                elif op == 'cal': # Call a procedure.
                    nargs = executed_instr.get('nargs', 0)
                    return_address_for_callee = self.pc
                    # Find the static link (the AR base of the function's lexical parent).
                    static_link_for_callee = self._get_frame_base(l_val)
                    bp_for_callee = self.sp
                    if bp_for_callee + 3 >= len(self.stack) : self._runtime_error(f"CAL Stack overflow for SL/DL/RA/NARGS. new_bp={bp_for_callee}")
                    # Push the control links onto the stack.
                    self.stack[bp_for_callee + 0] = static_link_for_callee # SL
                    self.stack[bp_for_callee + 1] = self.bp                 # DL
                    self.stack[bp_for_callee + 2] = return_address_for_callee # RA
                    self.stack[bp_for_callee + 3] = nargs                   # NARGS
                    # Update VM state for the new function's context.
                    self.level_history.append((self.bp, self.current_runtime_ar_level))
                    self.bp = bp_for_callee
                    self.current_runtime_ar_level += 1
                    self.pc = a_val # Jump to the function's code.
                    self.sp = bp_for_callee + 4

                    if self.debug_vm:
                        func_name_hint = executed_instr['comment'].replace("Call ", "")
                        self._print_debug_state(f"After CAL (to {func_name_hint}, target PC={self.pc}), new frame links set", executed_instr)
                elif op == 'int': # Allocate space for local variables in the current frame.
                    new_sp = self.bp + a_val
                    if not (self.bp + 4 <= new_sp <= len(self.stack)):
                        self._runtime_error(f"INT Stack allocation error: BP={self.bp}, FrameSize={a_val}, NewSP={new_sp}. Out of bounds or too small.")
                    self.sp = new_sp
                    if self.debug_vm:
                        self._print_debug_state(f"After INT (Frame size from BP: {a_val})", executed_instr)
                elif op == 'jmp': # Unconditional jump.
                    if not (0 <= a_val <= len(self.code)): self._runtime_error(f"JMP to invalid PC: {a_val}")
                    self.pc = a_val
                elif op == 'jpc': # Conditional jump (if top of stack is 0).
                    self.sp -= 1
                    if self.sp < 0 : self._runtime_error(f"JPC stack underflow. SP became {self.sp}")
                    if self.stack[self.sp] == 0:
                        if not (0 <= a_val <= len(self.code)): self._runtime_error(f"JPC to invalid PC: {a_val}")
                        self.pc = a_val
                elif op == 'chk': # Array bounds check.
                    # A-value holds the declared size of the array.
                    array_size = a_val
                    # The index to be checked is on top of the stack.
                    if self.sp <= 0: self._runtime_error("Stack underflow during array bounds check.")
                    index = self.stack[self.sp - 1]

                    if not (0 <= index < array_size):
                        self._runtime_error(f"Array index out of bounds. Index was {index}, but size is {array_size}.")
                    # If check passes, the index remains on the stack for address calculation.
                elif op == 'lods': # Load Indirect.
                    if self.sp <= 0: self._runtime_error(f"LODS stack underflow (no offset on stack). SP={self.sp}")
                    # The computed offset is on the top of the stack.
                    offset_from_stack = self.stack[self.sp - 1]
                    base_for_lods = self._get_frame_base(l_val)
                    addr_to_load_lods = base_for_lods + offset_from_stack
                    if not (0 <= addr_to_load_lods < len(self.stack)): self._runtime_error(f"LODS Address out of bounds: {addr_to_load_lods}")
                    # Replace the offset on the stack with the value from the computed address.
                    self.stack[self.sp - 1] = self.stack[addr_to_load_lods]
                elif op == 'stos': # Store Indirect.
                    if self.sp <= 1: self._runtime_error(f"STOS stack underflow (need value and offset). SP={self.sp}")
                    # The value to store and the computed offset are on the stack.
                    value_to_store = self.stack[self.sp - 1]
                    offset_from_stack = self.stack[self.sp - 2]
                    self.sp -= 2
                    base_for_stos = self._get_frame_base(l_val)
                    addr_to_store_stos = base_for_stos + offset_from_stack
                    if not (0 <= addr_to_store_stos < len(self.stack)): self._runtime_error(f"STOS Address out of bounds: {addr_to_store_stos}")
                    self.stack[addr_to_store_stos] = value_to_store

                elif op == 'opr': # General operations.
                    if a_val == 0: # Return from a procedure.
                        is_main_return = self.current_runtime_ar_level == 0
                        ret_val_preview = "N/A"

                        if self.debug_vm:
                            if self.sp > 0:
                                if not is_main_return:
                                    if self.bp >= 0 and self.sp > self.bp + 3:
                                        ret_val_preview = str(self.stack[self.sp-1])
                                    else:
                                        ret_val_preview = "(Stack invalid for func return value preview)"
                                else:
                                     ret_val_preview = str(self.stack[self.sp-1])
                            self._print_debug_state(f"Before OPR 0,0 (Return initiated) RetVal_on_TOS: {ret_val_preview}", executed_instr)

                        # If returning from main, halt the program.
                        if is_main_return:
                            self.running = False
                            if self.debug_vm:
                                self._print_debug_state("After OPR 0,0 (Program Halted by main's return)", executed_instr)
                            continue

                        if self.sp <= self.bp + 3 : self._runtime_error(f"OPR 0 (Return): SP ({self.sp}) too low relative to callee BP ({self.bp}).")
                        
                        # --- Return Sequence ---
                        # 1. Save the return value (it's on top of the stack).
                        return_value = self.stack[self.sp - 1]
                        
                        # 2. Find out where the caller's stack top was before arguments were pushed.
                        if not (0 <= self.bp + 3 < len(self.stack)): self._runtime_error("OPR 0 (Return): Invalid BP to fetch NARGS.")
                        num_args_passed_to_callee = self.stack[self.bp + 3]
                        sp_caller_target_for_ret_val = self.bp - num_args_passed_to_callee
                        if sp_caller_target_for_ret_val < 0: self._runtime_error(f"OPR 0 (Return): Calculated target SP ({sp_caller_target_for_ret_val}) negative.")

                        # 3. Restore caller's PC and BP from the control links.
                        if not (0 <= self.bp + 2 < len(self.stack)): self._runtime_error("OPR 0 (Return): Invalid BP to fetch RA/DL.")
                        caller_pc = self.stack[self.bp + 2] # Restore RA
                        caller_bp = self.stack[self.bp + 1] # Restore DL
                        if not self.level_history: self._runtime_error("OPR 0 (Return): Level history empty.")
                        _og_bp, caller_ar_level = self.level_history.pop()

                        # 4. Update VM state to the caller's context.
                        self.pc = caller_pc
                        self.bp = caller_bp
                        self.current_runtime_ar_level = caller_ar_level
                        
                        # 5. Place the return value onto the caller's new stack top.
                        if not (0 <= sp_caller_target_for_ret_val < len(self.stack)):
                            self._runtime_error(f"OPR 0 (Return): Invalid target stack address ({sp_caller_target_for_ret_val}) for return value.")
                        self.stack[sp_caller_target_for_ret_val] = return_value
                        
                        # 6. Set the SP to point after the return value.
                        self.sp = sp_caller_target_for_ret_val + 1
                        
                        if self.debug_vm:
                            self._print_debug_state(f"After OPR 0,0 (Restored Caller Context, RetVal: {return_value})", executed_instr)
                    
                    # Arithmetic operations
                    elif a_val == 2: self.sp -= 1; self.stack[self.sp-1] += self.stack[self.sp] # ADD
                    elif a_val == 3: self.sp -= 1; self.stack[self.sp-1] -= self.stack[self.sp] # SUB
                    elif a_val == 4: self.sp -= 1; self.stack[self.sp-1] *= self.stack[self.sp] # MUL
                    elif a_val == 5: # DIV
                        self.sp -= 1
                        if self.stack[self.sp] == 0: self._runtime_error("Division by zero.")
                        self.stack[self.sp-1] = int(self.stack[self.sp-1] / self.stack[self.sp])
                    # Relational operations
                    elif 8 <= a_val <= 13:
                        self.sp -= 1; v2 = self.stack[self.sp]; v1 = self.stack[self.sp-1]
                        res_map = {8: v1 == v2, 9: v1 != v2, 10: v1 < v2, 11: v1 >= v2, 12: v1 > v2, 13: v1 <= v2}
                        self.stack[self.sp-1] = 1 if res_map[a_val] else 0
                    # I/O operations
                    elif a_val == 14: self.sp -= 1; self.output_buffer.append(self.stack[self.sp]) # Output
                    elif a_val == 15: pass # No-op for post-output formatting (handled by parser).
                    # Stack management
                    elif a_val == 16: # Pop unused value.
                        if self.sp > 0 : self.sp -=1
                    else: self._runtime_error(f"Unknown OPR A-value: {a_val}")
                elif op == 'inp': # Input a value.
                    base = self._get_frame_base(l_val)
                    addr_to_store_input = base + a_val
                    if not (0 <= addr_to_store_input < len(self.stack)): 
                        self._runtime_error(f"INP Address out of bounds: {addr_to_store_input}")

                    prompt_message = f"输入给 {executed_instr.get('comment','variable')}: "
                    user_input_str = None
                    final_value = 0

                    if self.input_callback: # Use GUI callback if available.
                        user_input_str = self.input_callback(prompt_message)
                        if user_input_str is None:
                            print("VM Info: 用户取消输入，默认使用 0。")
                            final_value = 0
                        else:
                            try:
                                final_value = int(user_input_str)
                            except ValueError:
                                print(f"VM Info: 无效输入 '{user_input_str}'，默认使用 0。")
                                final_value = 0
                    else: # Fallback to console input.
                        try:
                            val_str_console = input(prompt_message)
                            final_value = int(val_str_console)
                        except (ValueError, EOFError):
                            print("VM Info: 控制台输入无效或结束，默认使用 0。")
                            final_value = 0

                    self.stack[addr_to_store_input] = final_value
                else: self._runtime_error(f"Unknown instruction opcode: {op}")
        finally: # Ensures the debug log file is properly closed.
            if self.debug_file_handle:
                final_status_msg = "Halted by main's return" if not self.running and self.current_runtime_ar_level == 0 else \
                                   "Halted by runtime error" if not self.running else \
                                   "Ended (PC out of bounds?)" if self.running else "Ended"
                self.debug_file_handle.write(f"\n--- VM DEBUG LOG END ({final_status_msg} - Final PC:{self.pc}, SP:{self.sp}, BP:{self.bp}, Output: {self.output_buffer}) ---\n")
                self.debug_file_handle.close()
                self.debug_file_handle = None
            elif self.debug_vm:
                final_status_msg = "Halted by main's return" if not self.running and self.current_runtime_ar_level == 0 and self.pc > 0 else \
                                   "Halted by runtime error" if not self.running and instr_count <= max_instr_cycles else \
                                   "Ended (PC out of bounds?)" if self.running else "Ended"
                print(f"--- VM RUN END ({final_status_msg} - Final PC:{self.pc}, SP:{self.sp}, BP:{self.bp}) ---")

        return self.output_buffer
    
    # Allows a GUI to set a function for handling input requests.
    def set_input_callback(self, callback_function):
        self.input_callback = callback_function

# --- Main Execution --- 
# This block contains the example source code and the driver logic to compile and run it.
source_code = """
program BubbleSortTest {

    // Since direct array passing is not supported, we wrap it in a struct.
    // Note: The array size is fixed at compile time.
    struct IntArrayContainer {
        int data[8];
    };

    // Function 1: Prints the contents of the array.
    // It receives a struct containing the array as a parameter (pass-by-value).
    func print_array(arr_container: struct IntArrayContainer) {
        let i = 0;
        output(99999); // Output a separator for clarity.
        while (i < 8) {
            output(arr_container.data[i]);
            i = i + 1;
        };
        return 0;
    }

    // Function 2: Performs a bubble sort on the array.
    // It also receives the struct by value.
    func bubble_sort(arr_container: struct IntArrayContainer) {
        let i = 0;
        let j;
        let temp;
        let n = 8;

        while (i < n - 1) {
            j = 0;
            while (j < n - i - 1) {
                if (arr_container.data[j] > arr_container.data[j+1]) {
                    // Swap elements
                    temp = arr_container.data[j];
                    arr_container.data[j] = arr_container.data[j+1];
                    arr_container.data[j+1] = temp;
                };
                j = j + 1;
            };
            i = i + 1;
        };

        // Print the sorted array from within the function to prove the sort worked on the local copy.
        output(88888); // Output another separator.
        let k = 0;
        while (k < n) {
            output(arr_container.data[k]);
            k = k + 1;
        };

        return 0; // All functions must return a value.
    }


    main {
        let my_array: struct IntArrayContainer;

        // Initialize the array data.
        my_array.data[0] = 64;
        my_array.data[1] = 34;
        my_array.data[2] = 25;
        my_array.data[3] = 12;
        my_array.data[4] = 22;
        my_array.data[5] = 11;
        my_array.data[6] = 90;
        my_array.data[7] = 5;

        // 1. Print the original array.
        // Expected output: 99999, 64, 34, 25, 12, 22, 11, 90, 5
        print_array(my_array);

        // 2. Call the bubble sort function.
        // The function will print the sorted result internally.
        // Expected output: 88888, 5, 11, 12, 22, 25, 34, 64, 90
        bubble_sort(my_array);

        // 3. Print the original array again.
        // Since structs are passed by value, 'my_array' in main should remain unchanged.
        // Expected output: 99999, 64, 34, 25, 12, 22, 11, 90, 5
        print_array(my_array);
    }
}
"""

current_source = source_code

# The main driver loop.
try:
    print(f"--- Compiling and Running ---")
    # 1. Lexical Analysis
    my_lexer = Lexer(current_source)
    # 2. Parsing and Code Generation
    my_parser = ParserAndGenerator(my_lexer)
    generated_vm_code, final_symtable = my_parser.parse()

    # 3. Display Compilation Artifacts
    print("\nGenerated VM Code:")
    for instr in generated_vm_code:
        l_str = str(instr['l']) if instr['l'] is not None else ""
        a_str = str(instr['a']) if instr['a'] is not None else ""
        nargs_str = f" nargs:{instr['nargs']}" if 'nargs' in instr else "" 
        print(f"{instr['pc']:<3} {instr['op']:<5} {l_str:<5} {str(a_str):<12} {nargs_str:<10};{instr['comment']}")

    final_symtable.display()

    # 4. Execute the generated code on the VM
    vm_instance = VM(generated_vm_code, debug_vm=True, debug_file_path="vm_log.txt")
    vm_output = vm_instance.run()
    
    # 5. Display the final output from the VM execution.
    print(f"\n--- VM Output ---")
    if vm_output:
        for item in vm_output:
            print(f"ANS={item}")
    else:
        print("No output from VM.")

except Exception as e:
    import traceback
    print("\n--- Compilation or Setup Error / VM Runtime Error ---") 
    traceback.print_exc()
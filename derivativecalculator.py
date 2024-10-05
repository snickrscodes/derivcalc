BIN_OPS = ['+', '-', '*', '/', '^']
SPECIAL_CONST = ['e', 'pi']
UN_OPS = [
    'inv', 'neg', 'sqrt', 'cbrt', 'exp', 'ln', 'log', 'abs', 'sign', 
    'sin', 'cos', 'tan', 'csc', 'sec', 'cot', 'asin', 'acos', 'atan', 
    'acsc', 'asec', 'acot', 'sinh', 'cosh', 'tanh', 'csch', 'sech', 'erf', 
    'coth', 'asinh', 'acosh', 'atanh', 'acsch', 'asech', 'acoth', 'W'
]

def prio(op: str):
    if op in '+-':
        return 1
    elif op in '*/':
        return 2
    elif op == '^':
        return 3
    elif op in UN_OPS:
        return 4
    return 0
 
def tokenize(expression):
    tokens = []
    index = 0
    while index < len(expression):
        char = expression[index]
        if char.isdigit():
            num, spaces = find_integer(expression, index)
            tokens.append(num)
            index += spaces
        elif char.isalpha():
            cur_op, length = find_operator(expression, index, False)
            if cur_op is not None:
                tokens.append(cur_op)
                index += length
            else:
                tokens.append(char)
                index += 1
        elif char in '()+-*/^':
            tokens.append(char)
            index += 1
        else:
            index += 1
    return tokens

def find_main_operator(expr):
        min_p = 4
        index = -1
        bracket_count = 0
        for i, c in enumerate(expr):
            if c == '(':
                bracket_count += 1
            elif c == ')':
                bracket_count -= 1
            elif bracket_count == 0 and c in BIN_OPS:
                p = prio(c)
                if p <= min_p:
                    min_p = p
                    index = i
        return index

def find_operator(expression, start_index, strict_unary=True):
    last_op, last_index = None, 0
    for index in range(start_index, min(len(expression), start_index+6)):
        segment = expression[start_index:index]
        if (segment in BIN_OPS and not strict_unary) or segment in UN_OPS:
            last_op, last_index = segment, index-start_index
    return last_op, last_index

def find_integer(expression, start_index):
    num, spaces = '', 0
    for index in range(start_index, len(expression)):
        char = expression[index]
        if char.isdigit() or char == '.' or (index == start_index and char == '-'):
            num += char
            spaces += 1
        else:
            break
    return num, spaces

def find_stop_index(expression: str, start_index=0):
    open_parentheses = 0
    i = start_index
    while i < len(expression):
        char = expression[i]
        if char == '(':
            open_parentheses += 1
        elif char == ')':
            open_parentheses -= 1
        if open_parentheses == 0:
            op, length = find_operator(expression, i, True)
            if op is not None:
                i += length-1
            else:
                return i
        i += 1
    return len(expression) # if all fails

def match_parentheses(expression: str, start_index=0):
    open_parentheses = 0
    for i in range(start_index, len(expression)):
        char = expression[i]
        if char == '(':
            open_parentheses += 1
        elif char == ')':
            open_parentheses -= 1
        if open_parentheses == 0:
            return i # only stop when we have closed all parentheses
    return len(expression) # if all fails

def add_parentheses(expr):
    if len(expr) > 0 and expr[0] == '(' and match_parentheses(expr) == len(expr)-1: # if the whole expression is in parentheses
        return add_parentheses(expr[1:-1]) # then get rid of it: ((x^2+1)) -> (x^2+1)
    main_op_index = find_main_operator(expr)
    if main_op_index == -1:
        return expr
    left_expr = add_parentheses(expr[:main_op_index])
    right_expr = add_parentheses(expr[main_op_index + 1:])
    main_op = expr[main_op_index]
    return '(' + left_expr + main_op + right_expr + ')'

def process_unary_operators(expression):
    result = ''
    i = 0
    while i < len(expression):
        cur_op, length = find_operator(expression, i, True)
        if cur_op is not None:
            result += cur_op
            i += length
            # first case: an integer immediately follows like 'sinh12' -> 'sinh(12)'
            num, spaces = find_integer(expression, i)
            if num != '':
                result += '(' + num + ')'
                i += spaces-1
            elif expression[i].isalnum():
                next_op, _ = find_operator(expression, i, True)
                if next_op is None:
                    result += '(' + expression[i] + ')' # case 2: 'sinhx' -> 'sinh(x)'
                else:
                    stop_index = find_stop_index(expression[i:])
                    stop_cond = stop_index+i+1
                    new_expr = process_unary_operators(expression[i:stop_cond])
                    result += '(' + new_expr + ')'
                    i += stop_index
                
            elif expression[i] == '(': # case 3: ops in parentheses 'sinh(expression)' remains unchanged
                stop_index = match_parentheses(expression[i:]) # only look from current position forward
                if stop_index > 0: 
                    result += process_unary_operators(expression[i:i+1+stop_index]) # nested operators
                    i += stop_index
            else:
                # case 4: if the next statement immediately after is a unary operator 'sinhtanx' -> 'sinh(tan(x))
                next_op, _ = find_operator(expression, i, True)
                if next_op is not None:
                    stop_index = find_stop_index(expression[i:])
                    stop_cond = stop_index+i+1
                    new_expr = process_unary_operators(expression[i:stop_cond])
                    result += '(' + new_expr + ')'
                    i += stop_index
                else:
                    result += expression[i]
        else:
            result += expression[i]
        i += 1
    return result

def find_unary_term(expression, start_index):
    op, length = find_operator(expression, start_index, True)
    if op is None: return None, 0, None, 0
    # because unary operators have been previously formatted
    # we can assume that all of their arguments will have proper closed parentheses
    arg_index = match_parentheses(expression, start_index+length)
    return (expression[start_index:arg_index+1], arg_index-start_index, op, length)

def insert_implicit_multiplication(input: str) -> str:
    table = []
    k = 0
    expression = ''
    while k < len(input):
        term, length, op, op_length = find_unary_term(input, k)
        if term is not None:
            expression += 'F'
            table.append((op+'('+insert_implicit_multiplication(term[op_length+1:-1])+')', 'F')) # replace unary expressions with a placeholder character
            k += length
        else:
            expression += input[k]
        k += 1
    expression = insert_multiplication_simple(expression)
    expression = add_parentheses(expression) # add parentheses around binary operators
    # use the lookup table to substitute unary terms back in
    result = ''
    count = 0
    for i in range(len(expression)):
        if expression[i] == 'F':
            result += table[count][0]
            count += 1
        else:
            result += expression[i]
    return result

def insert_multiplication_simple(expression: str) -> str:
    result = ""
    length = len(expression)
    i = 0
    while i < length:
        if i < length - 1 and expression[i].isdigit() and (expression[i + 1].isalpha() or expression[i + 1] == '('):
            result += expression[i] + '*'
        elif i < length - 1 and (expression[i].isalpha() or expression[i] == ')') and expression[i + 1].isdigit():
            result += expression[i] + '*'
        elif i < length - 1 and (expression[i].isalpha() or expression[i].isdigit()) and expression[i + 1] == '(':
            result += expression[i] + '*'
        elif i < length - 1 and expression[i] == ')' and (expression[i + 1].isalpha() or expression[i + 1].isdigit()):
            result += expression[i] + '*'
        elif i < length - 1 and expression[i] == ')' and expression[i + 1] == '(':
            result += expression[i] + '*'
        elif i < length - 1 and expression[i].isalpha() and expression[i + 1].isalpha():
            result += expression[i] + '*'
        else:
            result += expression[i]
        i += 1
    return result

def convert_unary_minus(expression):
    i = 0
    result = ''
    un_flag = False
    while i < len(expression):
        if expression[i] == '-' and (i == 0 or expression[i-1] in '+-*/^(' or un_flag):
            result += 'neg'
            i += 1
            cur_op, spaces = find_operator(expression, i, True)
            if cur_op is not None:
                stop_index = find_stop_index(expression, start_index=i+spaces)
                result += '(' + cur_op + convert_unary_minus(expression[i+spaces:stop_index+1]) + ')'
                i = stop_index+1
            elif expression[i] == '(':
                end_index = match_parentheses(expression, start_index=i)
                result += convert_unary_minus(expression[i:end_index+1])
                i = end_index+1
            elif expression[i].isalpha():
                if i < len(expression)-1 and expression[i+1] == '^':
                    end_index = find_stop_index(expression, start_index=i+2)
                    result += '(' + expression[i] + '^' + expression[i+2:end_index+1] + ')'
                    i = end_index+1
                else:
                    result += '(' + expression[i] + ')'
                    i += 1
            else:
                cur_int, digits = find_integer(expression, i)
                if cur_int is not None:
                    result += '(' + cur_int + ')'
                    i += digits
        else:
            cur_op, spaces = find_operator(expression, i, True)
            un_flag = cur_op is not None
            if cur_op is not None:
                result += cur_op
                i += spaces
            elif expression[i] == '(':
                end_index = match_parentheses(expression, i)
                result += '(' + convert_unary_minus(expression[i+1:end_index]) + ')'
                i = end_index+1
            else:
                result += expression[i]
                i += 1
    return result

def parse_expr(expression: str):
    parsed = expression.replace(' ', '')
    parsed = convert_unary_minus(parsed)
    parsed = process_unary_operators(parsed)
    parsed = insert_implicit_multiplication(parsed)
    return parsed

def is_number(val):
    if val in SPECIAL_CONST:
        return True
    try:
        res = float(val)
        return True
    except:
        return False

def replace_negation_pattern(tokens, variable='x'):
    i = 0
    result = []
    while i < len(tokens):
        if (i + 3 < len(tokens) and 
            tokens[i] == 'neg' and 
            tokens[i + 1] == '(' and 
            (is_number(tokens[i + 2]) or (tokens[i + 2].isalpha() and tokens[i + 2] != variable)) and
            tokens[i + 3] == ')'):
            result.append('-' + tokens[i+2])
            i += 4
        elif (i + 2 < len(tokens) and 
            tokens[i] == '(' and 
            tokens[i + 1] == '-' and 
            (is_number(tokens[i + 2]) or (tokens[i + 2].isalpha() and tokens[i + 2] != variable)) and
            tokens[i + 3] == ')'):
            result.append('-' + tokens[i+2])
            i += 4
        else:
            result.append(tokens[i])
            i += 1
    return result

def non_numeric_const(s, variable='x'):
    if len(s) == 2:
        return s[0] == '-' and s[1].isalpha() and s[1] != variable
    elif len(s) == 1:
        return s.isalpha() and s != variable
    return False

class Node(object):
    def __init__(self, val: str):
        self.val = val # the op
        self.is_constant = is_number(val)
        self.left = None
        self.right = None

    def express(self):
        if self.is_constant:
            return self.val
        elif is_number(self.val) or (self.val.isalpha() and len(self.val) == 1):
            return self.val
        elif self.val in BIN_OPS:
            return '(' + self.left.express() + self.val + self.right.express() + ')'
        elif self.val in UN_OPS:
            if self.val == 'neg':
                return '(-(' + self.right.express() + '))'
            return self.val + '(' + self.right.express() + ')'
        else:
            return ''
    def const_with_var(self, variable='x'):
        if self.val in UN_OPS:
            self.right.const_with_var(variable)
            self.is_constant = self.right.is_constant
            if self.is_constant:
                if self.val == 'neg':
                    self.val = '(-(' + self.right.express() + '))'
                else:
                    self.val += '(' + self.right.express() + ')'
                self.right = None
        elif self.val in BIN_OPS:
            self.left.const_with_var(variable)
            self.right.const_with_var(variable)
            self.is_constant = self.left.is_constant and self.right.is_constant
            if self.is_constant:
                self.val = '(' + self.left.express() + self.val + self.right.express() + ')'
                self.left = None; self.right = None
        else:
            # check for non numeric constant
            if non_numeric_const(self.val, variable):
                self.is_constant = True
    
    def diff(self, variable='x'):
        if self.val == variable:
            return '1'
        elif self.is_constant or (self.val.isalpha() and len(self.val) == 1):
            return '0'
        elif self.val in BIN_OPS:
            if self.left.is_constant and self.right.is_constant:
                return '0'
            if prio(self.val) == 1:  # either a sum or difference of derivatives
                if self.right.is_constant:
                    return self.left.diff(variable)
                elif self.left.is_constant:
                    if self.val == '+':
                        return self.right.diff(variable)
                    else:
                        return '-' + self.right.diff(variable)
                else:
                    return '({}{}{})'.format(self.left.diff(variable), self.val, self.right.diff(variable))
            elif self.val == '*':
                if self.left.is_constant:
                    d = self.right.diff(variable)
                    if d == '1':
                        return self.left.val
                    else:
                        if self.left.val == '1':
                            return d
                        else:
                            return '({}*{})'.format(self.left.val, d)

                elif self.right.is_constant:
                    d = self.left.diff(variable)
                    if d == '1':
                        return self.right.val
                    else:
                        if self.right.val == '1':
                            return d
                        else:
                            return '({}*{})'.format(self.right.val, d)
                else:
                    d1 = self.right.diff(variable)
                    d2 = self.left.diff(variable)
                    if d1 == '1' and d2 == '1':
                        return '({}+{})'.format(self.left.express(), self.right.express())
                    elif d1 == '1':
                        return '({}*{}+{})'.format(self.right.express(), d2, self.left.express())
                    elif d2 == '1':
                        return '({}*{}+{})'.format(self.left.express(), d1, self.right.express())
                    else:
                        return '({}*{}+{}*{})'.format(self.left.express(), d1, self.right.express(), d2)
            elif self.val == '/':
                if self.left.is_constant:
                    d = self.right.diff(variable)
                    if d == '1':
                        return '(-{}/({}^2))'.format(self.left.val, self.right.express())
                    else:
                        return '(-{}*{}/({}^2))'.format(self.left.val, d, self.right.express())
                elif self.right.is_constant:
                    d = self.left.diff(variable)
                    if self.right.val == '1':
                        return d
                    else:
                        return '({}/{})'.format(d, self.right.val)
                else:
                    s = self.right.express()
                    d1 = self.left.diff(variable)
                    d2 = self.right.diff(variable)
                    if d1 == '1' and d2 == '1':
                        return '(({0}-{1})/({0}^2))'.format(s, self.left.express())
                    elif d1 == '1':
                        return '(({0}-{1}*{2})/({0}^2))'.format(s, self.left.express(), d2)
                    elif d2 == '1':
                        return '(({0}*{1}-{2})/({0}^2))'.format(s, d1, self.left.express())
                    else:
                        return '(({0}*{1}-{2}*{3})/({0}^2))'.format(s, d1, self.left.express(), d2)
                    
            elif self.val == '^':
                if self.left.is_constant:
                    if self.left.val == 'e':
                        d = self.right.diff(variable)
                        if d == '1':
                            return '(e^{})'.format(self.right.express())
                        else:
                            return '({}*e^{})'.format(d, self.right.express())
                    else:
                        if self.left.val == '1':
                            return '0'
                        d = self.right.diff(variable)
                        if d == '1':
                            return '(log({0})*{0}^{1})'.format(self.left.val, self.right.express())
                        else:
                            return '(log({0})*{1}*{0}^{2})'.format(self.left.val, d, self.right.express())
                elif self.right.is_constant:
                    if self.right.val == '1':
                        return '1'
                    elif self.right.val == '0':
                        return '0'
                    try:
                        val = float(self.right.val)
                        if val.is_integer():
                            power = str(int(val) - 1)
                        else:
                            power = str(val - 1)
                    except:
                        power = '({}-1)'.format(self.right.val)
                    d = self.left.diff(variable)
                    if (power == '1' or power == '1.0') and d == '1':
                        return '({}*{})'.format(self.right.val, self.left.express())
                    elif d == '1':
                        return '({}*{}^{})'.format(self.right.val, self.left.express(), power)
                    elif power == '1':
                        return '({}*{}*{})'.format(self.right.val, d, self.left.express())
                    else:
                        return '({}*{}*{}^{})'.format(self.right.val, d, self.left.express(), power)
                else:
                    s2 = self.left.express()
                    s3 = self.right.express()
                    d2 = self.left.diff(variable)
                    d3 = self.right.diff(variable)
                    if d2 == '1' and d3 == '1':
                        return '({1}*{0}^({1}-1)+({0}^{1})*log({0}))'.format(s2, s3)
                    elif d2 == '1':
                        return '({1}*{0}^({1}-1)+({0}^{1})*log({0})*{2})'.format(s2, s3, d3)
                    elif d3 == '1':
                        return '({1}*{0}^({1}-1)*{2}+({0}^{1})*log({0})'.format(s2, s3, d2)
                    else:
                        return '({1}*{0}^({1}-1)*{2}+({0}^{1})*log({0})*{3})'.format(s2, s3, d2, d3)
            else:
                return '0'
        elif self.val in UN_OPS:
            s = self.right.express()  # always wrapped in parentheses
            d = self.right.diff(variable)  # will also always be wrapped in parentheses
            if self.val == 'inv':
                return '(-{}/({}^2))'.format(d, s)
            elif self.val == 'neg':
                return '(-{})'.format(d)
            elif self.val == 'sqrt':
                return '({}/(2*sqrt({}))'.format(d, s)
            elif self.val == 'cbrt':
                return '({}/(3*cbrt({}^2))'.format(d, s)
            elif self.val == 'exp':
                if d == '1':
                    return 'exp({})'.format(s)
                else:
                    return '({}*exp({}))'.format(d, s)
            elif self.val == 'ln':
                return '({}/{})'.format(d, s)
            elif self.val == 'log':
                return '({}/{})'.format(d, s)
            elif self.val == 'abs':
                if d == '1':
                    return 'sign({})'.format(s)
                else:
                    return '({}*sign({}))'.format(d, s)
            elif self.val == 'sign':
                return '0'
            elif self.val == 'sin':
                if d == '1':
                    return 'cos({})'.format(s)
                else:
                    return '({}*cos({}))'.format(d, s)
            elif self.val == 'cos':
                if d == '1':
                    return '(-sin({}))'.format(s)
                else:
                    return '(-{}*sin({}))'.format(d, s)
            elif self.val == 'tan':
                if d == '1':
                    return '(sec({})^2)'.format(s)
                else:
                    return '({}*sec({})^2)'.format(d, s)
            elif self.val == 'sec':
                if d == '1':
                    return '(sec({0})*tan({0}))'.format(s)
                else:
                    return '({0}*sec({1})*tan({1}))'.format(d, s)
            elif self.val == 'csc':
                if d == '1':
                    return '(-csc({0})*cot({0}))'.format(s)
                else:
                    return '(-{0}*csc({1})*cot({1}))'.format(d, s)
            elif self.val == 'cot':
                if d == '1':
                    return '(-csc({})^2)'.format(s)
                else:
                    return '(-{}*csc({})^2)'.format(d, s)
            elif self.val == 'asin':
                return '({}/sqrt(1-{}^2))'.format(d, s)
            elif self.val == 'acos':
                return '(-{}/sqrt(1-{}^2))'.format(d, s)
            elif self.val == 'atan':
                return '({}/(1+{}^2))'.format(d, s)
            elif self.val == 'asec':
                return '({0}/(abs({1})*sqrt({1}^2-1)))'.format(d, s)
            elif self.val == 'acsc':
                return '(-{0}/(abs({1})*sqrt({1}^2-1)))'.format(d, s)
            elif self.val == 'acot':
                return '(-{}/(1+{}^2))'.format(d, s)
            elif self.val == 'sinh':
                if d == '1':
                    return 'cosh({})'.format(s)
                else:
                    return '({}*cosh({}))'.format(d, s)
            elif self.val == 'cosh':
                if d == '1':
                    return 'sinh({})'.format(s)
                else:
                    return '({}*sinh({}))'.format(d, s)
            elif self.val == 'tanh':
                if d == '1':
                    return '(sech({})^2)'.format(s)
                else:
                    return '({}*sech({})^2)'.format(d, s)
            elif self.val == 'sech':
                if d == '1':
                    return '(-sech({0})*tanh({0}))'.format(s)
                else:
                    return '(-{0}*sech({1})*tanh({1}))'.format(d, s)
            elif self.val == 'csch':
                if d == '1':
                    return '(-csch({0})*coth({0}))'.format(s)
                else:
                    return '(-{0}*csch({1})*coth({1}))'.format(d, s)
            elif self.val == 'coth':
                if d == '1':
                    return '(-csch({})^2)'.format(s)
                else:
                    return '(-{}*csch({})^2)'.format(d, s)
            elif self.val == 'asinh':
                return '({}/sqrt({}^2+1))'.format(d, s)
            elif self.val == 'acosh':
                return '({}/sqrt({}^2-1))'.format(d, s)
            elif self.val == 'atanh':
                return '({}/(1-{}^2))'.format(d, s)
            elif self.val == 'asech':
                return '(-{0}/(abs({1})*sqrt(1-{1}^2)))'.format(d, s)
            elif self.val == 'acsch':
                return '(-{0}/(abs({1})*sqrt(1+{1}^2)))'.format(d, s)
            elif self.val == 'acoth':
                return '({}/(1-{}^2))'.format(d, s)
            elif self.val == 'erf':
                return '(2*{}*exp(-{}^2)/sqrt(pi))'.format(d, s)
            elif self.val == 'W':
                return '({0}*W({1})/({1}*(1+W({1}))))'.format(d, s)
            else:
                return '0'
        
def combine(ops: list, stack: list):
    root = Node(ops.pop())
    root.right = stack.pop()
    if root.val in BIN_OPS:
        root.left = stack.pop()
    stack.append(root)

def tree(expr, var):
    s = tokenize(expr)
    s = replace_negation_pattern(s, var)
    ops, stack = [], []
    i = 0
    while i < len(s):
        token = s[i]
        if is_number(token):
            stack.append(Node(token))
        elif token == '(':
            ops.append(token)
        elif token == ')':
            while ops and ops[-1] != '(':
                combine(ops, stack)
            if ops:
                ops.pop()
        elif token in UN_OPS:
            ops.append(token)
        elif non_numeric_const(token, ''): # x is allowed too
            stack.append(Node(token))
        else:
            while ops and prio(ops[-1]) >= prio(token):
                combine(ops, stack)
            ops.append(token)
        i += 1
    while ops:
        combine(ops, stack)
    stack[0].const_with_var(var)
    return stack[0]

def diff(expression: str, variable='x'):
    expr = parse_expr(expression)
    expression_tree = tree(expr, variable)
    return expression_tree.diff(variable)

print("before using, note that all inverse functions should use their respective names")
print("inverse sine = asin, inverse hyperbolic tangent = atanh, etc")
print("powers of trig CANNOT be written as sin^a(x), the exponent must come after such as sin(x)^a")
print("lambert w function is currently not working...")
expr = input("enter an expression to differentiate: ")
variable = input("with respect to: ")
print("d(" + expr + ")/d" + variable + " = " + diff(expr, variable))

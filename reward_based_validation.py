import ast
import json
from typing import Tuple, Dict, Any, Set

def good_message(msg: str) -> str:
    return f"✅ {msg}"

def bad_message(msg: str) -> str:
    return f"❌ {msg}"

def received_reward_template(reward: float, max_reward: float) -> str:
    return f"\nReward: {reward}/{max_reward}"



def extract_function_name_and_params(response: str):
    if response == "":
        return "", [], {}
    node = ast.parse(response, mode="eval")

    # Walk through the AST to extract the function name
    class FunctionNameExtractor(ast.NodeVisitor):
        def __init__(self):
            self.function_name = None

        def visit_Call(self, node):
            # Check if the node is a function call
            if isinstance(node.func, ast.Attribute):  # Handles dot notation (e.g., module.function)
                parts = []
                current = node.func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                # Join the parts in reverse to get the full function name
                self.function_name = '.'.join(reversed(parts))
            elif isinstance(node.func, ast.Name):  # Handles simple function names (e.g., functionName)
                self.function_name = node.func.id
            # No need to visit further
            self.generic_visit(node)

    extractor = FunctionNameExtractor()
    extractor.visit(node)
    function_name = extractor.function_name
    param_names = [kw.arg for kw in node.body.keywords]
    if param_names: 
        param_values = [ast.literal_eval(kw.value) for kw in node.body.keywords]
    else:
        param_values = []

    param_values_dict = {}
    for i, param_name in enumerate(param_names):
        param_values_dict[param_name] = param_values[i]

    return function_name, param_names, param_values_dict

def get_required_and_optional_args(expected_response: dict) -> Tuple[Set[str], Set[str]]:
    expected_args = set(expected_response['arguments'].keys())
    
    # For simplicity, we'll consider all arguments as required
    # In a real implementation, you might have a schema that specifies which are required
    required_args = expected_args
    optional_args = set()
    
    return required_args, optional_args

def correct_tool_call_function_format(response: str) -> Tuple[float, float, str]:
    max_reward = 1.0
    reward = 1.0

    try:
        ast.parse(response)
    except Exception as e:
        reward = -1.0
        feedback = bad_message(f"Response was not in the correct format - {e}")
        return reward, max_reward, feedback + received_reward_template(reward, max_reward)
    
    feedback = good_message(f"Response was in the correct format.")
    return reward, max_reward, feedback + received_reward_template(reward, max_reward)

def correct_tool_call_function_name(response: str, expected_response: dict) -> Tuple[float, float, str]:
    max_reward = 3.0
    reward = 3.0    

    function_name, _, _ = extract_function_name_and_params(response)
    expected_function_name = expected_response['name']

    if function_name.strip() == expected_function_name.strip():
        feedback = good_message(f"Function name matches the expected function name.")
        return reward, max_reward, feedback + received_reward_template(reward, max_reward)
    else:
        reward = -0.5
        feedback = bad_message(f"Function name does not match the expected function name. Got '{function_name}', expected '{expected_function_name}'")
        return reward, max_reward, feedback + received_reward_template(reward, max_reward)

def correct_tool_argument_names(response: str, expected_response: dict) -> Tuple[float, float, str]:
    max_reward = 3.0

    function_name, function_args, _ = extract_function_name_and_params(response)
    provided_args = set(function_args)
    expected_args = set(expected_response['arguments'].keys())

    # no-argument case
    if not expected_args:
        if not provided_args and function_name != "":
            feedback = good_message("Function expects no arguments, and you provided none. Good job!")
            return max_reward, max_reward, feedback + received_reward_template(max_reward, max_reward)
        else:
            # If they provided extra arguments, penalize -1 per extra
            extra_args = provided_args
            penalty = len(extra_args)
            score = max_reward - penalty
            score = max(score, 0.0)  # clamp at 0
            feedback = bad_message(f"Function expects no arguments, but you provided: {sorted(extra_args)}")
            return score, max_reward, feedback + received_reward_template(score, max_reward)

    required_args, optional_args = get_required_and_optional_args(expected_response)

    # Check missing required
    missing_required = required_args - provided_args
    if missing_required:
        feedback = bad_message(f"Missing required argument(s): {sorted(missing_required)}")
        # Immediately 0 if any required param is missing
        return 0.0, max_reward, feedback + received_reward_template(0.0, max_reward)

    # At this point, all required args are present, so we only do partial penalties for extras/missing optional
    score = max_reward

    extra_args = provided_args - expected_args
    penalty_extra = len(extra_args)
    # missing optional → -1 for each
    missing_optional = optional_args - provided_args
    penalty_missing_optional = len(missing_optional)

    total_penalty = penalty_extra + penalty_missing_optional
    score -= total_penalty
    score = max(score, 0.0)  # clamp at 0

    feedback_parts = []
    if penalty_extra > 0:
        feedback_parts.append(bad_message(f"Extra argument(s): {sorted(extra_args)}"))
    if penalty_missing_optional > 0:
        feedback_parts.append(bad_message(f"Missing optional argument(s): {sorted(missing_optional)}"))
    if not feedback_parts:
        feedback_parts.append(good_message("All required and optional arguments are present, and no extras. Good job!"))

    feedback = "\n".join(feedback_parts)
    return score, max_reward, feedback + received_reward_template(score, max_reward)

def correct_tool_argument_values(response: str, expected_response: dict) -> Tuple[float, float, str]:
    max_reward = 3.0

    function_name, function_args, function_values = extract_function_name_and_params(response)
    provided_args = set(function_args)
    expected_args = set(expected_response['arguments'].keys())
    required_args, optional_args = get_required_and_optional_args(expected_response)

    feedback_lines = []
    correct_count = 0

    def is_value_correct(arg: str) -> bool:
        expected_val = expected_response['arguments'][arg]
        provided_val = function_values.get(arg)

        if provided_val is None:
            # Should not happen if names-check is done first, but let's be safe:
            return False

        # For nested dictionaries, convert to string for comparison
        if isinstance(provided_val, dict) and isinstance(expected_val, dict):
            return str(provided_val) == str(expected_val)
        
        # For simple values, direct comparison
        return provided_val == expected_val

    for arg in expected_args:
        if arg in provided_args:
            if is_value_correct(arg):
                correct_count += 1
                feedback_lines.append(good_message(f"Correct value for '{arg}'."))
            else:
                if arg in required_args:
                    feedback_lines.append(bad_message(
                        f"Incorrect value for required argument: {arg}. "
                        f"Expected: {expected_response['arguments'][arg]}, got: {function_values.get(arg)}"
                    ))
                    feedback = "\n".join(feedback_lines)
                    return 0.0, max_reward, feedback + received_reward_template(0.0, max_reward)
                else:
                    # optional arg is incorrect, just note it; we don't zero out the score
                    feedback_lines.append(bad_message(
                        f"Incorrect value for optional argument: {arg}. "
                        f"Expected: {expected_response['arguments'][arg]}, got: {function_values.get(arg)}"
                    ))
        else:
            feedback_lines.append(bad_message(f"Argument not provided: {arg}"))

    score = max_reward * (correct_count / len(expected_args)) if expected_args else max_reward
    feedback = "\n".join(feedback_lines)
    return score, max_reward, feedback + received_reward_template(score, max_reward)

def validate_tool_call(tool_call_json: str, function_call: str) -> Dict[str, Any]:
    """
    Validates a tool call against a function call response.
    
    Args:
        tool_call_json: JSON string with the tool call specification
        function_call: String with the function call response
        
    Returns:
        Dictionary with validation results
    """
    try:
        tool_call = json.loads(tool_call_json)
        expected_response = tool_call["content"]
        
        # Step 1: Check function format
        format_reward, format_max, format_feedback = correct_tool_call_function_format(function_call)
        # If format is invalid, return early
        if format_reward < 0:
            return {
                "valid": False,
                "total_reward": format_reward,
                "max_reward": format_max,
                "feedback": format_feedback,
                "details": {
                    "format": {"reward": format_reward, "max": format_max, "feedback": format_feedback}
                }
            }
        
        # Step 2: Check function name
        name_reward, name_max, name_feedback = correct_tool_call_function_name(function_call, expected_response)
        # If name is incorrect, return early
        if name_reward < 0:
            total_reward = format_reward + name_reward
            max_reward = format_max + name_max
            return {
                "valid": False,
                "total_reward": total_reward,
                "max_reward": max_reward,
                "feedback": format_feedback + "\n" + name_feedback,
                "details": {
                    "format": {"reward": format_reward, "max": format_max, "feedback": format_feedback},
                    "name": {"reward": name_reward, "max": name_max, "feedback": name_feedback}
                }
            }
        
        # Step 3: Check argument names
        args_reward, args_max, args_feedback = correct_tool_argument_names(function_call, expected_response)
        # If argument names are incorrect, return early
        if args_reward <= 0:
            total_reward = format_reward + name_reward + args_reward
            max_reward = format_max + name_max + args_max
            return {
                "valid": False,
                "total_reward": total_reward,
                "max_reward": max_reward,
                "feedback": format_feedback + "\n" + name_feedback + "\n" + args_feedback,
                "details": {
                    "format": {"reward": format_reward, "max": format_max, "feedback": format_feedback},
                    "name": {"reward": name_reward, "max": name_max, "feedback": name_feedback},
                    "args": {"reward": args_reward, "max": args_max, "feedback": args_feedback}
                }
            }
        
        # Step 4: Check argument values
        values_reward, values_max, values_feedback = correct_tool_argument_values(function_call, expected_response)
        # Calculate total reward
        total_reward = format_reward + name_reward + args_reward + values_reward
        max_reward = format_max + name_max + args_max + values_max
        
        return {
            "valid": total_reward > 0 and values_reward > 0,
            "total_reward": total_reward,
            "max_reward": max_reward,
            "feedback": format_feedback + "\n" + name_feedback + "\n" + args_feedback + "\n" + values_feedback,
            "details": {
                "format": {"reward": format_reward, "max": format_max, "feedback": format_feedback},
                "name": {"reward": name_reward, "max": name_max, "feedback": name_feedback},
                "args": {"reward": args_reward, "max": args_max, "feedback": args_feedback},
                "values": {"reward": values_reward, "max": values_max, "feedback": values_feedback}
            }
        }
        
    except Exception as e:
        return {
            "valid": False,
            "total_reward": -1.0,
            "max_reward": 10.0,
            "feedback": bad_message(f"Error validating tool call: {str(e)}"),
            "details": {}
        }

# Example usage
if __name__ == "__main__":
    # Example tool call JSON
    tool_call_json = '''{"role": "tool call", "content": {"name": "debug_document", "arguments": {"speed": 3957, "has_permission": false, "is_verified": true, "options": {"anatomy": "plateau"}}}}'''
    
    # Example function call (correct)
    # correct_function_call = 'calculate_gpa(grades=["A", "B", "A", "C"], credit_hours=[3, 4, 3, 2])'
    correct_function_call = "retrieve_analysis(permissions=['avalanche forecasting', 'marsh dynamics'])"
    
    # Example function call (incorrect name)
    incorrect_name_function_call = '''debug_doc(speed=3957, has_permission=false, is_verified=true, options={"anatomy": "plateau"})'''
    
    # Example function call (missing argument)
    missing_arg_function_call = '''debug_document(speed=3957, has_permission=false, options={"anatomy": "plateau"})'''
    
    # Example function call (incorrect value)
    incorrect_value_function_call = '''debug_document(speed=1000, has_permission=false, is_verified=true, options={"anatomy": "plateau"})'''
    
    # Validate the correct function call
    result = validate_tool_call(tool_call_json, correct_function_call)
    print("Correct function call validation:")
    print(json.dumps(result, indent=2))
    
    # # Validate the incorrect name function call
    # result = validate_tool_call(tool_call_json, incorrect_name_function_call)
    # print("\nIncorrect name function call validation:")
    # print(json.dumps(result, indent=2))
    
    # # Validate the missing argument function call
    # result = validate_tool_call(tool_call_json, missing_arg_function_call)
    # print("\nMissing argument function call validation:")
    # print(json.dumps(result, indent=2))
    
    # # Validate the incorrect value function call
    # result = validate_tool_call(tool_call_json, incorrect_value_function_call)
    # print("\nIncorrect value function call validation:")
    # print(json.dumps(result, indent=2))